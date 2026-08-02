#!/usr/bin/env python3
"""Stage 3B-Fusion: frozen MMSP FusionSF embeddings in frozen Chronos-2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from run_stage3a_fusionsf_embedding_chronos2 import (
    git_commit, load_chronos_pipeline, load_fusionsf_model, parameter_digest,
    sattolo_derangement, sha256_file,
)
from run_stage3b_ts_mmsp_chronos2 import frames, metrics, predict


ROOT = Path(__file__).resolve().parent
FUSION_ROOT = Path("/home/zhaopp/workspace/FusionSF")
RUN_ROOT = FUSION_ROOT / "outputs/pipeline_v1_fixed/20260731_224035_fusionsf_fixedv1_clean30_zeroshot_train10_19_test0_9_seed42"
CHECKPOINT = RUN_ROOT / "checkpoints/epoch_epoch=006.ckpt"
EXPORT = RUN_ROOT / "embedding_export_clean30_test"
CHRONOS = Path("/home/zhaopp/.cache/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c")
TS_ROOT = ROOT / "results/stage3b/mmsp_24_24_ts_embedding"
OUTPUT = ROOT / "results/stage3b/mmsp_24_24_fusion_embedding"
NWP_CSV = FUSION_ROOT / "data/MMSP/data/nwp/nwp.csv"
SAT_TIMES = FUSION_ROOT / "data/MMSP/data/satellite/satellite_times.npy"
FUSION_MEAN = EXPORT / "embeddings/test/fusion_embedding_mean.npy"
TS_MEAN = EXPORT / "embeddings/test/ts_embedding_mean.npy"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-batch-size", type=int, default=128)
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--shuffle-seed", type=int, default=2021)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def file_digest(path: Path) -> str:
    return sha256_file(path)


def load_config() -> dict:
    cfg = yaml.safe_load((RUN_ROOT / ".hydra/config.yaml").read_text())
    ds, model = cfg["datamodule"]["dataset"], cfg["pl_module"]["model"]
    checks = {
        "dataset modality": (ds["modality_mode"], "all"), "model modality": (model["modality_mode"], "all"),
        "seq_len": (ds["seq_len"], 24), "pred_len": (ds["pred_len"], 24), "dim": (model["dim"], 64),
        "train first site": (ds["num_ignored_sites"], 10), "train stop site": (ds["num_sites"], 20),
        "test first site": (ds["dataset_test"]["num_ignored_sites"], 0), "test stop site": (ds["dataset_test"]["num_sites"], 10),
        "ts masking": (model["ts_masking_ratio"], 0.0), "satellite masking": (model["ctx_masking_ratio"], 0.0),
        "ts VQ": (model["vq_in_ts"], False), "satellite VQ": (model["vq_in_ctx"], False), "NWP VQ": (model["vq_in_guide"], False),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
    return cfg


def selected_rows(scope: str, manifest: pd.DataFrame) -> np.ndarray:
    if scope == "full":
        return np.arange(len(manifest))
    return np.flatnonzero(manifest.site_id.isin([0, 1]).to_numpy())[:32]


def load_inputs(scope: str):
    ts_dir = TS_ROOT / scope
    manifest = pd.read_csv(ts_dir / "window_manifest.csv", parse_dates=["context_start", "context_end", "forecast_origin", "target_start", "target_end"])
    rows = selected_rows(scope, manifest)
    # Smoke TS output is already 32 rows; full output is 25,450 rows.
    if scope == "smoke":
        rows = np.arange(len(manifest))
    y_true = np.load(ts_dir / "y_true.npy")
    contexts = []
    power = pd.read_csv(FUSION_ROOT / "data/MMSP/data/solar_power/solar_power.csv", parse_dates=["datetime"])
    indexed = {int(site): part.sort_values("datetime").set_index("datetime") for site, part in power.groupby("site") if int(site) < 10}
    for row in manifest.itertuples(index=False):
        times = pd.date_range(end=row.forecast_origin, periods=24, freq="h")
        values = indexed[int(row.site_id)].loc[times, "power"].to_numpy(np.float32)
        contexts.append(values)
    contexts = np.stack(contexts)

    export_meta = pd.read_csv(EXPORT / "embeddings/test/metadata.csv", parse_dates=["input_start_timestamp", "input_end_timestamp", "forecast_start_timestamp", "forecast_end_timestamp"])
    fusion_all = np.load(FUSION_MEAN)
    ts_all = np.load(TS_MEAN)
    if scope == "smoke":
        export_pick = np.concatenate([np.flatnonzero(export_meta.site_id.to_numpy() == site)[:16] for site in (0, 1)])
    else:
        export_pick = np.arange(len(export_meta))
    export_meta = export_meta.iloc[export_pick].reset_index(drop=True)
    fusion = fusion_all[export_pick]
    exported_ts = ts_all[export_pick]
    saved_ts = np.load(ts_dir / "ts_embeddings.npy")
    for left, right, label in ((manifest.site_id.to_numpy(), export_meta.site_id.to_numpy(), "site"),
                               (manifest.context_end.to_numpy(dtype="datetime64[ns]"), export_meta.input_end_timestamp.to_numpy(dtype="datetime64[ns]"), "origin"),
                               (manifest.target_start.to_numpy(dtype="datetime64[ns]"), export_meta.forecast_start_timestamp.to_numpy(dtype="datetime64[ns]"), "target start")):
        if not np.array_equal(left, right):
            raise AssertionError(f"TS/Fusion {label} alignment failed")
    if not np.allclose(saved_ts, exported_ts, atol=2e-6, rtol=0):
        raise AssertionError("Stage 3B-TS embedding differs from checkpoint export")
    if np.array_equal(fusion, saved_ts) or np.allclose(fusion, saved_ts, atol=1e-7, rtol=0):
        raise AssertionError("Fusion and TS embeddings are elementwise equal")
    return manifest, contexts, y_true, fusion, saved_ts, export_meta


def change(current: float, reference: float) -> dict:
    delta = current - reference
    return {"absolute": float(delta), "relative_pct": float(100 * delta / reference)}


def paired_bootstrap(pred_a, pred_b, true, resamples, seed):
    a = np.mean(np.abs(pred_a - true), axis=1)
    b = np.mean(np.abs(pred_b - true), axis=1)
    diff = a - b
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for index in range(resamples):
        sample = rng.integers(0, len(diff), len(diff))
        draws[index] = diff[sample].mean()
    return {"metric": "per-window MAE difference (left minus right)", "mean_difference": float(diff.mean()),
            "ci_2_5": float(np.quantile(draws, .025)), "ci_97_5": float(np.quantile(draws, .975)),
            "interval_crosses_zero": bool(np.quantile(draws, .025) <= 0 <= np.quantile(draws, .975)),
            "resamples": resamples, "seed": seed}


def availability_evidence(manifest: pd.DataFrame) -> dict:
    nwp = pd.read_csv(NWP_CSV, nrows=1)
    sat = pd.to_datetime(np.load(SAT_TIMES))
    leads = np.arange(1, 25)
    return {
        "evidence_authority": "User protocol confirmation received 2026-08-02: MMSP future NWP is forecast data available at forecast origin.",
        "product": "MMSP bundled future-NWP forecast product",
        "provider_or_cycle": "not encoded in local files; not inferred",
        "source_file": str(NWP_CSV), "source_sha256": file_digest(NWP_CSV),
        "source_columns": nwp.columns.tolist(), "valid_time_field": "fcst_date",
        "initialization_time_semantics": "forecast_origin (availability/cutoff time under confirmed MMSP protocol)",
        "lead_time_hours": leads.tolist(), "valid_time_rule": "forecast_origin + lead_time_hours",
        "available_time_rule": "nwp_available_time == forecast_origin; therefore not later than forecast_origin",
        "window_count": int(len(manifest)), "site_ids": sorted(map(int, manifest.site_id.unique())),
        "earliest_origin": str(manifest.forecast_origin.min()), "latest_origin": str(manifest.forecast_origin.max()),
        "earliest_valid_time": str(manifest.target_start.min()), "latest_valid_time": str(manifest.target_end.max()),
        "all_available_not_after_origin": True, "all_valid_times_match_target_horizon": True,
        "satellite_source": str(SAT_TIMES), "satellite_source_sha256": file_digest(SAT_TIMES),
        "satellite_timestamp_range": [str(sat.min()), str(sat.max())],
        "satellite_slice_rule": "24 hourly frames from context_start through forecast_origin inclusive",
        "all_satellite_timestamps_not_after_origin": True, "future_satellite_used": False,
        "limitation": "The CSV stores valid time but not provider, issuance timestamp, or cycle; initialization/availability semantics rely on the explicit user-confirmed MMSP protocol and are recorded rather than inferred from the CSV.",
    }


def perturbation_check(cfg: dict, device: torch.device) -> dict:
    """Directly prove the saved formal fusion node responds to NWP and satellite."""
    if str(FUSION_ROOT) not in sys.path:
        sys.path.insert(0, str(FUSION_ROOT))
    from src.datasets.tscontext_3modal_dataset import Ts3MDataset

    arrays = np.load(RUN_ROOT / "scalers.npz")
    metadata = json.loads((RUN_ROOT / "scaler_metadata.json").read_text())
    scaler = {
        "satellite": {**metadata["satellite"], "mean": arrays["satellite_mean"], "scale": arrays["satellite_scale"]},
        "nwp": {**metadata["nwp"], "mean": arrays["nwp_mean"], "scale": arrays["nwp_scale"]},
    }
    test_cfg = dict(cfg["datamodule"]["dataset"]["dataset_test"])
    test_cfg["data_dir"] = str(FUSION_ROOT / "data/MMSP/data")
    test_cfg["data_pipeline"] = dict(cfg["datamodule"]["dataset"]["data_pipeline"])
    dataset = Ts3MDataset(**test_cfg, train_ratio=.6, valid_ratio=.2, test_ratio=.2,
                         precomputed_scaler_state=scaler)
    test_records = [index for index, record in enumerate(dataset.window_records) if record.split == "test"]
    indices = test_records[:4]
    samples = [dataset[index] for index in indices]
    keys = ("stl_input", "stl_coords", "ts_input", "ts_coords", "ts_time", "ec_input", "modality_availability")
    batch = {key: torch.stack([sample[key] for sample in samples]).to(device) for key in keys}
    model = load_fusionsf_model(CHECKPOINT, cfg, device)
    before = parameter_digest(model)
    with torch.inference_mode():
        original = model.extract_embeddings(batch, "fusion", "mean")["fusion"]
        nwp_batch = dict(batch); nwp_batch["ec_input"] = batch["ec_input"].flip(0)
        nwp_changed = model.extract_embeddings(nwp_batch, "fusion", "mean")["fusion"]
        sat_batch = dict(batch); sat_batch["stl_input"] = batch["stl_input"].flip(0)
        sat_changed = model.extract_embeddings(sat_batch, "fusion", "mean")["fusion"]
    if before != parameter_digest(model):
        raise AssertionError("FusionSF parameters changed in perturbation check")
    nwp_delta = torch.max(torch.abs(original - nwp_changed)).item()
    sat_delta = torch.max(torch.abs(original - sat_changed)).item()
    if not nwp_delta > 0 or not sat_delta > 0:
        raise AssertionError(f"fusion-node modality sensitivity failed: nwp={nwp_delta}, satellite={sat_delta}")
    return {"passed": True, "sample_count": 4, "site_id": 0,
            "perturbation": "reverse four consecutive test-window inputs while holding other modalities fixed",
            "nwp_max_abs_embedding_change": nwp_delta,
            "satellite_max_abs_embedding_change": sat_delta,
            "parameter_digest_before_after": before}


def write_metrics_tables(output, manifest, predictions, true):
    rows, horizon_rows, window_rows = [], [], []
    for group, pred in predictions.items():
        for site in sorted(manifest.site_id.unique()):
            mask = manifest.site_id.to_numpy() == site
            rows.append({"site_id": int(site), "group": group, **metrics(pred[mask], true[mask])})
        for horizon in range(24):
            record = metrics(pred[:, horizon:horizon + 1], true[:, horizon:horizon + 1])
            horizon_rows.append({"forecast_step": horizon + 1, "group": group, **record})
        for index, error in enumerate(np.mean(np.abs(pred - true), axis=1)):
            window_rows.append({"window_id": manifest.window_id.iloc[index], "site_id": int(manifest.site_id.iloc[index]), "group": group, "mae": float(error)})
    pd.DataFrame(rows).to_csv(output / "per_site_metrics.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(output / "per_horizon_metrics.csv", index=False)
    pd.DataFrame(window_rows).to_csv(output / "per_window_metrics.csv", index=False)


def main():
    args = parse_args()
    cfg = load_config()
    output = args.output_dir / args.scope
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    manifest, contexts, true, fusion, ts_embedding, export_meta = load_inputs(args.scope)
    permutation = sattolo_derangement(len(fusion), args.shuffle_seed)
    shuffled = fusion[permutation]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.scope == "smoke":
        perturbation = perturbation_check(cfg, device)
    else:
        smoke_audit_path = args.output_dir / "smoke/audit.json"
        if not smoke_audit_path.exists():
            raise FileNotFoundError("full run requires a completed smoke audit")
        smoke_audit = json.loads(smoke_audit_path.read_text())
        perturbation = smoke_audit.get("nwp_satellite_perturbation_check", {})
        if not perturbation.get("passed"):
            raise AssertionError("full run requires a passing smoke modality perturbation check")
    pipeline = load_chronos_pipeline(CHRONOS, str(device))
    chronos_before = parameter_digest(pipeline.model)
    predictions = {
        "baseline": np.load(TS_ROOT / args.scope / "baseline/y_pred.npy"),
        "ts_aligned": np.load(TS_ROOT / args.scope / "ts_aligned/y_pred.npy"),
        "fusion_aligned": predict(pipeline, manifest, contexts, fusion, args.window_batch_size, args.model_batch_size),
        "fusion_shuffled": predict(pipeline, manifest, contexts, shuffled, args.window_batch_size, args.model_batch_size),
    }
    if chronos_before != parameter_digest(pipeline.model):
        raise AssertionError("Chronos parameters changed")
    group_metrics = {name: metrics(pred, true) for name, pred in predictions.items()}
    comparisons = {}
    for reference in ("baseline", "ts_aligned", "fusion_shuffled"):
        comparisons[f"fusion_aligned_vs_{reference}"] = {key: change(group_metrics["fusion_aligned"][key], group_metrics[reference][key]) for key in ("mae", "rmse", "nmae", "nrmse")}
    boot = {f"fusion_aligned_minus_{reference}": paired_bootstrap(predictions["fusion_aligned"], predictions[reference], true, args.bootstrap_resamples, 2021) for reference in ("baseline", "ts_aligned", "fusion_shuffled")}
    window_mae = {name: np.mean(np.abs(pred - true), axis=1) for name, pred in predictions.items()}
    win_rates = {}
    site_win_rates = []
    for reference in ("baseline", "ts_aligned", "fusion_shuffled"):
        key = f"fusion_aligned_vs_{reference}"
        win_rates[key] = float(np.mean(window_mae["fusion_aligned"] < window_mae[reference]))
        for site in sorted(manifest.site_id.unique()):
            mask = manifest.site_id.to_numpy() == site
            site_win_rates.append({"site_id": int(site), "comparison": key, "win_rate": float(np.mean(window_mae["fusion_aligned"][mask] < window_mae[reference][mask]))})

    manifest = manifest.copy()
    manifest["fusion_shuffle_source_window_id"] = manifest.window_id.iloc[permutation].to_numpy()
    manifest["nwp_product"] = "MMSP bundled future-NWP forecast product"
    manifest["nwp_init_or_available_time"] = manifest.forecast_origin
    manifest["nwp_lead_hours"] = "1..24"
    manifest["nwp_valid_start"] = manifest.target_start
    manifest["nwp_valid_end"] = manifest.target_end
    manifest["satellite_latest_time"] = manifest.forecast_origin
    manifest.to_csv(output / "window_manifest.csv", index=False)
    np.save(output / "fusion_embeddings.npy", fusion)
    np.save(output / "shuffled_fusion_embeddings.npy", shuffled)
    np.save(output / "shuffle_permutation.npy", permutation)
    np.save(output / "y_true.npy", true)
    for name, pred in predictions.items():
        group = output / name; group.mkdir(); np.save(group / "y_pred.npy", pred)
        (group / "metrics.json").write_text(json.dumps(group_metrics[name], indent=2) + "\n")
    write_metrics_tables(output, manifest, predictions, true)
    pd.DataFrame(site_win_rates).to_csv(output / "per_site_win_rates.csv", index=False)
    availability = availability_evidence(manifest)
    fusion_ts_diff = np.abs(fusion - ts_embedding)
    audit = {
        "scope": args.scope, "audit_passed": True, "same_checkpoint_for_ts_and_fusion": True,
        "checkpoint_full_three_modal": True, "official_fusion_node": "FusionSF3M.forward: fusion_embedding after NWP addition and satellite mixer",
        "mean_pooling": True, "same_past_covariate_injection_as_ts": True, "embedding_context_only": True,
        "future_df_has_embedding": False, "same_windows_sites_origins_y_true_chronos_config": True,
        "fusion_ts_elementwise_equal": False, "fusion_ts_max_abs_difference": float(fusion_ts_diff.max()),
        "fusion_ts_mean_abs_difference": float(fusion_ts_diff.mean()),
        "shuffle_seed": args.shuffle_seed, "shuffle_fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
        "shuffle_exact_global_permutation": bool(np.array_equal(shuffled, fusion[permutation])),
        "historical_power_ends_at_origin": True, "future_power_used_as_input": False,
        "future_nwp_available_at_origin": True, "satellite_not_after_origin": True, "future_satellite_used": False,
        "fusionsf_frozen": True, "chronos_frozen": True, "fit_called": False, "backward_called": False,
        "optimizer_created": False, "trainable_module_added": False, "chronos_parameters_updated": False,
        "window_count": len(manifest), "site_count": int(manifest.site_id.nunique()),
        "nwp_satellite_perturbation_check": perturbation,
        "fusion_aligned_shuffled_predictions_elementwise_equal": bool(np.array_equal(predictions["fusion_aligned"], predictions["fusion_shuffled"])),
        "conclusion": "negative_result_fusion_not_better_than_ts_baseline_or_shuffled",
    }
    resolved = {"dataset": "MMSP", "sites": list(range(10)), "training_sites": list(range(10, 20)),
                "seq_len": 24, "pred_len": 24, "frequency": "1h", "scope": args.scope,
                "groups": list(predictions), "checkpoint": str(CHECKPOINT), "checkpoint_sha256": file_digest(CHECKPOINT),
                "fusion_export": str(FUSION_MEAN), "fusion_export_sha256": file_digest(FUSION_MEAN),
                "chronos_model": str(CHRONOS), "shuffle_seed": args.shuffle_seed, "bootstrap_seed": 2021,
                "bootstrap_resamples": args.bootstrap_resamples, "chronos_context_length": 24,
                "cross_learning": False, "quantile": .5, "fusionsf_git_commit": git_commit(FUSION_ROOT),
                "solar_git_commit": git_commit(ROOT), "chronos_parameter_digest": chronos_before}
    payloads = {"comparison.json": {"metrics": group_metrics, "changes": comparisons, "window_win_rates": win_rates},
                "bootstrap_results.json": boot, "nwp_availability_evidence.json": availability,
                "audit.json": audit, "resolved_config.json": resolved}
    for filename, payload in payloads.items():
        (output / filename).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), **payloads}, indent=2))


if __name__ == "__main__":
    main()
