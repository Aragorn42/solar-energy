#!/usr/bin/env python3
"""Stage 3B-TS: frozen MMSP TS embeddings as Chronos-2 past covariates."""

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


ROOT = Path(__file__).resolve().parent
FUSION_ROOT = Path("/home/zhaopp/workspace/FusionSF")
CHECKPOINT = FUSION_ROOT / "outputs/pipeline_v1_fixed/20260731_224035_fusionsf_fixedv1_clean30_zeroshot_train10_19_test0_9_seed42/checkpoints/epoch_epoch=006.ckpt"
POWER_CSV = FUSION_ROOT / "data/MMSP/data/solar_power/solar_power.csv"
CHRONOS = Path("/home/zhaopp/.cache/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c")
OUTPUT = ROOT / "results/stage3b/mmsp_24_24_ts_embedding"
EMBED_COLS = [f"ts_emb_{i:03d}" for i in range(64)]


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=("smoke", "full"), required=True)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--embedding-batch-size", type=int, default=128)
    p.add_argument("--window-batch-size", type=int, default=128)
    p.add_argument("--model-batch-size", type=int, default=64)
    p.add_argument("--shuffle-seed", type=int, default=2021)
    p.add_argument("--bootstrap-resamples", type=int, default=2000)
    return p.parse_args()


def load_config() -> dict:
    cfg = yaml.safe_load((CHECKPOINT.parent.parent / ".hydra/config.yaml").read_text())
    ds, model = cfg["datamodule"]["dataset"], cfg["pl_module"]["model"]
    required = {
        "modality": (ds["modality_mode"], "all"), "seq": (ds["seq_len"], 24),
        "pred": (ds["pred_len"], 24), "train_first": (ds["num_ignored_sites"], 10),
        "train_stop": (ds["num_sites"], 20), "test_first": (ds["dataset_test"]["num_ignored_sites"], 0),
        "test_stop": (ds["dataset_test"]["num_sites"], 10), "dim": (model["dim"], 64),
        "ts_length": (model["ts_length"], 24), "ts_mask": (model["ts_masking_ratio"], 0.0),
        "vq_ts": (model["vq_in_ts"], False),
    }
    for name, (actual, expected) in required.items():
        if actual != expected:
            raise AssertionError(f"{name}: {actual!r} != {expected!r}")
    return cfg


def build_manifest(scope: str):
    df = pd.read_csv(POWER_CSV, parse_dates=["datetime"])
    selected = df[df.site.between(0, 9)].copy().sort_values(["site", "datetime"])
    rows, contexts, targets = [], [], []
    for site, part in selected.groupby("site", sort=True):
        part = part.reset_index(drop=True)
        times = pd.DatetimeIndex(part.datetime)
        time_values = times.to_numpy(dtype="datetime64[ns]")
        if times.has_duplicates or not times.is_monotonic_increasing or np.any(np.diff(time_values) != np.timedelta64(1, "h")):
            raise AssertionError(f"site {site} is not a continuous hourly series")
        test_lower = times[int(len(times) * 0.8)]
        site_rows = []
        for start in range(len(part) - 48 + 1):
            target_start, target_end = times[start + 24], times[start + 47]
            if target_start < test_lower:
                continue
            site_rows.append((start, target_start, target_end))
        if scope == "smoke":
            site_rows = site_rows[:16] if site in (0, 1) else []
        for start, target_start, target_end in site_rows:
            hist = part.iloc[start:start + 24]
            future = part.iloc[start + 24:start + 48]
            origin = pd.Timestamp(hist.datetime.iloc[-1])
            if not (hist.datetime <= origin).all() or not (future.datetime > origin).all():
                raise AssertionError("forecast-origin leakage")
            window_id = f"site{int(site)}_start{start}"
            rows.append({"window_id": window_id, "site_id": int(site), "split": "test",
                         "context_start": hist.datetime.iloc[0], "context_end": origin,
                         "forecast_origin": origin, "target_start": target_start,
                         "target_end": target_end})
            contexts.append(hist.power.to_numpy(np.float32))
            targets.append(future.power.to_numpy(np.float32))
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise AssertionError("no test windows")
    return manifest, np.stack(contexts), np.stack(targets)


def extract_ts(model, contexts, manifest, batch_size, device):
    before = parameter_digest(model)
    chunks = []
    with torch.inference_mode():
        for left in range(0, len(contexts), batch_size):
            right = min(left + batch_size, len(contexts))
            values = torch.from_numpy(contexts[left:right, :, None]).to(device)
            coords = []
            for value in manifest.forecast_origin.iloc[left:right]:
                ts = pd.date_range(end=pd.Timestamp(value), periods=24, freq="h")
                coords.append(np.stack([ts.month, ts.day, ts.hour], axis=1))
            time = torch.from_numpy(np.stack(coords).astype(np.float32)).to(device)[:, :, :, None, None]
            chunks.append(model.extract_ts_embeddings(values, time, pooling="mean").cpu().numpy())
    result = np.concatenate(chunks).astype(np.float32)
    if result.shape != (len(contexts), 64) or not np.isfinite(result).all():
        raise AssertionError(f"invalid TS embeddings: {result.shape}")
    after = parameter_digest(model)
    if before != after:
        raise AssertionError("FusionSF parameters changed")
    return result, before


def frames(manifest, contexts, embeddings, left, right):
    context_rows, future_rows = [], []
    for i in range(left, right):
        row = manifest.iloc[i]
        hist_times = pd.date_range(end=pd.Timestamp(row.forecast_origin), periods=24, freq="h")
        future_times = pd.date_range(start=pd.Timestamp(row.target_start), periods=24, freq="h")
        for timestamp, target in zip(hist_times, contexts[i]):
            item = {"item_id": row.window_id, "timestamp": timestamp, "target": float(target)}
            if embeddings is not None:
                item.update(zip(EMBED_COLS, map(float, embeddings[i])))
            context_rows.append(item)
        future_rows.extend({"item_id": row.window_id, "timestamp": value} for value in future_times)
    context_df, future_df = pd.DataFrame(context_rows), pd.DataFrame(future_rows)
    if any(c.startswith("ts_emb_") for c in future_df):
        raise AssertionError("future_df contains embedding")
    if embeddings is None and any(c.startswith("ts_emb_") for c in context_df):
        raise AssertionError("baseline contains embedding")
    return context_df, future_df


def predict(pipeline, manifest, contexts, embeddings, window_batch, model_batch):
    out = np.empty((len(manifest), 24), np.float32)
    for left in range(0, len(manifest), window_batch):
        right = min(left + window_batch, len(manifest))
        context_df, future_df = frames(manifest, contexts, embeddings, left, right)
        result = pipeline.predict_df(context_df, future_df=future_df, prediction_length=24,
            quantile_levels=[0.5], id_column="item_id", timestamp_column="timestamp",
            target="target", batch_size=model_batch, validate_inputs=True,
            cross_learning=False, context_length=24)
        columns = [c for c in result if c not in {"item_id", "timestamp", "target_name", "predictions"}]
        point = "0.5" if "0.5" in columns else columns[0]
        for i in range(left, right):
            values = result[result.item_id == manifest.window_id.iloc[i]].sort_values("timestamp")[point].to_numpy(np.float32)
            if len(values) != 24: raise AssertionError("prediction length mismatch")
            out[i] = values
        print(f"[stage3b-ts] windows {left}:{right}", flush=True)
    if not np.isfinite(out).all(): raise AssertionError("non-finite prediction")
    return out


def metrics(pred, true):
    error = pred - true
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "nmae": float(np.mean(np.abs(error))), "nrmse": float(np.sqrt(np.mean(error ** 2))),
            "normalization_capacity": 1.0, "shape": list(pred.shape)}


def change(current, reference):
    delta = current - reference
    return {"absolute": delta, "relative_pct": 100 * delta / reference}


def bootstrap(a, b, resamples, seed):
    diff = np.mean(np.abs(a[None] if False else a), axis=1) - np.mean(np.abs(b), axis=1)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for i in range(resamples): means[i] = diff[rng.integers(0, len(diff), len(diff))].mean()
    return {"mean_difference": float(diff.mean()), "ci_2_5": float(np.quantile(means, .025)),
            "ci_97_5": float(np.quantile(means, .975)), "resamples": resamples, "seed": seed}


def main():
    args = args_parser(); cfg = load_config()
    output = args.output_dir / "smoke" if args.scope == "smoke" else args.output_dir / "full"
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    manifest, contexts, y_true = build_manifest(args.scope)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_fusionsf_model(CHECKPOINT, cfg, device)
    embeddings, fusion_hash = extract_ts(model, contexts, manifest, args.embedding_batch_size, device)
    repeated, _ = extract_ts(model, contexts, manifest, args.embedding_batch_size, device)
    if not np.array_equal(embeddings, repeated): raise AssertionError("embedding extraction not deterministic")
    permutation = sattolo_derangement(len(embeddings), args.shuffle_seed)
    shuffled = embeddings[permutation]
    pipeline = load_chronos_pipeline(CHRONOS, str(device))
    chronos_hash_before = parameter_digest(pipeline.model)
    predictions = {
        "baseline": predict(pipeline, manifest, contexts, None, args.window_batch_size, args.model_batch_size),
        "ts_aligned": predict(pipeline, manifest, contexts, embeddings, args.window_batch_size, args.model_batch_size),
        "ts_shuffled": predict(pipeline, manifest, contexts, shuffled, args.window_batch_size, args.model_batch_size),
    }
    if chronos_hash_before != parameter_digest(pipeline.model): raise AssertionError("Chronos parameters changed")
    manifest["shuffle_source_window_id"] = manifest.window_id.iloc[permutation].to_numpy()
    manifest["shuffle_source_origin"] = manifest.forecast_origin.iloc[permutation].to_numpy()
    manifest.to_csv(output / "window_manifest.csv", index=False)
    np.save(output / "ts_embeddings.npy", embeddings); np.save(output / "shuffled_ts_embeddings.npy", shuffled)
    np.save(output / "shuffle_permutation.npy", permutation); np.save(output / "y_true.npy", y_true)
    np.save(output / "t_origin.npy", manifest.forecast_origin.to_numpy(dtype="datetime64[ns]"))
    group_metrics = {}
    for name, pred in predictions.items():
        group = output / name; group.mkdir(); np.save(group / "y_pred.npy", pred)
        group_metrics[name] = metrics(pred, y_true)
        (group / "metrics.json").write_text(json.dumps(group_metrics[name], indent=2) + "\n")
    per_site = []
    for site in sorted(manifest.site_id.unique()):
        mask = manifest.site_id.to_numpy() == site
        for name, pred in predictions.items(): per_site.append({"site_id": int(site), "group": name, **metrics(pred[mask], y_true[mask])})
    pd.DataFrame(per_site).to_csv(output / "per_site_metrics.csv", index=False)
    error = {k: v-y_true for k,v in predictions.items()}
    win_aligned_baseline = float(np.mean(np.mean(np.abs(error["ts_aligned"]),1) < np.mean(np.abs(error["baseline"]),1)))
    win_aligned_shuffled = float(np.mean(np.mean(np.abs(error["ts_aligned"]),1) < np.mean(np.abs(error["ts_shuffled"]),1)))
    boot = {
        "aligned_minus_baseline_mae": bootstrap(error["ts_aligned"], error["baseline"], args.bootstrap_resamples, 2021),
        "aligned_minus_shuffled_mae": bootstrap(error["ts_aligned"], error["ts_shuffled"], args.bootstrap_resamples, 2021),
    }
    comparison = {"metrics": group_metrics,
        "aligned_vs_baseline": {k: change(group_metrics["ts_aligned"][k], group_metrics["baseline"][k]) for k in ("mae","rmse")},
        "aligned_vs_shuffled": {k: change(group_metrics["ts_aligned"][k], group_metrics["ts_shuffled"][k]) for k in ("mae","rmse")},
        "aligned_window_win_rate_vs_baseline": win_aligned_baseline,
        "aligned_window_win_rate_vs_shuffled": win_aligned_shuffled}
    audit = {"scope": args.scope, "full_multimodal_checkpoint_ts_branch": True,
        "historical_power_only": True, "nwp_used": False, "satellite_used": False,
        "future_power_used_as_input": False, "context_ends_at_origin": True,
        "same_windows_origins_targets_all_groups": True, "future_df_has_embedding": False,
        "shuffle_fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
        "shuffle_exact": bool(np.array_equal(shuffled, embeddings[permutation])),
        "fusionsf_parameters_updated": False, "chronos_parameters_updated": False,
        "optimizer_created": False, "backward_called": False, "fit_called": False,
        "trainable_module_added": False, "window_count": len(manifest),
        "site_count": int(manifest.site_id.nunique())}
    stats = {"shape": list(embeddings.shape), "min": float(embeddings.min()), "max": float(embeddings.max()),
        "mean": float(embeddings.mean()), "std": float(embeddings.std()), "finite": bool(np.isfinite(embeddings).all()),
        "zero_variance_dimensions": int(np.sum(embeddings.std(0) == 0)), "rank": int(np.linalg.matrix_rank(embeddings))}
    for name, payload in (("comparison.json", comparison), ("bootstrap_results.json", boot),
                          ("audit.json", audit), ("embedding_statistics.json", stats)):
        (output / name).write_text(json.dumps(payload, indent=2) + "\n")
    resolved = {"scope": args.scope, "dataset": "MMSP", "sites": list(range(10)), "seq_len": 24,
        "pred_len": 24, "split": "fixed_v1 final 20% target-time test", "groups": list(predictions),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "chronos_model": str(CHRONOS), "shuffle_seed": args.shuffle_seed, "quantile": .5,
        "context_length": 24, "cross_learning": False, "model_batch_size": args.model_batch_size,
        "fusionsf_parameter_digest": fusion_hash, "chronos_parameter_digest": chronos_hash_before,
        "fusionsf_git_commit": git_commit(FUSION_ROOT), "solar_git_commit": git_commit(ROOT)}
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    print(json.dumps({"output": str(output), "comparison": comparison, "bootstrap": boot, "audit": audit}, indent=2))


if __name__ == "__main__": main()
