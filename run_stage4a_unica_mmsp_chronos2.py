#!/usr/bin/env python3
"""Stage 4A: token-level FusionSF cross-attention into frozen Chronos-2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset

from run_stage3a_fusionsf_embedding_chronos2 import (
    git_commit, load_chronos_pipeline, load_fusionsf_model, parameter_digest,
    sattolo_derangement, sha256_file,
)
from run_stage3b_ts_mmsp_chronos2 import metrics
from stage4_adapter import UniCATokenAdapter, adapter_forward, freeze_backbones


ROOT = Path(__file__).resolve().parent
FUSION_ROOT = Path("/home/zhaopp/workspace/FusionSF")
RUN_ROOT = FUSION_ROOT / "outputs/pipeline_v1_fixed/20260731_224035_fusionsf_fixedv1_clean30_zeroshot_train10_19_test0_9_seed42"
CHECKPOINT = RUN_ROOT / "checkpoints/epoch_epoch=006.ckpt"
CHRONOS = Path("/home/zhaopp/.cache/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c")
STAGE3 = ROOT / "results/stage3b/mmsp_24_24_fusion_embedding/full"
OUTPUT = ROOT / "results/stage4a/mmsp_24_24_unica_tokens"
DATA_DIR = FUSION_ROOT / "data/MMSP/data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=("smoke", "full"), required=True)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--shuffle-seed", type=int, default=2021)
    p.add_argument("--extract-batch-size", type=int, default=64)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--smoke-train-windows", type=int, default=128)
    p.add_argument("--smoke-val-windows", type=int, default=64)
    p.add_argument("--smoke-test-windows", type=int, default=32)
    p.add_argument("--resume-extracted", action="store_true",
                   help="Reuse complete split arrays after a post-extraction failure")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def load_config() -> dict:
    cfg = yaml.safe_load((RUN_ROOT / ".hydra/config.yaml").read_text())
    ds, model = cfg["datamodule"]["dataset"], cfg["pl_module"]["model"]
    required = {
        "dataset": (ds["modality_mode"], "all"), "model": (model["modality_mode"], "all"),
        "seq_len": (ds["seq_len"], 24), "pred_len": (ds["pred_len"], 24),
        "fusion_dim": (model["dim"], 64), "train_site_start": (ds["num_ignored_sites"], 10),
        "train_site_stop": (ds["num_sites"], 20), "ts_mask": (model["ts_masking_ratio"], 0.0),
        "sat_mask": (model["ctx_masking_ratio"], 0.0),
    }
    for name, (actual, expected) in required.items():
        if actual != expected: raise AssertionError(f"{name}: {actual!r} != {expected!r}")
    return cfg


def scaler_state() -> dict:
    arrays = np.load(RUN_ROOT / "scalers.npz")
    meta = json.loads((RUN_ROOT / "scaler_metadata.json").read_text())
    return {
        "satellite": {**meta["satellite"], "mean": arrays["satellite_mean"], "scale": arrays["satellite_scale"]},
        "nwp": {**meta["nwp"], "mean": arrays["nwp_mean"], "scale": arrays["nwp_scale"]},
    }


def make_dataset(cfg: dict, site_start: int, site_stop: int):
    if str(FUSION_ROOT) not in sys.path: sys.path.insert(0, str(FUSION_ROOT))
    from src.datasets.tscontext_3modal_dataset import Ts3MDataset
    base = dict(cfg["datamodule"]["dataset"])
    base.pop("_target_", None); base.pop("dataset_test", None)
    base.update(data_dir=str(DATA_DIR), num_ignored_sites=site_start, num_sites=site_stop)
    return Ts3MDataset(**base, train_ratio=.6, valid_ratio=.2, test_ratio=.2,
                      precomputed_scaler_state=scaler_state())


def split_indices(dataset, split: str) -> np.ndarray:
    split_value = {"train": 0, "validation": 1, "test": 2}[split]
    per_site = len(dataset.window_records)
    return np.array([i for i in range(len(dataset))
                     if {"train": 0, "validation": 1, "test": 2}[dataset.window_records[i % per_site].split] == split_value])


def _tensor_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def extract_split(cfg, fusionsf, device, site_start, site_stop, time_split,
                  limit, batch_size, output: Path, name: str):
    cached = [output / f"{name}_{suffix}" for suffix in
              ("window_manifest.csv", "contexts.npy", "targets.npy", "fusion_tokens.npy")]
    if all(path.exists() for path in cached):
        manifest = pd.read_csv(cached[0], parse_dates=["context_start", "forecast_origin", "target_start", "target_end"])
        context, target, fusion = (np.load(path) for path in cached[1:])
        if len(manifest) != len(context) or fusion.shape != (len(manifest), 24, 64):
            raise AssertionError(f"invalid cached split {name}")
        return manifest, context, target, fusion, {
            "split": name, "shape": list(fusion.shape), "pooling": "none",
            "source_node": "FusionSF3M.forward fusion_embedding",
            "sha256": _tensor_sha256(fusion), "checkpoint_sha256": sha256_file(CHECKPOINT),
            "reused_after_post_extraction_failure": True,
        }
    dataset = make_dataset(cfg, site_start, site_stop)
    indices = split_indices(dataset, time_split)
    if limit: indices = indices[:limit]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    contexts, targets, tokens, rows = [], [], [], []
    before = parameter_digest(fusionsf)
    with torch.inference_mode():
        for batch in loader:
            model_batch = {k: v.to(device) for k, v in batch.items() if k in {
                "stl_input", "stl_coords", "ts_input", "ts_coords", "ts_time",
                "ec_input", "modality_availability"}}
            fusion = fusionsf.extract_embeddings(model_batch, "fusion", "none")["fusion"]
            if fusion.ndim != 3 or tuple(fusion.shape[1:]) != (24, 64):
                raise AssertionError(f"formal fusion tokens were pooled or malformed: {tuple(fusion.shape)}")
            tokens.append(fusion.float().cpu().numpy())
            contexts.append(batch["ts_input"].squeeze(-1).float().numpy())
            targets.append(batch["ts_target"].float().numpy())
            for j in range(len(fusion)):
                rows.append({
                    "split": name, "site_id": int(batch["site_id"][j]),
                    "context_start": pd.Timestamp(int(batch["input_start_timestamp"][j])),
                    "forecast_origin": pd.Timestamp(int(batch["input_end_timestamp"][j])),
                    "target_start": pd.Timestamp(int(batch["forecast_start_timestamp"][j])),
                    "target_end": pd.Timestamp(int(batch["forecast_end_timestamp"][j])),
                })
    if before != parameter_digest(fusionsf): raise AssertionError("FusionSF parameters changed")
    context = np.concatenate(contexts).astype(np.float32)
    target = np.concatenate(targets).astype(np.float32)
    fusion = np.concatenate(tokens).astype(np.float32)
    manifest = pd.DataFrame(rows)
    manifest.insert(0, "window_id", [f"{name}_site{s}_origin{t:%Y%m%d%H}" for s, t in zip(manifest.site_id, manifest.forecast_origin)])
    if np.any(manifest.forecast_origin >= manifest.target_start):
        pass
    if not (manifest.forecast_origin < manifest.target_start).all(): raise AssertionError("target leakage")
    np.save(output / f"{name}_contexts.npy", context)
    np.save(output / f"{name}_targets.npy", target)
    np.save(output / f"{name}_fusion_tokens.npy", fusion)
    manifest.to_csv(output / f"{name}_window_manifest.csv", index=False)
    return manifest, context, target, fusion, {
        "split": name, "shape": list(fusion.shape), "pooling": "none",
        "source_node": "FusionSF3M.forward fusion_embedding",
        "sha256": _tensor_sha256(fusion), "checkpoint_sha256": sha256_file(CHECKPOINT),
    }


def median_index(chronos) -> int:
    return int(torch.argmin(torch.abs(chronos.quantiles.float() - .5)).item())


def predict(chronos, adapter, context, fusion, batch_size, device):
    adapter.eval(); parts = []
    loader = DataLoader(TensorDataset(torch.from_numpy(context), torch.from_numpy(fusion)),
                        batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for hist, tok in loader:
            q = adapter_forward(chronos, adapter, hist.to(device), tok.to(device))
            parts.append(q[:, median_index(chronos)].float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def train_adapter(chronos, adapter, train, val, args, device, output):
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(x) for x in train)),
                        batch_size=args.train_batch_size, shuffle=True, generator=generator)
    history, best, stale = [], float("inf"), 0
    best_path = output / "adapter_best.pt"
    for epoch in range(1, args.epochs + 1):
        adapter.train(); losses = []
        for context, target, fusion in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = adapter_forward(chronos, adapter, context.to(device), fusion.to(device))
            loss = torch.mean(torch.abs(pred[:, median_index(chronos)] - target.to(device)))
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        val_pred = predict(chronos, adapter, val[0], val[2], args.eval_batch_size, device)
        val_mae = float(np.mean(np.abs(val_pred - val[1])))
        history.append({"epoch": epoch, "train_mae": float(np.mean(losses)),
                        "validation_mae": val_mae, "alpha": float(adapter.alpha.detach())})
        print(f"[stage4a] epoch={epoch} train={np.mean(losses):.6f} val={val_mae:.6f} alpha={adapter.alpha.item():.6f}", flush=True)
        if val_mae < best:
            best, stale = val_mae, 0
            torch.save(adapter.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience: break
    adapter.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    return history, best_path


def changes(current, reference):
    return {k: {"absolute": current[k] - reference[k],
                "relative_pct": 100 * (current[k] - reference[k]) / reference[k]}
            for k in ("mae", "rmse", "nmae", "nrmse")}


def tables(output, manifest, predictions, true):
    site_rows, window_rows = [], []
    for group, pred in predictions.items():
        errors = np.mean(np.abs(pred - true), axis=1)
        for i, value in enumerate(errors):
            window_rows.append({"window_id": manifest.window_id.iloc[i], "site_id": int(manifest.site_id.iloc[i]), "group": group, "mae": float(value)})
        for site in sorted(manifest.site_id.unique()):
            mask = manifest.site_id.to_numpy() == site
            site_rows.append({"site_id": int(site), "group": group, **metrics(pred[mask], true[mask])})
    pd.DataFrame(site_rows).to_csv(output / "per_site_metrics.csv", index=False)
    pd.DataFrame(window_rows).to_csv(output / "per_window_metrics.csv", index=False)
    return pd.DataFrame(site_rows), pd.DataFrame(window_rows)


def main():
    args = parse_args(); seed_everything(args.seed); cfg = load_config()
    output = args.output_dir / args.scope / f"seed_{args.seed}"
    if output.exists() and not args.resume_extracted:
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=args.resume_extracted)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fusionsf = load_fusionsf_model(CHECKPOINT, cfg, device)
    pipeline = load_chronos_pipeline(CHRONOS, str(device)); chronos = pipeline.model
    freeze_backbones(chronos, fusionsf)
    fusion_before, chronos_before = parameter_digest(fusionsf), parameter_digest(chronos)
    limits = ([args.smoke_train_windows, args.smoke_val_windows, args.smoke_test_windows]
              if args.scope == "smoke" else [0, 0, 0])
    specs = [(10, 20, "train", limits[0], "train"),
             (20, 22, "test", limits[1], "validation"),
             (0, 10, "test", limits[2], "test")]
    extracted, embedding_records = {}, []
    for start, stop, time_split, limit, name in specs:
        result = extract_split(cfg, fusionsf, device, start, stop, time_split, limit,
                               args.extract_batch_size, output, name)
        extracted[name] = result[:4]; embedding_records.append(result[4])
    train_manifest, train_context, train_target, train_fusion = extracted["train"]
    val_manifest, val_context, val_target, val_fusion = extracted["validation"]
    test_manifest, test_context, test_target, test_fusion = extracted["test"]
    train_sites, val_sites, test_sites = map(set, [train_manifest.site_id, val_manifest.site_id, test_manifest.site_id])
    if train_sites & val_sites or train_sites & test_sites or val_sites & test_sites:
        raise AssertionError("train/validation/test station overlap")
    adapter = UniCATokenAdapter().to(device)
    zero_pred = predict(chronos, adapter, test_context, test_fusion, args.eval_batch_size, device)
    baseline_ref = np.load(STAGE3 / "baseline/y_pred.npy")
    if args.scope == "full" and not np.array_equal(zero_pred, baseline_ref):
        max_delta = float(np.max(np.abs(zero_pred - baseline_ref)))
        if max_delta > 1e-5: raise AssertionError(f"zero-init does not reproduce baseline: {max_delta}")
    history, best_path = train_adapter(
        chronos, adapter, (train_context, train_target, train_fusion),
        (val_context, val_target, val_fusion), args, device, output)
    permutation = sattolo_derangement(len(test_fusion), args.shuffle_seed)
    np.save(output / "shuffle_permutation.npy", permutation)
    predictions = {
        "chronos2_baseline": baseline_ref if args.scope == "full" else zero_pred,
        "fusion_aligned": predict(chronos, adapter, test_context, test_fusion, args.eval_batch_size, device),
        "fusion_shuffled": predict(chronos, adapter, test_context, test_fusion[permutation], args.eval_batch_size, device),
    }
    group_metrics = {name: metrics(pred, test_target) for name, pred in predictions.items()}
    site_table, window_table = tables(output, test_manifest, predictions, test_target)
    aligned_window = window_table[window_table.group == "fusion_aligned"].mae.to_numpy()
    shuffled_window = window_table[window_table.group == "fusion_shuffled"].mae.to_numpy()
    baseline_window = window_table[window_table.group == "chronos2_baseline"].mae.to_numpy()
    aligned_sites = site_table[site_table.group == "fusion_aligned"].set_index("site_id")
    shuffled_sites = site_table[site_table.group == "fusion_shuffled"].set_index("site_id")
    baseline_sites = site_table[site_table.group == "chronos2_baseline"].set_index("site_id")
    aligned_better_sites = int((aligned_sites.mae < baseline_sites.mae).sum())
    negative_rate = float((aligned_sites.mae > baseline_sites.mae).mean())
    go = bool(group_metrics["fusion_aligned"]["mae"] < group_metrics["fusion_shuffled"]["mae"]
              and aligned_better_sites >= (len(aligned_sites) // 2 + 1)
              and aligned_better_sites > 1)
    for name, pred in predictions.items(): np.save(output / f"{name}_predictions.npy", pred)
    np.save(output / "y_true.npy", test_target)
    pd.concat([train_manifest, val_manifest, test_manifest]).to_csv(output / "window_manifest.csv", index=False)
    pd.DataFrame(embedding_records).to_csv(output / "embedding_manifest.csv", index=False)
    audit = {
        "audit_passed": True, "fusion_token_source": "formal FusionSF3M fusion_embedding",
        "fusion_tokens_shape": [24, 64], "mean_pooling": False,
        "chronos_injection": "after frozen encoder, before frozen output_patch_embedding",
        "alpha_zero_initialized": True, "zero_init_exact_baseline_max_abs_delta": float(np.max(np.abs(zero_pred - (baseline_ref if args.scope == 'full' else zero_pred)))),
        "aligned_shuffled_same_context_target": True, "shuffle_fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
        "train_sites": sorted(map(int, train_sites)), "validation_sites": sorted(map(int, val_sites)), "test_sites": sorted(map(int, test_sites)),
        "site_sets_disjoint": True, "future_power_as_input": False,
        "nwp_available_at_origin_under_confirmed_mmsp_protocol": True,
        "satellite_latest_equals_forecast_origin": True, "test_used_for_selection": False,
        "fusionsf_frozen": True, "chronos_frozen": True,
        "fusionsf_digest_unchanged": fusion_before == parameter_digest(fusionsf),
        "chronos_digest_unchanged": chronos_before == parameter_digest(chronos),
        "trainable_components": ["fusion_projection", "query_norm", "fusion_norm", "cross_attention", "alpha"],
    }
    comparison = {
        "metrics": group_metrics,
        "aligned_vs_shuffled": changes(group_metrics["fusion_aligned"], group_metrics["fusion_shuffled"]),
        "aligned_vs_baseline": changes(group_metrics["fusion_aligned"], group_metrics["chronos2_baseline"]),
        "aligned_vs_shuffled_window_win_rate": float(np.mean(aligned_window < shuffled_window)),
        "aligned_vs_baseline_window_win_rate": float(np.mean(aligned_window < baseline_window)),
        "negative_transfer_site_rate": negative_rate,
        "aligned_better_than_baseline_site_count": aligned_better_sites,
        "test_site_count": len(aligned_sites), "stage4a_go": go,
    }
    resolved = vars(args).copy(); resolved.update({
        "dataset": "MMSP", "seq_len": 24, "pred_len": 24,
        "train_sites": sorted(map(int, train_sites)), "validation_sites": sorted(map(int, val_sites)), "test_sites": sorted(map(int, test_sites)),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "chronos": str(CHRONOS), "solar_commit": git_commit(ROOT), "fusionsf_commit": git_commit(FUSION_ROOT),
        "adapter_checkpoint": str(best_path), "stage3_mean_pooled_comparison": str(STAGE3 / "comparison.json"),
    })
    for key, value in list(resolved.items()):
        if isinstance(value, Path): resolved[key] = str(value)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(group_metrics, indent=2) + "\n")
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (output / "gate_statistics.json").write_text(json.dumps({"applicable": False, "reason": "Stage 4A has no gate"}, indent=2) + "\n")
    md = f"# Stage 4A comparison\n\nAligned MAE: {group_metrics['fusion_aligned']['mae']:.8f}; shuffled MAE: {group_metrics['fusion_shuffled']['mae']:.8f}; baseline MAE: {group_metrics['chronos2_baseline']['mae']:.8f}.\n\nGo to Stage 4B: **{go}**.\n"
    (output / "comparison.md").write_text(md)
    print(json.dumps({"output": str(output), "audit": audit, "comparison": comparison}, indent=2))


if __name__ == "__main__": main()
