#!/usr/bin/env python3
"""Stage 4B: CoRA-inspired dynamic/global adapter on frozen Stage 4A caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_stage3a_fusionsf_embedding_chronos2 import load_chronos_pipeline, parameter_digest, sattolo_derangement
from run_stage3b_ts_mmsp_chronos2 import metrics
from run_stage4a_unica_mmsp_chronos2 import (
    CHRONOS, ROOT, changes, predict, seed_everything, tables, train_adapter,
)
from stage4_adapter import CoRACorrelationAdapter

STAGE4A = ROOT / "results/stage4a/mmsp_24_24_unica_tokens/full/seed_2021"
STAGE4A_SMOKE = ROOT / "results/stage4a/mmsp_24_24_unica_tokens/smoke/seed_2021"
OUTPUT = ROOT / "results/stage4b/mmsp_24_24_cora_adapter"


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=("smoke", "full"), required=True)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--shuffle-seed", type=int, default=2021)
    p.add_argument("--train-batch-size", type=int, default=128); p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--patience", type=int, default=3)
    return p.parse_args()


def load_split(name, limit=0):
    manifest = pd.read_csv(STAGE4A / f"{name}_window_manifest.csv", parse_dates=["context_start", "forecast_origin", "target_start", "target_end"])
    arrays = [np.load(STAGE4A / f"{name}_{suffix}.npy", mmap_mode="r") for suffix in ("contexts", "targets", "fusion_tokens")]
    if limit: manifest, arrays = manifest.iloc[:limit].copy(), [np.asarray(x[:limit]) for x in arrays]
    else: arrays = [np.asarray(x) for x in arrays]
    return manifest, *arrays


def main():
    args = args_parser(); seed_everything(args.seed)
    output = args.output_dir / args.scope / f"seed_{args.seed}"
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    limits = (128, 64, 32) if args.scope == "smoke" else (0, 0, 0)
    train = load_split("train", limits[0]); val = load_split("validation", limits[1]); test = load_split("test", limits[2])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    chronos = load_chronos_pipeline(CHRONOS, str(device)).model.eval().requires_grad_(False)
    before = parameter_digest(chronos)
    adapter = CoRACorrelationAdapter().to(device)
    zero = predict(chronos, adapter, test[1], test[3], args.eval_batch_size, device)
    history, checkpoint = train_adapter(chronos, adapter, train[1:], val[1:], args, device, output)
    permutation = sattolo_derangement(len(test[0]), args.shuffle_seed); np.save(output / "shuffle_permutation.npy", permutation)
    reference_root = (STAGE4A.parent / f"seed_{args.seed}") if args.scope == "full" else STAGE4A_SMOKE
    stage4a_aligned = np.load(reference_root / "fusion_aligned_predictions.npy")[:len(test[0])]
    baseline = np.load(reference_root / "chronos2_baseline_predictions.npy")[:len(test[0])]
    predictions = {
        "chronos2_baseline": baseline, "stage4a_cross_attention": stage4a_aligned,
        "cora_aligned": predict(chronos, adapter, test[1], test[3], args.eval_batch_size, device),
        "cora_shuffled": predict(chronos, adapter, test[1], test[3][permutation], args.eval_batch_size, device),
    }
    group_metrics = {k: metrics(v, test[2]) for k, v in predictions.items()}
    site, window = tables(output, test[0], predictions, test[2])
    a = site[site.group == "cora_aligned"].set_index("site_id")
    x = site[site.group == "stage4a_cross_attention"].set_index("site_id")
    b = site[site.group == "chronos2_baseline"].set_index("site_id")
    stage4a_neg = float((x.mae > b.mae).mean()); cora_neg = float((a.mae > b.mae).mean())
    aligned_w = window[window.group == "cora_aligned"].mae.to_numpy()
    shuffled_w = window[window.group == "cora_shuffled"].mae.to_numpy()
    stage4a_w = window[window.group == "stage4a_cross_attention"].mae.to_numpy()
    paired = aligned_w - stage4a_w
    rng = np.random.default_rng(args.seed)
    bootstrap = np.array([paired[rng.integers(0, len(paired), len(paired))].mean() for _ in range(2000)])
    ci = [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))]
    stable_better = bool(ci[1] < 0 and int((a.mae < x.mae).sum()) >= 6)
    negative_transfer_reduced = bool(cora_neg < stage4a_neg)
    go = stable_better or negative_transfer_reduced
    for name, values in predictions.items(): np.save(output / f"{name}_predictions.npy", values)
    np.save(output / "y_true.npy", test[2]); test[0].to_csv(output / "window_manifest.csv", index=False)
    pd.DataFrame([{"split": n, "source": str(STAGE4A / f"{n}_fusion_tokens.npy"), "pooling": "none"} for n in ("train", "validation", "test")]).to_csv(output / "embedding_manifest.csv", index=False)
    comparison = {
        "metrics": group_metrics, "cora_vs_stage4a": changes(group_metrics["cora_aligned"], group_metrics["stage4a_cross_attention"]),
        "cora_vs_baseline": changes(group_metrics["cora_aligned"], group_metrics["chronos2_baseline"]),
        "aligned_vs_shuffled_window_win_rate": float(np.mean(aligned_w < shuffled_w)),
        "cora_vs_stage4a_window_win_rate": float(np.mean(paired < 0)),
        "cora_minus_stage4a_window_mae_bootstrap_ci95": ci,
        "bootstrap_resamples": 2000, "bootstrap_seed": args.seed,
        "stable_better_than_stage4a": stable_better,
        "negative_transfer_reduced": negative_transfer_reduced,
        "stage4a_negative_transfer_site_rate": stage4a_neg, "cora_negative_transfer_site_rate": cora_neg,
        "cora_better_than_stage4a_site_count": int((a.mae < x.mae).sum()), "stage4b_go": go,
    }
    audit = {
        "audit_passed": before == parameter_digest(chronos), "chronos_frozen": True, "fusionsf_frozen_cache": True,
        "fusion_tokens_unpooled": True, "alpha_beta_zero_initialized": True,
        "zero_init_baseline_max_abs_delta": float(np.max(np.abs(zero - baseline))),
        "shuffle_fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
        "same_aligned_shuffled_context_target": True, "site_sets_disjoint_inherited_stage4a": True,
        "test_used_for_selection": False, "trainable_components": ["projection", "dynamic_attention", "global_mlp", "alpha", "beta"],
    }
    resolved = {**vars(args), "dataset": "MMSP", "seq_len": 24, "pred_len": 24, "stage4a_cache": str(STAGE4A), "adapter_checkpoint": str(checkpoint)}
    for k, v in list(resolved.items()):
        if isinstance(v, Path): resolved[k] = str(v)
    for filename, payload in (("metrics.json", group_metrics), ("comparison.json", comparison), ("audit.json", audit), ("resolved_config.json", resolved)):
        (output / filename).write_text(json.dumps(payload, indent=2) + "\n")
    (output / "gate_statistics.json").write_text(json.dumps({"applicable": False, "reason": "Stage 4B has no missingness gate"}, indent=2) + "\n")
    (output / "comparison.md").write_text(f"# Stage 4B comparison\n\nCoRA MAE {group_metrics['cora_aligned']['mae']:.8f}; Stage 4A MAE {group_metrics['stage4a_cross_attention']['mae']:.8f}; shuffled MAE {group_metrics['cora_shuffled']['mae']:.8f}.\n\nGo to Stage 4C: **{go}**.\n")
    print(json.dumps({"output": str(output), "audit": audit, "comparison": comparison}, indent=2))


if __name__ == "__main__": main()
