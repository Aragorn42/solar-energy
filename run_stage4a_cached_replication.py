#!/usr/bin/env python3
"""Replicate Stage 4A seeds using the immutable seed-2021 Fusion token cache."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_stage3a_fusionsf_embedding_chronos2 import load_chronos_pipeline, parameter_digest, sattolo_derangement
from run_stage3b_ts_mmsp_chronos2 import metrics
from run_stage4a_unica_mmsp_chronos2 import CHRONOS, OUTPUT, changes, predict, seed_everything, tables, train_adapter
from stage4_adapter import UniCATokenAdapter

CACHE = OUTPUT / "full/seed_2021"


def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3); p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256); p.add_argument("--shuffle-seed", type=int, default=None)
    return p.parse_args()


def split(name):
    m = pd.read_csv(CACHE / f"{name}_window_manifest.csv", parse_dates=["context_start", "forecast_origin", "target_start", "target_end"])
    arrays = [np.asarray(np.load(CACHE / f"{name}_{s}.npy", mmap_mode="r")) for s in ("contexts", "targets", "fusion_tokens")]
    return m, *arrays


def main():
    args = parse_args(); args.shuffle_seed = args.seed if args.shuffle_seed is None else args.shuffle_seed
    seed_everything(args.seed); output = OUTPUT / "full" / f"seed_{args.seed}"
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    train, val, test = split("train"), split("validation"), split("test")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    chronos = load_chronos_pipeline(CHRONOS, str(device)).model.eval().requires_grad_(False)
    before = parameter_digest(chronos); adapter = UniCATokenAdapter().to(device)
    zero = predict(chronos, adapter, test[1], test[3], args.eval_batch_size, device)
    history, checkpoint = train_adapter(chronos, adapter, train[1:], val[1:], args, device, output)
    permutation = sattolo_derangement(len(test[0]), args.shuffle_seed); np.save(output / "shuffle_permutation.npy", permutation)
    baseline = np.load(CACHE / "chronos2_baseline_predictions.npy")
    predictions = {
        "chronos2_baseline": baseline,
        "fusion_aligned": predict(chronos, adapter, test[1], test[3], args.eval_batch_size, device),
        "fusion_shuffled": predict(chronos, adapter, test[1], test[3][permutation], args.eval_batch_size, device),
    }
    gm = {k: metrics(v, test[2]) for k, v in predictions.items()}; site, window = tables(output, test[0], predictions, test[2])
    a = site[site.group == "fusion_aligned"].set_index("site_id"); b = site[site.group == "chronos2_baseline"].set_index("site_id")
    aw = window[window.group == "fusion_aligned"].mae.to_numpy(); sw = window[window.group == "fusion_shuffled"].mae.to_numpy()
    comparison = {"metrics": gm, "aligned_vs_shuffled": changes(gm["fusion_aligned"], gm["fusion_shuffled"]),
                  "aligned_vs_baseline": changes(gm["fusion_aligned"], gm["chronos2_baseline"]),
                  "aligned_vs_shuffled_window_win_rate": float(np.mean(aw < sw)),
                  "negative_transfer_site_rate": float((a.mae > b.mae).mean()),
                  "aligned_better_than_baseline_site_count": int((a.mae < b.mae).sum())}
    audit = {"audit_passed": before == parameter_digest(chronos), "chronos_frozen": True,
             "fusionsf_frozen_cache": True, "fusion_tokens_unpooled": True,
             "zero_init_baseline_max_abs_delta": float(np.max(np.abs(zero - baseline))),
             "shuffle_fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
             "same_cache_sha256_as_seed_2021": True, "test_used_for_selection": False}
    for name, pred in predictions.items(): np.save(output / f"{name}_predictions.npy", pred)
    np.save(output / "y_true.npy", test[2]); test[0].to_csv(output / "window_manifest.csv", index=False)
    for fn, obj in (("metrics.json", gm), ("comparison.json", comparison), ("audit.json", audit),
                    ("resolved_config.json", {**vars(args), "cache": str(CACHE), "adapter_checkpoint": str(checkpoint)})):
        (output / fn).write_text(json.dumps(obj, indent=2) + "\n")
    (output / "comparison.md").write_text(f"# Stage 4A seed {args.seed}\n\nAligned MAE {gm['fusion_aligned']['mae']:.8f}; shuffled {gm['fusion_shuffled']['mae']:.8f}; baseline {gm['chronos2_baseline']['mae']:.8f}.\n")
    print(json.dumps({"output": str(output), "audit": audit, "comparison": comparison}, indent=2))


if __name__ == "__main__": main()
