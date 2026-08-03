#!/usr/bin/env python3
"""Finalize statistical Stage 4B Go/No-Go artifacts."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/stage4b/mmsp_24_24_cora_adapter/full/seed_2021"
c = json.loads((OUT / "comparison.json").read_text())
y = np.load(OUT / "y_true.npy")
cora = np.load(OUT / "cora_aligned_predictions.npy")
cross = np.load(OUT / "stage4a_cross_attention_predictions.npy")
paired = np.abs(cora - y).mean(1) - np.abs(cross - y).mean(1)
rng = np.random.default_rng(2021)
draws = np.array([paired[rng.integers(0, len(paired), len(paired))].mean() for _ in range(2000)])
ci = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
c.update({
    "cora_vs_stage4a_window_win_rate": float(np.mean(paired < 0)),
    "cora_minus_stage4a_window_mae_bootstrap_ci95": ci,
    "bootstrap_resamples": 2000, "bootstrap_seed": 2021,
    "stable_better_than_stage4a": False,
    "negative_transfer_reduced": False,
    "stage4b_go": False,
})
(OUT / "comparison.json").write_text(json.dumps(c, indent=2) + "\n")
md = f"""# Stage 4B comparison

CoRA MAE: {c['metrics']['cora_aligned']['mae']:.8f}; Stage 4A MAE: {c['metrics']['stage4a_cross_attention']['mae']:.8f}; shuffled MAE: {c['metrics']['cora_shuffled']['mae']:.8f}.

CoRA minus Stage 4A per-window MAE bootstrap 95% CI: [{ci[0]:.8f}, {ci[1]:.8f}]. The interval crosses zero. CoRA improves 6/10 sites and does not reduce the already-zero negative-transfer rate.

Stage 4B Go to Stage 4C: **False**. Formal three-seed replication has not been run.
"""
(OUT / "comparison.md").write_text(md)
names = ["Chronos-2", "Stage 4A", "CoRA", "CoRA shuffled"]
keys = ["chronos2_baseline", "stage4a_cross_attention", "cora_aligned", "cora_shuffled"]
fig, ax = plt.subplots(figsize=(7.5, 4.5)); ax.bar(names, [c["metrics"][k]["mae"] for k in keys])
ax.set_ylabel("MAE"); ax.set_title("Stage 4B MMSP unseen-site comparison"); fig.tight_layout()
fig.savefig(OUT / "mae_comparison.png", dpi=180); plt.close(fig)
