#!/usr/bin/env python3
"""Complete Stage 4A comparison/report artifacts from immutable run outputs."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/stage4a/mmsp_24_24_unica_tokens/full/seed_2021"
OLD = ROOT / "results/stage3b/mmsp_24_24_fusion_embedding/full"

comparison = json.loads((OUT / "comparison.json").read_text())
old = json.loads((OLD / "comparison.json").read_text())
comparison["metrics"]["stage3_mean_pooled_fusion"] = old["metrics"]["fusion_aligned"]
comparison["stage3_mean_pooled_source"] = str(OLD / "comparison.json")
(OUT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

current_site = pd.read_csv(OUT / "per_site_metrics.csv")
old_site = pd.read_csv(OLD / "per_site_metrics.csv")
old_site = old_site[old_site.group == "fusion_aligned"].copy()
old_site["group"] = "stage3_mean_pooled_fusion"
pd.concat([current_site, old_site], ignore_index=True).to_csv(OUT / "per_site_metrics.csv", index=False)

current_window = pd.read_csv(OUT / "per_window_metrics.csv")
old_window = pd.read_csv(OLD / "per_window_metrics.csv")
old_window = old_window[old_window.group == "fusion_aligned"].copy()
old_window["group"] = "stage3_mean_pooled_fusion"
pd.concat([current_window, old_window], ignore_index=True).to_csv(OUT / "per_window_metrics.csv", index=False)

names = ["chronos2_baseline", "stage3_mean_pooled_fusion", "fusion_aligned", "fusion_shuffled"]
values = [comparison["metrics"][name]["mae"] for name in names]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(names, values, color=["#777777", "#b99a45", "#2a7f62", "#b54b4b"])
ax.set_ylabel("MAE"); ax.set_title("Stage 4A MMSP unseen-site comparison")
ax.tick_params(axis="x", rotation=18); fig.tight_layout()
fig.savefig(OUT / "mae_comparison.png", dpi=180); plt.close(fig)

c = comparison
md = f"""# Stage 4A comparison

| Group | MAE | RMSE |
|---|---:|---:|
| Chronos-2 baseline | {c['metrics']['chronos2_baseline']['mae']:.8f} | {c['metrics']['chronos2_baseline']['rmse']:.8f} |
| Stage 3 mean-pooled Fusion | {c['metrics']['stage3_mean_pooled_fusion']['mae']:.8f} | {c['metrics']['stage3_mean_pooled_fusion']['rmse']:.8f} |
| Stage 4A aligned tokens | {c['metrics']['fusion_aligned']['mae']:.8f} | {c['metrics']['fusion_aligned']['rmse']:.8f} |
| Stage 4A shuffled tokens | {c['metrics']['fusion_shuffled']['mae']:.8f} | {c['metrics']['fusion_shuffled']['rmse']:.8f} |

Aligned-vs-shuffled window win rate: {c['aligned_vs_shuffled_window_win_rate']:.4%}.
Negative-transfer test-site rate: {c['negative_transfer_site_rate']:.4%}.
Aligned improves {c['aligned_better_than_baseline_site_count']}/{c['test_site_count']} unseen test sites.

Stage 4A Go to Stage 4B: **{c['stage4a_go']}**.
"""
(OUT / "comparison.md").write_text(md)
