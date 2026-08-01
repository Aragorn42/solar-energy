#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sites = [("StateGrid", f"site{i}", "15min") for i in range(1, 9)]
sites += [("SKIPPD", "site1", "15min")]
sites += [("GEFCom", f"zone{i}", "1h") for i in range(1, 4)]
tasks = {
    "nowcasting": ("PatchTST", {"15min": 1, "1h": 1}),
    "ultra_short": ("DLinear", {"15min": 16, "1h": 4}),
    "short_term": ("NLinear", {"15min": 288, "1h": 72}),
}
seeds = [42, 123]
rows = []
keys = []
for dataset, site, frequency in sites:
    for task, (model, horizons) in tasks.items():
        for seed in seeds:
            key = (dataset, site, task, horizons[frequency], model, seed)
            keys.append(key)
            rows.append(key)
duplicates = len(keys) - len(set(keys))
lines = [
    "# run_solarv4_full_verified full-run plan",
    "",
    "This is a plan only. The final full run has not been started.",
    "",
    f"- Sites: {len(sites)} (8 StateGrid + 1 SKIPPD + 3 GEFCom)",
    f"- Task configurations: {len(sites) * len(tasks)}",
    f"- Seeds per task: {len(seeds)} (`{seeds[0]}`, `{seeds[1]}`)",
    f"- Total model runs: {len(rows)}",
    f"- Duplicate configurations: {duplicates}",
    "- Output rule: `results/verified/run_solarv4_full/<unique_run_tag>/<dataset>_<site>_<task>_seed42-123/`",
    "- Collision rule: an existing run directory raises `FileExistsError`; it is never overwritten.",
    "- Retry rule: use a new run tag and rerun only the failed dataset/site/task; cache remains disabled unless its data/config/code fingerprint matches exactly.",
    "",
    "| Dataset | Site | Task | Model | Pred len | Seed |",
    "| :--- | :--- | :--- | :--- | ---: | ---: |",
]
for dataset, site, task, pred_len, model, seed in rows:
    lines.append(f"| {dataset} | {site} | {task} | {model} | {pred_len} | {seed} |")
(ROOT / "reports" / "run_solarv4_full_run_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
if duplicates:
    raise SystemExit(f"duplicate configurations detected: {duplicates}")
