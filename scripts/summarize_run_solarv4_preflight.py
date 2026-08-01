#!/usr/bin/env python3
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_TAG = "preflight_acceptance_20260802"
RUN_ROOT = ROOT / "results" / "verified" / "run_solarv4_full" / RUN_TAG
tasks = []
for run_dir in sorted(RUN_ROOT.glob("GEFCom_zone1_*_seed2026")):
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    required = [run_dir / name for name in ("resolved_config.json", "data_manifest.json", "metrics.json")]
    finite = all(metrics[key] for key in ("predictions_finite", "targets_finite", "timestamps_finite"))
    checkpoint_metadata = []
    for checkpoint in metrics["checkpoint_paths"]:
        meta_path = ROOT / f"{checkpoint}.cache.json"
        checkpoint_metadata.append(json.loads(meta_path.read_text(encoding="utf-8")))
    tasks.append({
        "task": config["task"], "model_type": config["model_type"], "target": config["target"],
        "pred_len": config["pred_len"], "eval_idx": config["eval_idx"], "seed": config["seeds"][0],
        "max_epochs": config["max_epochs"], "use_cache": config["use_cache"],
        "sample_count": metrics["sample_count"], "mae_accuracy_pct": metrics["mae_acc"],
        "rmse_accuracy_pct": metrics["rmse_acc"], "finite": finite,
        "aligned_lengths": metrics["sample_count"] == manifest["sample_count"],
        "checkpoint_selected_by_validation": all(item["checkpoint_selected_by"] == "validation_loss" for item in checkpoint_metadata),
        "artifacts_present": all(path.exists() for path in required),
        "output_directory": str(run_dir.relative_to(ROOT)),
    })
expected = {"nowcasting", "ultra_short", "short_term"}
payload = {
    "run_tag": RUN_TAG,
    "dataset": "GEFCom", "site": "zone1", "seed": 2026, "max_epochs": 1, "use_cache": False,
    "all_three_tasks_completed": {item["task"] for item in tasks} == expected,
    "all_checks_passed": len(tasks) == 3 and all(
        item["finite"] and item["aligned_lengths"] and item["checkpoint_selected_by_validation"] and item["artifacts_present"]
        for item in tasks
    ),
    "tasks": tasks,
    "excluded_from_final_results": True,
}
path = ROOT / "reports" / "run_solarv4_full_verified_preflight.json"
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
lines = [
    "# run_solarv4_full_verified preflight smoke", "",
    f"Run tag: `{RUN_TAG}`. This smoke is excluded from the final result table.", "",
    "| Task | Model | Target | Pred len | Samples | MAE accuracy | RMSE accuracy | Finite/aligned | Validation checkpoint |",
    "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: |",
]
for item in tasks:
    lines.append(
        f"| {item['task']} | {item['model_type']} | {item['target']} | {item['pred_len']} | {item['sample_count']} | "
        f"{item['mae_accuracy_pct']:.1f}% | {item['rmse_accuracy_pct']:.1f}% | "
        f"{'PASS' if item['finite'] and item['aligned_lengths'] else 'FAIL'} | "
        f"{'PASS' if item['checkpoint_selected_by_validation'] else 'FAIL'} |"
    )
(ROOT / "reports" / "run_solarv4_full_verified_preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
if not payload["all_checks_passed"]:
    raise SystemExit("preflight acceptance failed")
