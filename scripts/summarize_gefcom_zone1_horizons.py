#!/usr/bin/env python3
"""Summarize the audited GEFCom zone1 Power-only horizon experiments."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "solar"
REPORTS = ROOT / "reports"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_summary(pred_len, model):
    if pred_len == 72:
        pattern = f"GEFCOM_ZONE1_336_72_FAIR_{model}_*/training_summary.json"
    else:
        pattern = f"GEFCOM_ZONE1_336_{pred_len}_SEED2021_{model}_*/training_summary.json"
    matches = list(RESULTS.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {pattern}, found {len(matches)}")
    return load_json(matches[0]), str(matches[0].relative_to(ROOT))


def row(horizon, scope, model, summary, source):
    return {
        "horizon": horizon,
        "scope": scope,
        "model": model,
        "params": summary["trainable_parameter_count"],
        "best_epoch": summary["best_epoch"],
        "raw_mae": summary["raw_metrics"]["mae"],
        "raw_rmse": summary["raw_metrics"]["rmse"],
        "physical_mae": summary["inverse_space_metrics"]["mae"],
        "physical_rmse": summary["inverse_space_metrics"]["rmse"],
        "source": source,
    }


def relative(fusion, transformer, key):
    absolute = fusion[key] - transformer[key]
    return {"absolute_change": absolute, "relative_change_pct": 100 * absolute / transformer[key]}


def main():
    rows = []
    comparisons = {}
    for horizon in (1, 4, 72):
        pair = {}
        for model in ("Transformer", "FusionSFSolar"):
            summary, source = find_summary(horizon, model)
            item = row(horizon, "seed2021", model, summary, source)
            rows.append(item)
            pair[model] = item
        comparisons[f"pred_len_{horizon}_seed2021"] = {
            metric: relative(pair["FusionSFSolar"], pair["Transformer"], metric)
            for metric in ("raw_mae", "raw_rmse", "physical_mae", "physical_rmse")
        }

    three_seed_path = REPORTS / "gefcom_zone1_pred72_power_only_three_seed.json"
    three_seed = load_json(three_seed_path)["model_summary"]
    pair = {}
    for model in ("Transformer", "FusionSFSolar"):
        summary = three_seed[model]
        item = {
            "horizon": 72,
            "scope": "three_seed_mean",
            "model": model,
            "params": summary["trainable_parameter_count"],
            "best_epoch": summary["best_epoch"]["mean"],
            "raw_mae": summary["raw_mae"]["mean"],
            "raw_rmse": summary["raw_rmse"]["mean"],
            "physical_mae": summary["physical_mae"]["mean"],
            "physical_rmse": summary["physical_rmse"]["mean"],
            "source": str(three_seed_path.relative_to(ROOT)),
        }
        rows.append(item)
        pair[model] = item
    comparisons["pred_len_72_three_seed_mean"] = {
        metric: relative(pair["FusionSFSolar"], pair["Transformer"], metric)
        for metric in ("raw_mae", "raw_rmse", "physical_mae", "physical_rmse")
    }

    audits = {}
    for horizon in (1, 4):
        path = REPORTS / f"gefcom_zone1_pred{horizon}_power_only_seed2021_fairness.json"
        audit = load_json(path)
        audits[str(horizon)] = {
            "path": str(path.relative_to(ROOT)),
            "fair_benchmark": audit["fair_benchmark"],
            "checks": audit["checks"],
        }
        if not audit["fair_benchmark"] or not all(audit["checks"].values()):
            raise RuntimeError(f"Fairness audit failed for pred_len={horizon}")

    payload = {
        "experiment": "gefcom_zone1_power_only_horizon_comparison",
        "short_horizon_seed": 2021,
        "rows": rows,
        "fusionsf_minus_transformer": comparisons,
        "fairness_audits": audits,
        "decision": {
            "short_horizons_both_improve": False,
            "recommend_additional_seeds": [1, 4],
            "reason": "pred_len=1 worsens MAE and RMSE; pred_len=4 worsens MAE while improving RMSE",
        },
        "constraints": {"future_weather_used": False, "embedding_fusion_started": False},
    }
    json_path = REPORTS / "gefcom_zone1_power_only_horizon_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# GEFCom zone1 Power-only horizon comparison",
        "",
        "All short-horizon runs use seed 2021. The pred_len=72 three-seed rows are means over seeds 2021/2022/2023; the seed-2021 rows are retained separately.",
        "",
        "| Horizon | Scope | Model | Params | Best epoch | Raw MAE | Raw RMSE | Physical MAE | Physical RMSE |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in rows:
        lines.append(
            f"| {item['horizon']} | {item['scope']} | {item['model']} | {item['params']:,} | "
            f"{item['best_epoch']:.3f} | {item['raw_mae']:.9f} | {item['raw_rmse']:.9f} | "
            f"{item['physical_mae']:.9f} | {item['physical_rmse']:.9f} |"
        )
    lines += ["", "## FusionSFSolar relative to Transformer", ""]
    for name, values in comparisons.items():
        lines.append(
            f"- {name}: Raw MAE {values['raw_mae']['absolute_change']:+.9f} "
            f"({values['raw_mae']['relative_change_pct']:+.3f}%); Raw RMSE "
            f"{values['raw_rmse']['absolute_change']:+.9f} "
            f"({values['raw_rmse']['relative_change_pct']:+.3f}%)."
        )
    lines += [
        "",
        "## Fairness and decision",
        "",
        "Both pred_len=1 and pred_len=4 audits passed every manifest check (`fair_benchmark=true`). "
        "Checks include file SHA256, Dataset class, splits, scaler fit range and parameters, complete origins and targets, "
        "test/evaluated/drop counts, metric source SHA, seed, and batch size.",
        "",
        "FusionSFSolar is worse on both metrics at pred_len=1. At pred_len=4 its MAE is worse while RMSE is better. "
        "The short-horizon direction is therefore not stable; seeds 2022 and 2023 are recommended for both short horizons before drawing a conclusion.",
        "",
        "No future weather was used. MMSP embedding and Chronos-2 fusion were not started.",
    ]
    (REPORTS / "gefcom_zone1_power_only_horizon_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
