"""Summarize the fixed GEFCom zone1 336->72 paired three-seed experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2021, 2022, 2023)
MODELS = ("Transformer", "FusionSFSolar")
SUFFIX = "custom_solar_ftS_sl336_ll48_pl72_dm512_nh8_el3_dl2_df2048_fc1_ebtimeF_dtTrue_formal_power_only_pair_0"
BLOCK_LENGTH = 168
RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def setting(seed, model):
    prefix = "GEFCOM_ZONE1_336_72_FAIR" if seed == 2021 else f"GEFCOM_ZONE1_336_72_SEED{seed}"
    return f"{prefix}_{model}_{SUFFIX}"


def load_run(seed, model):
    path = ROOT / "results" / "solar" / setting(seed, model)
    return path, json.loads((path / "training_summary.json").read_text())


def moving_block_draws(differences, starts):
    n = len(differences)
    full_blocks = n // BLOCK_LENGTH
    remainder = n - full_blocks * BLOCK_LENGTH
    cumulative = np.concatenate(([0.0], np.cumsum(differences, dtype=np.float64)))
    block_sums = cumulative[BLOCK_LENGTH:] - cumulative[:-BLOCK_LENGTH]
    totals = block_sums[starts[:, :full_blocks]].sum(axis=1)
    if remainder:
        remainder_sums = cumulative[starts[:, full_blocks] + remainder] - cumulative[starts[:, full_blocks]]
        totals += remainder_sums
    return totals / n


def bootstrap_record(mae_diff, mse_diff, rng):
    n = len(mae_diff)
    blocks = int(np.ceil(n / BLOCK_LENGTH))
    starts = rng.integers(0, n - BLOCK_LENGTH + 1, size=(RESAMPLES, blocks))
    mae_draws = moving_block_draws(mae_diff, starts)
    mse_draws = moving_block_draws(mse_diff, starts)
    return mae_draws, mse_draws, {
        "origin_count": n,
        "block_length": BLOCK_LENGTH,
        "resamples": RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "mean_mae_difference_fusion_minus_transformer": float(mae_diff.mean()),
        "mae_95_ci": np.quantile(mae_draws, [0.025, 0.975]).tolist(),
        "fusion_better_mae_probability": float(np.mean(mae_draws < 0)),
        "mean_mse_difference_fusion_minus_transformer": float(mse_diff.mean()),
        "mse_95_ci": np.quantile(mse_draws, [0.025, 0.975]).tolist(),
        "fusion_better_mse_probability": float(np.mean(mse_draws < 0)),
    }


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def main():
    runs = {}
    per_seed = []
    bootstrap = {}
    sensitivity = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    aggregate_mae_draws, aggregate_mse_draws = [], []

    for seed in SEEDS:
        runs[seed] = {}
        for model in MODELS:
            path, summary = load_run(seed, model)
            runs[seed][model] = {"path": path, "summary": summary}

        transformer = runs[seed]["Transformer"]["summary"]
        fusion = runs[seed]["FusionSFSolar"]["summary"]
        row = {"seed": seed}
        for metric_group, metric_names in (
            ("raw_metrics", ("mae", "mse", "rmse")),
            ("inverse_space_metrics", ("mae", "rmse")),
        ):
            for metric_name in metric_names:
                t_value = transformer[metric_group][metric_name]
                f_value = fusion[metric_group][metric_name]
                key = ("physical" if metric_group == "inverse_space_metrics" else "raw") + "_" + metric_name
                row[f"transformer_{key}"] = t_value
                row[f"fusionsf_{key}"] = f_value
                row[f"fusionsf_relative_{key}_pct"] = (f_value - t_value) / t_value * 100.0
        per_seed.append(row)

        t_path = runs[seed]["Transformer"]["path"]
        f_path = runs[seed]["FusionSFSolar"]["path"]
        t_pred, f_pred = np.load(t_path / "y_pred.npy"), np.load(f_path / "y_pred.npy")
        t_true, f_true = np.load(t_path / "y_true.npy"), np.load(f_path / "y_true.npy")
        if not np.array_equal(t_true, f_true):
            raise ValueError(f"seed {seed}: paired y_true arrays differ")
        t_mae = np.mean(np.abs(t_pred - t_true), axis=(1, 2))
        f_mae = np.mean(np.abs(f_pred - f_true), axis=(1, 2))
        t_mse = np.mean((t_pred - t_true) ** 2, axis=(1, 2))
        f_mse = np.mean((f_pred - f_true) ** 2, axis=(1, 2))
        mae_diff, mse_diff = f_mae - t_mae, f_mse - t_mse
        mae_draws, mse_draws, bootstrap[str(seed)] = bootstrap_record(mae_diff, mse_diff, rng)
        aggregate_mae_draws.append(mae_draws)
        aggregate_mse_draws.append(mse_draws)
        indices = np.arange(0, len(mae_diff), 72)
        sensitivity[str(seed)] = {
            "origin_stride": 72,
            "sample_count": int(len(indices)),
            "mean_mae_difference_fusion_minus_transformer": float(mae_diff[indices].mean()),
            "mean_mse_difference_fusion_minus_transformer": float(mse_diff[indices].mean()),
        }

    model_stats = {}
    for model in MODELS:
        summaries = [runs[seed][model]["summary"] for seed in SEEDS]
        model_stats[model] = {
            "raw_mae": mean_std([s["raw_metrics"]["mae"] for s in summaries]),
            "raw_rmse": mean_std([s["raw_metrics"]["rmse"] for s in summaries]),
            "physical_mae": mean_std([s["inverse_space_metrics"]["mae"] for s in summaries]),
            "physical_rmse": mean_std([s["inverse_space_metrics"]["rmse"] for s in summaries]),
            "best_epoch": mean_std([s["best_epoch"] for s in summaries]),
            "training_time_seconds": mean_std([s["training_time_seconds"] for s in summaries]),
            "trainable_parameter_count": summaries[0]["trainable_parameter_count"],
        }

    aggregate_mae = np.mean(np.stack(aggregate_mae_draws), axis=0)
    aggregate_mse = np.mean(np.stack(aggregate_mse_draws), axis=0)
    bootstrap["three_seed_average"] = {
        "mean_mae_difference_fusion_minus_transformer": float(np.mean([bootstrap[str(s)]["mean_mae_difference_fusion_minus_transformer"] for s in SEEDS])),
        "mae_95_ci": np.quantile(aggregate_mae, [0.025, 0.975]).tolist(),
        "fusion_better_mae_probability": float(np.mean(aggregate_mae < 0)),
        "mean_mse_difference_fusion_minus_transformer": float(np.mean([bootstrap[str(s)]["mean_mse_difference_fusion_minus_transformer"] for s in SEEDS])),
        "mse_95_ci": np.quantile(aggregate_mse, [0.025, 0.975]).tolist(),
        "fusion_better_mse_probability": float(np.mean(aggregate_mse < 0)),
    }
    sensitivity["three_seed_average"] = {
        "mean_mae_difference_fusion_minus_transformer": float(np.mean([sensitivity[str(s)]["mean_mae_difference_fusion_minus_transformer"] for s in SEEDS])),
        "mean_mse_difference_fusion_minus_transformer": float(np.mean([sensitivity[str(s)]["mean_mse_difference_fusion_minus_transformer"] for s in SEEDS])),
    }

    raw_mae_rel = [row["fusionsf_relative_raw_mae_pct"] for row in per_seed]
    raw_rmse_rel = [row["fusionsf_relative_raw_rmse_pct"] for row in per_seed]
    decision = {
        "mae_improved_seed_count": int(sum(value < 0 for value in raw_mae_rel)),
        "rmse_improved_seed_count": int(sum(value < 0 for value in raw_rmse_rel)),
        "average_relative_mae_change_pct": float(np.mean(raw_mae_rel)),
        "average_relative_rmse_change_pct": float(np.mean(raw_rmse_rel)),
    }
    stable_better = (
        decision["mae_improved_seed_count"] >= 2
        and decision["rmse_improved_seed_count"] >= 2
        and model_stats["FusionSFSolar"]["raw_mae"]["mean"] < model_stats["Transformer"]["raw_mae"]["mean"]
        and model_stats["FusionSFSolar"]["raw_rmse"]["mean"] < model_stats["Transformer"]["raw_rmse"]["mean"]
    )
    if stable_better:
        if abs(decision["average_relative_mae_change_pct"]) < 1 and abs(decision["average_relative_rmse_change_pct"]) < 1:
            decision["recommendation"] = "stable_but_below_1pct_keep_as_baseline_then_prioritize_embedding"
        else:
            decision["recommendation"] = "recommend_expand_to_gefcom_pred_len_1_and_4"
    else:
        decision["recommendation"] = "do_not_expand_analyze_stability_or_parameter_matching"
    fairness_audits = {}
    window_protocol = {}
    for seed in SEEDS:
        audit_name = (
            "gefcom_zone1_pred72_power_only_fairness.json" if seed == 2021
            else f"gefcom_zone1_pred72_power_only_seed{seed}_fairness.json"
        )
        fairness_audits[str(seed)] = json.loads((ROOT / "reports" / audit_name).read_text())
        summary = runs[seed]["Transformer"]["summary"]
        dataset_count = int(summary.get("dataset_test_window_count", 5625))
        evaluated_count = int(summary["evaluated_test_window_count"])
        window_protocol[str(seed)] = {
            "data_loader_drop_last": bool(summary.get("data_loader_drop_last", evaluated_count < dataset_count)),
            "dataset_test_window_count": dataset_count,
            "evaluated_test_window_count": evaluated_count,
            "dropped_test_window_count": int(dataset_count - evaluated_count),
        }
    payload = {
        "experiment": "gefcom_zone1_336_72_power_only_three_seed",
        "seeds": list(SEEDS), "std_definition": "sample standard deviation (ddof=1)",
        "per_seed": per_seed, "model_summary": model_stats,
        "moving_block_bootstrap": bootstrap,
        "non_overlapping_origin_sensitivity": sensitivity,
        "fairness_audits": fairness_audits,
        "test_window_protocol": window_protocol,
        "decision": decision,
    }
    output = ROOT / "reports" / "gefcom_zone1_pred72_power_only_three_seed.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
