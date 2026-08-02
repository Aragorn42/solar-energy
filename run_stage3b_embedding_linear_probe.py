#!/usr/bin/env python3
"""Stage 3B: frozen FusionSF embedding linear predictability diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.spatial.distance import pdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
STAGE3A = ROOT / "results/stage3a/gefcom_zone1_336_72"
DEFAULT_OUTPUT = ROOT / "results/stage3b/gefcom_zone1_336_72_linear_probe"
CSV_PATH = ROOT / "dataset/GEFCom/task15.csv"
ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_scope", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke_windows", type=int, default=512)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_stage3a_inputs(stage3a_dir: Path = STAGE3A, max_windows: int = 0):
    manifest = pd.read_csv(stage3a_dir / "window_manifest.csv")
    aligned_all = np.load(stage3a_dir / "aligned_embeddings.npy")
    shuffled_all = np.load(stage3a_dir / "shuffled_embeddings.npy")
    permutation_all = np.load(stage3a_dir / "shuffle_permutation.npy")
    y_all = np.load(stage3a_dir / "y_true.npy")
    origins_all = np.load(stage3a_dir / "t_origin.npy")
    if y_all.ndim == 3 and y_all.shape[-1] == 1:
        y_all = y_all[..., 0]
    n = len(manifest)
    if not (aligned_all.shape == shuffled_all.shape == (n, 64)):
        raise AssertionError("Stage 3A embedding shape/row mismatch")
    if y_all.shape != (n, 72) or origins_all.shape != (n,) or permutation_all.shape != (n,):
        raise AssertionError("Stage 3A target/origin/permutation shape mismatch")
    if not np.array_equal(shuffled_all, aligned_all[permutation_all]):
        raise AssertionError("Saved shuffled embedding is not aligned[permutation]")
    if np.any(permutation_all == np.arange(n)):
        raise AssertionError("Saved Stage 3A derangement has fixed points")
    manifest_origins = manifest["t_origin"].to_numpy(dtype="datetime64[ns]")
    np.testing.assert_array_equal(manifest_origins, origins_all)
    if max_windows:
        selection = np.arange(min(max_windows, n))
    else:
        selection = np.arange(n)
    return {
        "manifest": manifest.iloc[selection].reset_index(drop=True),
        "aligned": aligned_all[selection], "shuffled": shuffled_all[selection],
        "zero": np.zeros_like(aligned_all[selection]), "y": y_all[selection],
        "origins": origins_all[selection], "permutation_full": permutation_all,
        "full_n": n, "selection": selection,
    }


def extract_raw24(inputs: dict, csv_path: Path = CSV_PATH):
    frame = pd.read_csv(csv_path)[["date", "zone1"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    timestamps = frame["date"].to_numpy(dtype="datetime64[ns]")
    power = frame["zone1"].to_numpy(np.float32)
    rows = []
    for row, origin in zip(inputs["manifest"].itertuples(index=False), inputs["origins"]):
        start = int(row.idx_start)
        indices = np.arange(start + 336 - 24, start + 336)
        used_timestamps = timestamps[indices]
        if used_timestamps[-1] != origin or np.any(used_timestamps > origin):
            raise AssertionError("raw24 does not end at origin or contains future values")
        if not np.all(np.diff(used_timestamps) == np.timedelta64(1, "h")):
            raise AssertionError("raw24 timestamps are not consecutive hourly values")
        if str(used_timestamps[0]) != str(row.fusionsf_start) or str(used_timestamps[-1]) != str(row.fusionsf_end):
            raise AssertionError("raw24 differs from saved FusionSF input interval")
        rows.append(power[indices])
    raw24 = np.stack(rows).astype(np.float32)
    if raw24.shape != (len(inputs["origins"]), 24):
        raise AssertionError("raw24 shape mismatch")
    return raw24


def chronological_split(origins: np.ndarray):
    order = np.argsort(origins, kind="stable")
    sorted_origins = origins[order]
    if np.any(np.diff(sorted_origins) <= np.timedelta64(0, "ns")):
        raise AssertionError("Origins must be unique and strictly increasing")
    n = len(order)
    train_end, val_end = int(np.floor(0.6 * n)), int(np.floor(0.7 * n))
    split = np.full(n, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "validation"
    if not (sorted_origins[train_end - 1] < sorted_origins[train_end] and sorted_origins[val_end - 1] < sorted_origins[val_end]):
        raise AssertionError("Chronological split overlaps")
    return order, split, {"train": np.arange(train_end), "validation": np.arange(train_end, val_end), "test": np.arange(val_end, n)}


def error_metrics(y_pred: np.ndarray, y_true: np.ndarray):
    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "per_window_mae": np.mean(np.abs(error), axis=1).tolist(),
        "per_horizon_mae": np.mean(np.abs(error), axis=0).tolist(),
        "per_horizon_rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "prediction_shape": list(y_pred.shape), "prediction_sha256": sha256_array(y_pred),
    }


def choose_alpha(x: np.ndarray, y: np.ndarray, indices: dict):
    scaler = StandardScaler().fit(x[indices["train"]])
    x_train = scaler.transform(x[indices["train"]])
    x_val = scaler.transform(x[indices["validation"]])
    records, failures = [], []
    for alpha in ALPHAS:
        try:
            model = Ridge(alpha=alpha).fit(x_train, y[indices["train"]])
            prediction = model.predict(x_val)
            metric = error_metrics(prediction, y[indices["validation"]])
            records.append({"alpha": alpha, "validation_mae": metric["mae"], "validation_rmse": metric["rmse"], "status": "ok"})
        except Exception as exc:
            failures.append({"alpha": alpha, "status": "failed", "error": repr(exc)})
    if not records:
        raise RuntimeError("All Ridge alpha candidates failed")
    # Fixed rule: minimum validation MAE, and larger alpha for an exact tie.
    best = min(records, key=lambda row: (row["validation_mae"], -row["alpha"]))
    return scaler, best["alpha"], pd.DataFrame(records + failures)


def fit_group(name: str, x: np.ndarray, y: np.ndarray, indices: dict, output: Path):
    scaler, alpha, search = choose_alpha(x, y, indices)
    model = Ridge(alpha=alpha).fit(scaler.transform(x[indices["train"]]), y[indices["train"]])
    prediction = model.predict(scaler.transform(x[indices["test"]])).astype(np.float32)
    metrics = error_metrics(prediction, y[indices["test"]])
    metrics.update({"selected_alpha": alpha, "input_shape": list(x.shape),
                    "scaler_fit_sample_count": len(indices["train"]),
                    "zero_variance_feature_count": int(np.sum(scaler.scale_ == 1.0) if name == "zero" else np.sum(np.var(x[indices["train"]], axis=0) == 0))})
    group = output / name
    group.mkdir(parents=True)
    with (group / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    with (group / "ridge.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    search.to_csv(group / "alpha_search.csv", index=False)
    np.save(group / "y_pred.npy", prediction)
    (group / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (group / "scaler.pkl").open("rb") as handle:
        reloaded_scaler = pickle.load(handle)
    with (group / "ridge.pkl").open("rb") as handle:
        reloaded_model = pickle.load(handle)
    repeated = reloaded_model.predict(reloaded_scaler.transform(x[indices["test"]])).astype(np.float32)
    if not np.array_equal(prediction, repeated):
        raise AssertionError(f"Reloaded {name} Ridge prediction changed")
    return prediction, metrics


def paired_bootstrap(left: np.ndarray, right: np.ndarray, resamples=2000, seed=2021):
    difference = np.asarray(left) - np.asarray(right)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for index in range(resamples):
        sample = rng.integers(0, len(difference), size=len(difference))
        means[index] = difference[sample].mean()
    return {"mean_difference": float(difference.mean()), "ci_2_5": float(np.quantile(means, 0.025)),
            "ci_97_5": float(np.quantile(means, 0.975)), "resamples": resamples,
            "seed": seed, "sampling_unit": "forecast_window",
            "interval_crosses_zero": bool(np.quantile(means, 0.025) <= 0 <= np.quantile(means, 0.975))}


def representation_statistics(aligned: np.ndarray):
    covariance = np.cov(aligned, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)[::-1]
    positive = eigenvalues[eigenvalues > np.finfo(float).eps * eigenvalues.max()]
    probabilities = positive / positive.sum()
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    ratios = eigenvalues / eigenvalues.sum()
    distances = pdist(aligned.astype(np.float64), metric="euclidean")
    return {"per_dimension_mean": aligned.mean(0).tolist(), "per_dimension_std": aligned.std(0).tolist(),
            "per_dimension_min": aligned.min(0).tolist(), "per_dimension_max": aligned.max(0).tolist(),
            "zero_variance_dimensions": int(np.sum(aligned.std(0) == 0)),
            "covariance_matrix_rank": int(np.linalg.matrix_rank(covariance)),
            "effective_rank": effective_rank, "pca_explained_variance_ratio_first_10": ratios[:10].tolist(),
            "pairwise_distance_mean": float(distances.mean()), "pairwise_distance_std": float(distances.std()),
            "sample_count": len(aligned), "dimension": aligned.shape[1]}


def comparison_payload(metrics: dict):
    payload = {}
    for left, right in (("aligned", "shuffled"), ("aligned", "raw24"), ("shuffled", "zero")):
        row = {}
        for metric in ("mae", "rmse"):
            absolute = metrics[left][metric] - metrics[right][metric]
            row[metric] = {"absolute": absolute, "relative_pct": 100 * absolute / metrics[right][metric]}
        left_window = np.asarray(metrics[left]["per_window_mae"])
        right_window = np.asarray(metrics[right]["per_window_mae"])
        row["left_window_mae_win_rate"] = float(np.mean(left_window < right_window))
        payload[f"{left}_vs_{right}"] = row
    return payload


def main():
    args = parse_args()
    output = args.output_dir / "smoke" if args.run_scope == "smoke" else args.output_dir
    if args.run_scope == "smoke":
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.mkdir(parents=True)
    elif args.output_dir.exists():
        if {p.name for p in args.output_dir.iterdir()} != {"smoke"}:
            raise FileExistsError("Full output root may contain only the isolated smoke directory")
    else:
        output.mkdir(parents=True)
    max_windows = args.smoke_windows if args.run_scope == "smoke" else 0
    data = load_stage3a_inputs(max_windows=max_windows)
    raw24 = extract_raw24(data)
    order, split_labels, indices = chronological_split(data["origins"])
    manifest = data["manifest"].iloc[order].reset_index(drop=True)
    origins = data["origins"][order]
    y = data["y"][order]
    features = {"aligned": data["aligned"][order], "shuffled": data["shuffled"][order],
                "raw24": raw24[order], "zero": data["zero"][order]}
    manifest["split"] = split_labels
    manifest.to_csv(output / "split_manifest.csv", index=False)
    predictions, metrics = {}, {}
    for name, x in features.items():
        predictions[name], metrics[name] = fit_group(name, x, y, indices, output)
    y_test, origins_test = y[indices["test"]], origins[indices["test"]]
    np.save(output / "y_true_test.npy", y_test)
    np.save(output / "t_origin_test.npy", origins_test)
    per_window = pd.DataFrame({"window_id": manifest.iloc[indices["test"]]["window_id"].to_numpy(),
                               "t_origin": origins_test})
    per_horizon = pd.DataFrame({"horizon": np.arange(1, 73)})
    for name in features:
        per_window[f"{name}_mae"] = metrics[name]["per_window_mae"]
        per_horizon[f"{name}_mae"] = metrics[name]["per_horizon_mae"]
        per_horizon[f"{name}_rmse"] = metrics[name]["per_horizon_rmse"]
    per_window.to_csv(output / "per_window_metrics.csv", index=False)
    per_horizon.to_csv(output / "per_horizon_metrics.csv", index=False)
    bootstrap = {
        "aligned_minus_shuffled_mae": paired_bootstrap(per_window["aligned_mae"], per_window["shuffled_mae"]),
        "aligned_minus_raw24_mae": paired_bootstrap(per_window["aligned_mae"], per_window["raw24_mae"]),
    }
    comparison = comparison_payload(metrics)
    stats = representation_statistics(features["aligned"])
    # Separate representation reconstruction diagnostic: aligned -> raw24.
    reconstruction_scaler, reconstruction_alpha, reconstruction_search = choose_alpha(features["aligned"], raw24[order], indices)
    reconstruction_model = Ridge(alpha=reconstruction_alpha).fit(
        reconstruction_scaler.transform(features["aligned"][indices["train"]]), raw24[order][indices["train"]]
    )
    reconstruction_prediction = reconstruction_model.predict(reconstruction_scaler.transform(features["aligned"][indices["test"]]))
    reconstruction = {"selected_alpha": reconstruction_alpha,
                      **error_metrics(reconstruction_prediction, raw24[order][indices["test"]]),
                      "alpha_search": reconstruction_search.to_dict(orient="records")}
    aligned_better = comparison["aligned_vs_shuffled"]["mae"]["absolute"] < 0
    ci = bootstrap["aligned_minus_shuffled_mae"]
    if aligned_better and ci["ci_97_5"] < 0 and comparison["aligned_vs_shuffled"]["left_window_mae_win_rate"] > 0.5:
        conclusion = "A"
    elif aligned_better and (ci["interval_crosses_zero"] or comparison["aligned_vs_shuffled"]["left_window_mae_win_rate"] <= 0.55):
        conclusion = "B"
    else:
        conclusion = "C"
    split_ranges = {name: {"count": len(idx), "origin_start": str(origins[idx[0]]), "origin_end": str(origins[idx[-1]])}
                    for name, idx in indices.items()}
    audit = {"stage3a_shapes_and_rows_valid": True, "forecast_origin_unchanged": True,
             "raw24_exactly_origin_minus_23h_through_origin": True, "future_information_used": False,
             "chronological_split": True, "split_overlap": False, "scalers_fit_train_only": True,
             "alpha_selected_validation_only": True, "test_used_once_after_selection": True,
             "saved_shuffle_used_without_regeneration": True, "shuffle_fixed_points": 0,
             "aligned_shuffled_same_samples_marginal_distribution_shape": True,
             "all_groups_same_y_and_test_origins": True, "future_weather_used": False,
             "fusionsf_trained": False, "chronos_imported_or_run": False,
             "neural_network_or_trainable_fusion_added": False,
             "only_trainable_model": "sklearn.linear_model.Ridge", "bootstrap_resamples": 2000,
             "bootstrap_seed": 2021, "conclusion": conclusion}
    data_audit = {"stage3a_full_sample_count": data["full_n"], "used_sample_count": len(y),
                  "aligned_shape": list(features["aligned"].shape), "shuffled_shape": list(features["shuffled"].shape),
                  "raw24_shape": list(features["raw24"].shape), "zero_shape": list(features["zero"].shape),
                  "y_shape": list(y.shape), "t_origin_shape": list(origins.shape), "split_ranges": split_ranges,
                  "aligned_sha256": sha256_array(data["aligned"]), "shuffled_sha256": sha256_array(data["shuffled"]),
                  "y_sha256": sha256_array(data["y"]), "origin_sha256": sha256_array(data["origins"])}
    resolved = {"run_scope": args.run_scope, "dataset": "GEFCom2014 solar zone1", "seq_len": 336,
                "pred_len": 72, "freq": "1h", "split": "chronological 60/10/30",
                "alphas": ALPHAS, "alpha_selection": "minimum validation MAE; larger alpha on exact tie",
                "target_standardized": False, "bootstrap_resamples": 2000, "bootstrap_seed": 2021,
                "input_groups": ["aligned", "shuffled", "raw24", "zero"]}
    environment = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                   "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
    payloads = {"resolved_config.json": resolved, "environment.json": environment, "data_audit.json": data_audit,
                "bootstrap_results.json": bootstrap, "embedding_statistics.json": stats,
                "reconstruction_probe.json": reconstruction, "comparison.json": comparison, "audit.json": audit}
    for filename, payload in payloads.items():
        (output / filename).write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Stage 3B frozen embedding linear probe", "", f"Conclusion: **{conclusion}**", "",
             "| Input | Alpha | MAE | RMSE |", "|---|---:|---:|---:|"]
    for name in features:
        lines.append(f"| {name} | {metrics[name]['selected_alpha']} | {metrics[name]['mae']:.9f} | {metrics[name]['rmse']:.9f} |")
    lines += ["", "```json", json.dumps({"comparison": comparison, "bootstrap": bootstrap, "audit": audit}, indent=2), "```"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "split_ranges": split_ranges, "metrics": metrics,
                      "comparison": comparison, "bootstrap": bootstrap,
                      "reconstruction": reconstruction, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
