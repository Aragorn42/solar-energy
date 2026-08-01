"""Compare two independently saved native-Solar protocol manifests."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(left, right):
    exact_fields = [
        "dataset_class", "data_file_sha256", "features", "target", "freq",
        "seq_len", "label_len", "pred_len", "train_scaler_fit_rows",
        "inverse", "batch_size", "seed",
    ]
    checks = {f"same_{field}": left.get(field) == right.get(field) for field in exact_fields}
    checks.update({
        "same_train_val_test_boundaries": left.get("split_boundaries") == right.get("split_boundaries"),
        "same_scaler_mean": np.array_equal(left.get("scaler_mean"), right.get("scaler_mean")),
        "same_scaler_scale": np.array_equal(left.get("scaler_scale"), right.get("scaler_scale")),
        "same_test_window_count": left.get("test_window_count") == right.get("test_window_count"),
        "same_origin_timestamps": left.get("test_origin_timestamps") == right.get("test_origin_timestamps"),
        "same_full_target_timestamps": left.get("test_target_timestamps") == right.get("test_target_timestamps"),
        "same_raw_metric_function": (
            left.get("metric_function") == right.get("metric_function")
            and left.get("metric_source_sha256") == right.get("metric_source_sha256")
        ),
    })
    return {
        "left_manifest": left.get("run_setting"),
        "right_manifest": right.get("run_setting"),
        "checks": checks,
        "fair_benchmark": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-manifest", type=Path, required=True)
    parser.add_argument("--fusionsf-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = json.loads(args.transformer_manifest.read_text())
    right = json.loads(args.fusionsf_manifest.read_text())
    report = compare(left, right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["fair_benchmark"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
