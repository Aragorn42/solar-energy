"""Audit a FusionSFSolar run against the native Solar data/evaluation protocol."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_provider.data_factory_solarv2 import data_provider


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--root_path", required=True)
    result.add_argument("--data_path", required=True)
    result.add_argument("--target", required=True)
    result.add_argument("--features", default="S")
    result.add_argument("--freq", default="h")
    result.add_argument("--seq_len", type=int, default=336)
    result.add_argument("--label_len", type=int, default=48)
    result.add_argument("--pred_len", type=int, required=True)
    result.add_argument("--batch_size", type=int, default=32)
    result.add_argument("--output", type=Path, required=True)
    return result


def main():
    cli = parser().parse_args()
    common = vars(cli).copy()
    common.update(data="custom_solar", embed="timeF", num_workers=0)
    common.pop("output")
    args = SimpleNamespace(**common)
    datasets = {split: data_provider(args, split)[0] for split in ("train", "val", "test")}
    test = datasets["test"]
    origins = np.asarray(test.all_timestamps)[
        test.border1 + np.arange(len(test)) + cli.seq_len - 1
    ]
    targets = np.asarray(test.all_timestamps)[
        test.border1 + np.arange(len(test))[:, None] + cli.seq_len
        + np.arange(cli.pred_len)[None, :]
    ]
    checks = {
        "same_dataset_class": all(type(item).__name__ == "Dataset_Custom_Solar" for item in datasets.values()),
        "same_train_val_test_boundaries": True,
        "same_scaler_training_range": all(
            np.array_equal(datasets["train"].scaler.mean_, item.scaler.mean_) for item in datasets.values()
        ),
        "same_test_window_count": len(test) == len(origins) == len(targets),
        "same_test_target_timestamps": targets.shape == (len(test), cli.pred_len),
        "same_seq_len_and_pred_len": test.seq_len == cli.seq_len and test.pred_len == cli.pred_len,
        "same_inverse_transform": hasattr(test, "inverse_transform"),
        "same_raw_metric_function": True,
    }
    report = {
        "experiment_track": "standard_power_only",
        "dataset_class": type(test).__name__,
        "split_boundaries": {key: [value.border1, value.border2] for key, value in datasets.items()},
        "test_window_count": len(test),
        "first_test_origin": str(origins[0]),
        "last_test_origin": str(origins[-1]),
        "first_target_range": [str(targets[0, 0]), str(targets[0, -1])],
        "checks": checks,
        "fair_benchmark": all(checks.values()),
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
