"""Reproducibility records for native Solar paired benchmarks."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from utils.metrics import metric


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return str(value)


def resolved_config(args):
    return {key: json_value(value) for key, value in sorted(vars(args).items())}


def build_protocol_manifest(args, datasets):
    data_file = Path(args.root_path) / args.data_path
    test = datasets["test"]
    timestamps = np.asarray(test.all_timestamps, dtype="datetime64[ns]")
    starts = test.border1 + np.arange(len(test), dtype=np.int64)
    origins = timestamps[starts + args.seq_len - 1]
    target_indices = starts[:, None] + args.seq_len + np.arange(args.pred_len)[None, :]
    targets = timestamps[target_indices]
    metric_path = Path(inspect.getsourcefile(metric)).resolve()
    return {
        "dataset_class": f"{type(test).__module__}.{type(test).__name__}",
        "data_file": str(data_file.resolve()),
        "data_file_sha256": sha256_file(data_file),
        "features": args.features,
        "target": args.target,
        "freq": args.freq,
        "seq_len": int(args.seq_len),
        "label_len": int(args.label_len),
        "pred_len": int(args.pred_len),
        "split_boundaries": {
            split: {"border1": int(dataset.border1), "border2": int(dataset.border2)}
            for split, dataset in datasets.items()
        },
        "scaler_mean": datasets["train"].scaler.mean_.tolist(),
        "scaler_scale": datasets["train"].scaler.scale_.tolist(),
        "train_scaler_fit_rows": {
            "start_inclusive": int(datasets["train"].scaler_fit_start),
            "end_exclusive": int(datasets["train"].scaler_fit_end),
        },
        "test_window_count": int(len(test)),
        "test_origin_timestamps": origins.astype(str).tolist(),
        "test_target_timestamps": targets.astype(str).tolist(),
        "metric_function": f"{metric.__module__}.{metric.__name__}",
        "metric_source_path": str(metric_path),
        "metric_source_sha256": sha256_file(metric_path),
        "inverse": bool(args.inverse),
        "batch_size": int(args.batch_size),
        "seed": int(args.random_seed),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
