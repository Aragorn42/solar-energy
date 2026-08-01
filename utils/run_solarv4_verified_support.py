"""Pure, testable safety helpers for the verified SolarV4 runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def select_target(pred_len: int) -> str:
    return "power" if pred_len == 1 else "power_kt"


def chronological_split(df, train_ratio=0.6, val_ratio=0.1):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    mean = train.to_numpy(dtype=np.float32).mean(0).astype(np.float32)
    scale = train.to_numpy(dtype=np.float32).std(0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return train, val, test, mean, scale, (0, train_end, val_end, n)


def target_timestamps(test_index, seq_len, eval_idx, count):
    start = seq_len + eval_idx
    result = pd.DatetimeIndex(test_index)[start:start + count]
    if len(result) != count:
        raise ValueError("target timestamp count does not match predictions")
    return result


def power_to_kt(power, cs_ghi):
    power = np.asarray(power, dtype=np.float64)
    cs_ghi = np.asarray(cs_ghi, dtype=np.float64)
    result = np.clip(power / np.maximum(cs_ghi, 10.0), 0.0, 10.0)
    return np.where(cs_ghi > 0, result, 0.0)


def kt_to_power(power_kt, cs_ghi):
    power_kt = np.asarray(power_kt, dtype=np.float64)
    cs_ghi = np.asarray(cs_ghi, dtype=np.float64)
    return np.where(cs_ghi > 0, power_kt * np.maximum(cs_ghi, 10.0), 0.0)


def capacity_clip(predictions, targets, capacity):
    return (
        np.clip(np.asarray(predictions), 0.0, capacity),
        np.clip(np.asarray(targets), 0.0, capacity),
    )


def assert_aligned(predictions, targets, timestamps):
    lengths = (len(predictions), len(targets), len(timestamps))
    if len(set(lengths)) != 1:
        raise ValueError(f"prediction/target/timestamp length mismatch: {lengths}")
    return lengths[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_fingerprint(data_path, config, code_path):
    payload = {
        "data_sha256": sha256_file(data_path),
        "config": config,
        "code_sha256": sha256_file(code_path),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def cache_is_compatible(metadata_path, expected_fingerprint):
    path = Path(metadata_path)
    if not path.exists():
        return False
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return metadata.get("cache_fingerprint") == expected_fingerprint


def reserve_output_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def split_targets_are_disjoint(index, boundaries, seq_len, pred_len):
    _, train_end, val_end, n = boundaries
    index = pd.DatetimeIndex(index)
    ranges = []
    for start, end in ((0, train_end), (train_end, val_end), (val_end, n)):
        first = start + seq_len
        last_exclusive = end
        ranges.append(set(index[first:last_exclusive]))
    return all(ranges[i].isdisjoint(ranges[j]) for i in range(3) for j in range(i + 1, 3))
