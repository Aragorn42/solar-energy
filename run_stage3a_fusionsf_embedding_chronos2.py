#!/usr/bin/env python3
"""Stage 3A: zero-shot MMSP FusionSF embeddings as Chronos-2 past covariates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from run_chronos2_zero_shot_solar_v4 import (
    _delta_from_freq,
    _ensure_datetime64_ns,
    build_windows,
)


ROOT = Path(__file__).resolve().parent
FUSIONSF_ROOT = Path("/home/zhaopp/workspace/FusionSF")
DEFAULT_CHECKPOINT = FUSIONSF_ROOT / (
    "outputs/pipeline_v1_fixed/20260731_223710_"
    "fusionsf_fixedv1_clean30_power_sites10_seed42/checkpoints/epoch_epoch=002.ckpt"
)
DEFAULT_CSV = ROOT / "dataset/GEFCom/task15.csv"
DEFAULT_CHRONOS = Path(
    "/home/zhaopp/.cache/huggingface/hub/models--amazon--chronos-2/"
    "snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c"
)
DEFAULT_OUTPUT = ROOT / "results/stage3a/gefcom_zone1_336_72"
EMBEDDING_COLUMNS = [f"fusion_emb_{i:03d}" for i in range(64)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--chronos_model_dir", type=Path, default=DEFAULT_CHRONOS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run_scope", choices=("smoke", "full"), required=True)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--window_batch_size", type=int, default=64)
    parser.add_argument("--model_batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--quantile", type=float, default=0.5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def load_checkpoint_config(checkpoint: Path) -> dict:
    config_path = checkpoint.parent.parent / ".hydra/config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Required Hydra config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["datamodule"]["dataset"]
    model = config["pl_module"]["model"]
    assertions = {
        "dataset.modality_mode": (dataset["modality_mode"], "power"),
        "dataset.seq_len": (dataset["seq_len"], 24),
        "model.modality_mode": (model["modality_mode"], "power"),
        "model.ts_length": (model["ts_length"], 24),
        "model.dim": (model["dim"], 64),
        "model.ctx_masking_ratio": (model["ctx_masking_ratio"], 0.0),
        "model.ts_masking_ratio": (model["ts_masking_ratio"], 0.0),
        "model.vq_in_ts": (model["vq_in_ts"], False),
        "model.vq_in_ctx": (model["vq_in_ctx"], False),
        "model.vq_in_guide": (model["vq_in_guide"], False),
    }
    for name, (actual, expected) in assertions.items():
        if actual != expected:
            raise AssertionError(f"Checkpoint config {name}: {actual!r} != {expected!r}")
    return config


def load_fusionsf_model(checkpoint: Path, config: dict, device: torch.device):
    if str(FUSIONSF_ROOT) not in sys.path:
        sys.path.insert(0, str(FUSIONSF_ROOT))
    from src.models.fusionSF_3modal import FusionSF3M
    from src.models.modules.positional_encoding import Cyclical_embedding

    model_cfg = dict(config["pl_module"]["model"])
    model_cfg.pop("_target_", None)
    time_cfg = dict(model_cfg.pop("time_coords_encoder"))
    time_cfg.pop("_target_", None)
    model_cfg["time_coords_encoder"] = Cyclical_embedding(**time_cfg)
    model = FusionSF3M(**model_cfg)

    # The Lightning checkpoint contains OmegaConf objects. Append (never prepend)
    # the trusted FusionSF environment so the already-imported current torch wins.
    fusion_site = Path("/home/zhaopp/miniconda3/envs/FusionSF/lib/python3.10/site-packages")
    if fusion_site.exists() and str(fusion_site) not in sys.path:
        sys.path.append(str(fusion_site))
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch 1.x
        payload = torch.load(checkpoint, map_location="cpu")
    state = {
        key[len("model."):]: value
        for key, value in payload["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("FusionSF parameters must all be frozen")
    return model.to(device)


def load_data_and_manifest(csv_path: Path, max_windows: int = 0):
    frame = pd.read_csv(csv_path)
    required = {"date", "zone1"}
    if not required.issubset(frame.columns):
        raise KeyError(f"Expected columns {required}; got {set(frame.columns)}")
    frame = frame[["date", "zone1"]].copy()
    frame["date"] = _ensure_datetime64_ns(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    windows = build_windows(
        df=frame,
        date_col="date",
        target_col="zone1",
        seq_len=336,
        pred_len=72,
        test_ratio=0.3,
        strict_test_only=1,
        expected_delta=_delta_from_freq("1h"),
        skip_irregular_windows=True,
    )
    starts = windows["idx_start"]
    y_true = windows["y_true"].astype(np.float32)[..., None]
    origins = windows["t_origin"].astype("datetime64[ns]")
    if max_windows:
        starts, y_true, origins = starts[:max_windows], y_true[:max_windows], origins[:max_windows]
    rows = []
    timestamps = frame["date"].to_numpy(dtype="datetime64[ns]")
    for window_id, start in enumerate(starts):
        start = int(start)
        history = timestamps[start:start + 336]
        fusion = history[-24:]
        future = timestamps[start + 336:start + 336 + 72]
        origin = origins[window_id]
        if fusion[-1] != origin:
            raise AssertionError("FusionSF window must end at canonical Chronos origin")
        if np.any(np.diff(fusion) != np.timedelta64(1, "h")):
            raise AssertionError("FusionSF timestamps must be 24 continuous hourly values")
        if np.any(fusion > origin):
            raise AssertionError("FusionSF window accessed a row after origin")
        rows.append({
            "window_id": f"w{window_id}", "idx_start": start,
            "context_start": str(history[0]), "t_origin": str(origin),
            "fusionsf_start": str(fusion[0]), "fusionsf_end": str(fusion[-1]),
            "forecast_start": str(future[0]), "forecast_end": str(future[-1]),
            "target_timestamps": json.dumps([str(value) for value in future]),
        })
    return frame, starts, y_true, origins, pd.DataFrame(rows)


def build_fusionsf_batch(frame: pd.DataFrame, starts: np.ndarray, device: torch.device):
    values, times = [], []
    for start in starts:
        history_end = int(start) + 336
        fusion_slice = frame.iloc[history_end - 24:history_end]
        values.append(fusion_slice["zone1"].to_numpy(np.float32)[:, None])
        timestamp = pd.DatetimeIndex(fusion_slice["date"])
        times.append(np.stack([timestamp.month, timestamp.day, timestamp.hour], axis=1).astype(np.float32))
    ts_input = torch.from_numpy(np.stack(values)).to(device)
    coords = torch.from_numpy(np.stack(times)).to(device)[:, :, :, None, None]
    batch_size = len(starts)
    return {
        "ts_input": ts_input,
        "ts_time": coords,
        "stl_input": torch.zeros(batch_size, 24, 1, 1, 1, device=device),
        "stl_coords": torch.zeros(batch_size, 2, 1, 1, device=device),
        "ts_coords": torch.zeros(batch_size, 2, 1, 1, device=device),
        "ec_input": torch.zeros(batch_size, 24, 15, device=device),
        "modality_availability": torch.zeros(batch_size, 2, device=device),
    }


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def extract_embeddings(model, frame, starts, batch_size, device):
    before = parameter_digest(model)
    ts_parts, fusion_parts = [], []
    with torch.inference_mode():
        for left in range(0, len(starts), batch_size):
            batch = build_fusionsf_batch(frame, starts[left:left + batch_size], device)
            extracted = model.extract_embeddings(batch, "both", "mean")
            ts_parts.append(extracted["ts"].cpu().numpy())
            fusion_parts.append(extracted["fusion"].cpu().numpy())
    ts_embedding = np.concatenate(ts_parts).astype(np.float32)
    fusion_embedding = np.concatenate(fusion_parts).astype(np.float32)
    if ts_embedding.shape != (len(starts), 64):
        raise AssertionError(f"Unexpected embedding shape: {ts_embedding.shape}")
    if not np.array_equal(ts_embedding, fusion_embedding):
        raise AssertionError("Power-only fusion mean must exactly equal TS mean")
    if not np.isfinite(ts_embedding).all():
        raise AssertionError("Non-finite FusionSF embedding")
    after = parameter_digest(model)
    if before != after:
        raise AssertionError("FusionSF parameters changed during inference")
    return ts_embedding, fusion_embedding, before


def sattolo_derangement(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("Derangement requires at least two windows")
    rng = np.random.default_rng(seed)
    permutation = np.arange(size)
    for index in range(size - 1, 0, -1):
        other = int(rng.integers(0, index))
        permutation[index], permutation[other] = permutation[other], permutation[index]
    if np.any(permutation == np.arange(size)):
        raise AssertionError("Derangement contains fixed points")
    return permutation


def build_chronos_frames(
    frame: pd.DataFrame,
    starts: np.ndarray,
    offset: int,
    embeddings: Optional[np.ndarray],
):
    context_rows, future_rows = [], []
    for local_index, start in enumerate(starts):
        window_index = offset + local_index
        item_id = f"w{window_index}"
        start = int(start)
        history = frame.iloc[start:start + 336]
        future = frame.iloc[start + 336:start + 336 + 72]
        vector = None if embeddings is None else embeddings[window_index]
        for timestamp, target in zip(history["date"], history["zone1"]):
            row = {"item_id": item_id, "timestamp": timestamp, "target": float(target)}
            if vector is not None:
                row.update(zip(EMBEDDING_COLUMNS, map(float, vector)))
            context_rows.append(row)
        for timestamp in future["date"]:
            future_rows.append({"item_id": item_id, "timestamp": timestamp})
    context_df = pd.DataFrame(context_rows).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    future_df = pd.DataFrame(future_rows).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    embedding_in_context = [column for column in context_df if column.startswith("fusion_emb_")]
    embedding_in_future = [column for column in future_df if column.startswith("fusion_emb_")]
    if embeddings is None and embedding_in_context:
        raise AssertionError("Baseline context contains embedding columns")
    if embeddings is not None and embedding_in_context != EMBEDDING_COLUMNS:
        raise AssertionError("Embedding context columns are incomplete or reordered")
    if embedding_in_future:
        raise AssertionError("future_df must not contain embedding columns")
    return context_df, future_df


def load_chronos_pipeline(model_dir: Path, device: str):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(str(model_dir), device_map=device)


def run_chronos_group(
    pipeline, frame, starts, embeddings, window_batch_size, model_batch_size, quantile
):
    predictions = np.zeros((len(starts), 72, 1), dtype=np.float32)
    for left in range(0, len(starts), window_batch_size):
        right = min(left + window_batch_size, len(starts))
        context_df, future_df = build_chronos_frames(
            frame, starts[left:right], left, embeddings
        )
        result = pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=72,
            quantile_levels=[quantile],
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            batch_size=model_batch_size,
            validate_inputs=True,
            cross_learning=False,
            context_length=336,
        )
        candidate_columns = [
            column for column in result.columns
            if column not in {"item_id", "timestamp", "target_name", "predictions"}
        ]
        point_column = str(quantile) if str(quantile) in candidate_columns else candidate_columns[0]
        for index in range(left, right):
            values = result[result["item_id"] == f"w{index}"].sort_values("timestamp")[point_column].to_numpy(np.float32)
            if len(values) != 72:
                raise RuntimeError(f"Prediction length mismatch for w{index}: {len(values)}")
            predictions[index, :, 0] = values
        print(f"[stage3a] Chronos windows {left}:{right} complete")
    if not np.isfinite(predictions).all():
        raise AssertionError("Chronos predictions contain NaN/Inf")
    return predictions


def metric_payload(prediction, target):
    error = prediction - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "finite": bool(np.isfinite(prediction).all()),
        "shape": list(prediction.shape),
    }


def changes(current, reference):
    absolute = current - reference
    relative = None if reference == 0 else 100.0 * absolute / reference
    return {"absolute": absolute, "relative_pct": relative}


def write_outputs(
    output_dir, args, config, manifest, y_true, origins, embeddings, shuffled,
    permutation, predictions, parameter_hash,
):
    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir()}
        if existing != {"smoke"}:
            raise FileExistsError(
                f"Refusing to overwrite non-smoke Stage 3A output: {output_dir}; "
                f"existing entries={sorted(existing)}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    manifest = manifest.copy()
    manifest["shuffled_source_window_id"] = [f"w{i}" for i in permutation]
    manifest["shuffled_source_origin"] = [str(origins[i]) for i in permutation]
    manifest.to_csv(output_dir / "window_manifest.csv", index=False)
    np.save(output_dir / "aligned_embeddings.npy", embeddings)
    np.save(output_dir / "shuffled_embeddings.npy", shuffled)
    np.save(output_dir / "shuffle_permutation.npy", permutation)
    np.save(output_dir / "y_true.npy", y_true)
    np.save(output_dir / "t_origin.npy", origins)
    metrics = {}
    for name, prediction in predictions.items():
        group = output_dir / name
        group.mkdir()
        np.save(group / "y_pred.npy", prediction)
        metrics[name] = metric_payload(prediction, y_true)
        (group / "metrics.json").write_text(json.dumps(metrics[name], indent=2) + "\n")
    window_mae = {
        name: np.mean(np.abs(prediction - y_true), axis=(1, 2))
        for name, prediction in predictions.items()
    }
    comparison = {
        "metrics": metrics,
        "aligned_vs_baseline": {
            key: changes(metrics["aligned"][key], metrics["baseline"][key])
            for key in ("mae", "rmse")
        },
        "shuffled_vs_baseline": {
            key: changes(metrics["shuffled"][key], metrics["baseline"][key])
            for key in ("mae", "rmse")
        },
        "aligned_vs_shuffled": {
            key: changes(metrics["aligned"][key], metrics["shuffled"][key])
            for key in ("mae", "rmse")
        },
        "aligned_window_mae_win_rate_vs_baseline": float(np.mean(window_mae["aligned"] < window_mae["baseline"])),
        "shuffled_window_mae_win_rate_vs_baseline": float(np.mean(window_mae["shuffled"] < window_mae["baseline"])),
        "aligned_window_mae_win_rate_vs_shuffled": float(np.mean(window_mae["aligned"] < window_mae["shuffled"])),
        "baseline_aligned_max_abs_prediction_difference": float(np.max(np.abs(predictions["baseline"] - predictions["aligned"]))),
        "baseline_shuffled_max_abs_prediction_difference": float(np.max(np.abs(predictions["baseline"] - predictions["shuffled"]))),
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    resolved = {
        "dataset": "GEFCom zone1", "seq_len": 336, "pred_len": 72,
        "test_ratio": 0.3, "strict_test_only": 1, "freq": "1h", "cross_learning": 0,
        "power_only": True, "run_scope": args.run_scope, "max_windows": args.max_windows,
        "window_count": len(manifest), "quantile": args.quantile,
        "model_batch_size": args.model_batch_size, "window_batch_size": args.window_batch_size,
        "chronos_context_length": 336, "shuffle_seed": args.shuffle_seed,
        "chronos_model_dir": str(args.chronos_model_dir), "checkpoint": str(args.checkpoint),
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    checkpoint_manifest = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "fusionsf_git_commit": git_commit(FUSIONSF_ROOT),
        "solar_energy_git_commit": git_commit(ROOT),
        "hydra_config": str(args.checkpoint.parent.parent / ".hydra/config.yaml"),
        "parameter_digest_before_after_inference": parameter_hash,
    }
    (output_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest, indent=2) + "\n")
    std = embeddings.std(axis=0)
    audit = {
        "checkpoint_config_assertions_passed": True,
        "manifest_built_once_and_shared": True,
        "same_y_true_t_origin_window_id_all_groups": True,
        "fusionsf_window_ends_at_origin": True,
        "fusionsf_window_continuous_24h": True,
        "no_rows_after_origin_accessed": True,
        "embedding_shape": list(embeddings.shape),
        "embedding_all_finite": bool(np.isfinite(embeddings).all()),
        "embedding_dimension_std": std.tolist(),
        "zero_variance_dimensions": int(np.sum(std == 0)),
        "ts_mean_equals_fusion_mean_elementwise": True,
        "shuffle_fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
        "shuffled_exactly_aligned_permutation": bool(np.array_equal(shuffled, embeddings[permutation])),
        "embedding_columns_context_only": True,
        "future_df_has_embedding_columns": False,
        "baseline_context_has_embedding_columns": False,
        "fusionsf_training_steps": 0, "chronos_training_steps": 0,
        "optimizer_created": False, "chronos_fit_called": False,
        "fusionsf_parameters_updated": False,
        "future_weather_nwp_satellite_future_power_used": False,
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "# Stage 3A GEFCom zone1 336→72", "",
        f"Scope: `{args.run_scope}`; windows: {len(manifest)}. No model training or future covariates.", "",
        "| Group | MAE | RMSE |", "| :--- | ---: | ---: |",
    ]
    for name in ("baseline", "aligned", "shuffled"):
        lines.append(f"| {name} | {metrics[name]['mae']:.9f} | {metrics[name]['rmse']:.9f} |")
    lines += ["", "```json", json.dumps(comparison, indent=2), "```"]
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    return comparison, audit


def main():
    args = parse_args()
    if args.run_scope == "smoke" and args.max_windows not in (32, 64):
        raise ValueError("Smoke must use exactly 32 or 64 windows")
    if args.run_scope == "full" and args.max_windows != 0:
        raise ValueError("Full run must use all windows (--max_windows 0)")
    output_dir = args.output_dir / "smoke" if args.run_scope == "smoke" else args.output_dir
    config = load_checkpoint_config(args.checkpoint)
    frame, starts, y_true, origins, manifest = load_data_and_manifest(args.csv, args.max_windows)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_fusionsf_model(args.checkpoint, config, device)
    aligned, fusion_mean, parameter_hash = extract_embeddings(
        model, frame, starts, args.embedding_batch_size, device
    )
    # A second extraction is a strict reproducibility check.
    repeated, repeated_fusion, _ = extract_embeddings(
        model, frame, starts, args.embedding_batch_size, device
    )
    if not np.array_equal(aligned, repeated) or not np.array_equal(fusion_mean, repeated_fusion):
        raise AssertionError("Repeated embedding extraction is not deterministic")
    permutation = sattolo_derangement(len(starts), args.shuffle_seed)
    shuffled = aligned[permutation]
    pipeline = load_chronos_pipeline(args.chronos_model_dir, str(device))
    predictions = {
        "baseline": run_chronos_group(pipeline, frame, starts, None, args.window_batch_size, args.model_batch_size, args.quantile),
        "aligned": run_chronos_group(pipeline, frame, starts, aligned, args.window_batch_size, args.model_batch_size, args.quantile),
        "shuffled": run_chronos_group(pipeline, frame, starts, shuffled, args.window_batch_size, args.model_batch_size, args.quantile),
    }
    comparison, audit = write_outputs(
        output_dir, args, config, manifest, y_true, origins, aligned, shuffled,
        permutation, predictions, parameter_hash,
    )
    print(json.dumps({"output_dir": str(output_dir), "comparison": comparison, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
