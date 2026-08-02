#!/usr/bin/env python3
"""Stage 3B MMSP preflight; fail closed when future-NWP availability is unverified."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


SOLAR_ROOT = Path(__file__).resolve().parent
FUSIONSF_ROOT = Path("/home/zhaopp/workspace/FusionSF")
RUN_ROOT = FUSIONSF_ROOT / "outputs/pipeline_v1_fixed/20260731_224035_fusionsf_fixedv1_clean30_zeroshot_train10_19_test0_9_seed42"
CHECKPOINT = RUN_ROOT / "checkpoints/epoch_epoch=006.ckpt"
CONFIG = RUN_ROOT / ".hydra/config.yaml"
NWP_CSV = FUSIONSF_ROOT / "data/MMSP/data/nwp/nwp.csv"
SATELLITE_TIMES = FUSIONSF_ROOT / "data/MMSP/data/satellite/satellite_times.npy"
DATASET_SOURCE = FUSIONSF_ROOT / "src/datasets/tscontext_3modal_dataset.py"
OUTPUT = SOLAR_ROOT / "results/stage3b/mmsp_24_24_ts_fusion_embedding"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def checkpoint_audit() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    dataset = config["datamodule"]["dataset"]
    model = config["pl_module"]["model"]
    expected = {
        "dataset.modality_mode": (dataset["modality_mode"], "all"),
        "dataset.seq_len": (dataset["seq_len"], 24),
        "dataset.pred_len": (dataset["pred_len"], 24),
        "dataset.num_sites": (dataset["num_sites"], 20),
        "dataset.num_ignored_sites": (dataset["num_ignored_sites"], 10),
        "test.num_sites": (dataset["dataset_test"]["num_sites"], 10),
        "test.num_ignored_sites": (dataset["dataset_test"]["num_ignored_sites"], 0),
        "model.modality_mode": (model["modality_mode"], "all"),
        "model.dim": (model["dim"], 64),
    }
    for name, (actual, required) in expected.items():
        if actual != required:
            raise AssertionError(f"{name}: {actual!r} != {required!r}")
    return {
        "selected_checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "selection_basis": "fixed_v1 clean30 cross-site full-modal registry row; val/mae-selected epoch 006",
        "training_sites": list(range(10, 20)), "validation_sites": list(range(10, 20)),
        "unseen_test_sites": list(range(0, 10)),
        "seq_len": 24, "pred_len": 24, "frequency": "1h", "embedding_dim": 64,
        "enabled_modalities": ["historical_power", "future_nwp", "historical_satellite"],
        "ts_encoder": {"depth": model["depth"], "heads": model["heads"], "dim_head": model["dim_head"], "ts_length": model["ts_length"]},
        "nwp_encoder": {"guide_channels": model["guide_channels"]},
        "satellite_encoder": {"image_size": model["image_size"], "patch_size": model["patch_size"]},
        "fusion": {"model_class": model["_target_"], "use_self_attention": model["use_self_attention"], "num_mlp_heads": model["num_mlp_heads"]},
        "vq": {"ts": model["vq_in_ts"], "context": model["vq_in_ctx"], "guide": model["vq_in_guide"]},
        "checkpoint_selection": {"monitor": config["callbacks"]["model_checkpoint"]["monitor"], "mode": config["callbacks"]["model_checkpoint"]["mode"]},
        "alternative_full_modal_candidate": str(FUSIONSF_ROOT / "outputs/pipeline_v1_fixed/20260731_224035_fusionsf_fixedv1_clean30_full_sites10_seed42/checkpoints/epoch_epoch=001.ckpt"),
        "alternative_rejected_reason": "training and test sites are both 0-9; not the requested unseen-site primary protocol",
    }


def availability_audit() -> dict:
    nwp = pd.read_csv(NWP_CSV, nrows=64)
    columns = nwp.columns.tolist()
    issue_candidates = [name for name in columns if name.lower() in {"issue_time", "issue_date", "run_time", "forecast_reference_time", "cycle", "lead_time"}]
    satellite_times = np.load(SATELLITE_TIMES)
    source = DATASET_SOURCE.read_text(encoding="utf-8")
    future_nwp_slice_present = "ec_begin_index = int(ec_begin_index.total_seconds() // 3600) + self.seq_len" in source
    historical_satellite_slice_present = "stl_end_index = stl_begin_index + self.seq_len" in source
    nwp_availability_verified = bool(issue_candidates)
    return {
        "nwp_columns": columns,
        "nwp_time_column": "fcst_date",
        "nwp_issue_or_publication_columns": issue_candidates,
        "nwp_issue_time_available": nwp_availability_verified,
        "nwp_valid_time_available": "fcst_date" in columns,
        "nwp_future_horizon_used_by_dataset": future_nwp_slice_present,
        "nwp_available_at_forecast_origin_proven": False,
        "nwp_limitation": "Only valid time (fcst_date) is present; issue/publication time, forecast cycle, and lead time are absent.",
        "satellite_timestamp_start": str(satellite_times[0]),
        "satellite_timestamp_end": str(satellite_times[-1]),
        "satellite_history_slice_used_by_dataset": historical_satellite_slice_present,
        "satellite_target_interval_used": False,
        "dataset_source": str(DATASET_SOURCE), "dataset_source_sha256": sha256_file(DATASET_SOURCE),
        "block_fusion_embedding_experiment": not nwp_availability_verified,
    }


def environment_payload() -> dict:
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"], text=True
        ).strip().splitlines()
    except Exception as exc:
        gpu = [f"unavailable: {exc!r}"]
    try:
        chronos_version = importlib.metadata.version("chronos-forecasting")
    except importlib.metadata.PackageNotFoundError:
        chronos_version = "not installed in this interpreter"
    return {
        "python": sys.version, "platform": platform.platform(), "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": gpu,
        "chronos_package_version": chronos_version,
        "chronos_model": "amazon/chronos-2 local snapshot 29ec3766d36d6f73f0696f85560a422f50e8498c",
        "fusionsf_git_commit": git_commit(FUSIONSF_ROOT), "solar_energy_git_commit": git_commit(SOLAR_ROOT),
    }


def run_preflight(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    checkpoint = checkpoint_audit()
    availability = availability_audit()
    environment = environment_payload()
    audit = {
        "checkpoint_is_full_three_modal": True,
        "checkpoint_is_cross_site": True,
        "training_sites": checkpoint["training_sites"],
        "unseen_test_sites": checkpoint["unseen_test_sites"],
        "satellite_history_only_confirmed": availability["satellite_history_slice_used_by_dataset"],
        "satellite_timestamp_not_after_origin_by_indexing": availability["satellite_history_slice_used_by_dataset"],
        "future_nwp_valid_times_used": availability["nwp_future_horizon_used_by_dataset"],
        "nwp_issue_time_available": availability["nwp_issue_time_available"],
        "nwp_available_at_forecast_origin_confirmed": availability["nwp_available_at_forecast_origin_proven"],
        "future_information_risk_unresolved": True,
        "window_manifest_constructed": False, "embeddings_extracted": False,
        "chronos_loaded_or_run": False, "fusionsf_loaded_or_run": False,
        "fusionsf_trained": False, "chronos_trained": False, "chronos_fit_called": False,
        "backward_called": False, "optimizer_created": False, "trainable_module_added": False,
        "smoke_run": False, "full_run": False,
        "blocked_at_execution_step": 2,
        "conclusion": "E",
        "block_reason": "MMSP nwp.csv lacks issue/publication time, forecast cycle, and lead time, so future NWP availability at forecast origin cannot be verified.",
    }
    resolved = {
        "dataset": "MMSP", "seq_len": 24, "pred_len": 24, "frequency": "1h",
        "requested_groups": ["baseline", "ts_aligned", "fusion_aligned", "ts_shuffled", "fusion_shuffled"],
        "selected_checkpoint": str(CHECKPOINT), "execution_status": "blocked_preflight",
        "conclusion_rule": "E", "downstream_prediction_experiments_started": False,
    }
    for filename, payload in (("resolved_config.json", resolved), ("environment.json", environment),
                              ("checkpoint_manifest.json", checkpoint), ("data_audit.json", availability),
                              ("audit.json", audit)):
        (output / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = """# Stage 3B MMSP TS/Fusion embedding preflight audit

## Decision

**Conclusion E — experiment audit failed before window construction.**

The selected checkpoint is the fixed_v1 clean30 full-modal cross-site run trained on sites 10–19 and evaluated on unseen sites 0–9. Satellite indexing uses only the 24 historical frames ending at forecast origin.

The experiment cannot proceed because `nwp.csv` contains only `fcst_date` (valid time). It has no issue/publication timestamp, forecast-reference time, cycle, or lead time. The dataset loader selects the 24 NWP valid times in the target horizon, but neither local metadata nor the official public description proves those values were available at forecast origin. Under the approved protocol this unresolved availability is a blocking future-information risk.

No model was loaded, no window manifest or embedding was generated, and no Chronos prediction, smoke, or full run was started.
"""
    (output / "audit_report.md").write_text(report, encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true", required=True)
    args = parser.parse_args()
    audit = run_preflight()
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
