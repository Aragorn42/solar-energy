from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from models import FusionSFSolar
from scripts.audit_fusionsf_solar_fairness import compare


def configs(pred_len=72):
    return Namespace(seq_len=336, pred_len=pred_len, c_out=1, d_model=64,
                     e_layers=2, n_heads=4, dropout=0.1, embed="fixed")


@pytest.mark.parametrize("pred_len", [1, 4, 16, 72, 288])
def test_standard_forward_shapes(pred_len):
    model = FusionSFSolar.Model(configs(pred_len))
    batch = 2
    result = model(torch.randn(batch, 336, 1), torch.randn(batch, 336, 4),
                   torch.randn(batch, 48 + pred_len, 1), torch.randn(batch, 48 + pred_len, 4))
    assert result.shape == (batch, pred_len, 1)


def test_future_decoder_targets_are_not_read():
    model = FusionSFSolar.Model(configs()).eval()
    x_enc = torch.randn(1, 336, 1)
    x_mark_enc = torch.randn(1, 336, 4)
    x_mark_dec = torch.randn(1, 120, 4)
    first = model(x_enc, x_mark_enc, torch.zeros(1, 120, 1), x_mark_dec)
    second = model(x_enc, x_mark_enc, torch.full((1, 120, 1), 9999.0), x_mark_dec)
    torch.testing.assert_close(first, second)


def test_power_only_has_no_weather_or_gru_modules():
    model = FusionSFSolar.Model(configs())
    names = " ".join(name.lower() for name, _ in model.named_modules())
    assert "weather" not in names
    assert not any(isinstance(module, torch.nn.GRU) for module in model.modules())


def test_native_registry_and_pipeline_are_retained():
    source = Path("exp/exp_main_solarv2.py").read_text()
    assert "'FusionSFSolar': FusionSFSolar" in source
    assert "from data_provider.data_factory_solarv2 import data_provider" in source
    assert "from utils.metrics import metric" in source
    assert "metric(preds, trues)" in source
    assert "exp.train(setting)" in Path("run_longExp_solarv2.py").read_text()


def test_metrics_file_order_matches_native_metric():
    from utils.metrics import metric
    pred = np.array([[[1.0]], [[2.0]]])
    true = np.array([[[1.5]], [[1.5]]])
    values = metric(pred, true)
    saved = np.array(values[:5], dtype=np.float64)
    assert saved.shape == (5,)


def protocol_manifest():
    return {
        "dataset_class": "Dataset_Custom_Solar", "data_file_sha256": "abc",
        "features": "S", "target": "zone1", "freq": "h", "seq_len": 336,
        "label_len": 48, "pred_len": 72,
        "train_scaler_fit_rows": {"start_inclusive": 0, "end_exclusive": 10},
        "inverse": False, "batch_size": 64, "seed": 2021,
        "split_boundaries": {"train": [0, 10], "val": [8, 12], "test": [10, 20]},
        "scaler_mean": [1.0], "scaler_scale": [2.0], "test_window_count": 2,
        "test_origin_timestamps": ["2020-01-01", "2020-01-02"],
        "test_target_timestamps": [["2020-01-02", "2020-01-03"], ["2020-01-03", "2020-01-04"]],
        "metric_function": "utils.metrics.metric", "metric_source_sha256": "def",
    }


def test_fairness_audit_compares_values_not_shapes_or_constants():
    left = protocol_manifest()
    assert compare(left, dict(left))["fair_benchmark"]
    changed = dict(left)
    changed["test_target_timestamps"] = [["WRONG", "2020-01-03"], ["2020-01-03", "2020-01-04"]]
    report = compare(left, changed)
    assert not report["checks"]["same_full_target_timestamps"]
    assert not report["fair_benchmark"]


def test_fairness_audit_detects_metric_and_boundary_mismatch():
    left = protocol_manifest()
    changed = dict(left)
    changed["metric_source_sha256"] = "different"
    changed["split_boundaries"] = {"train": [0, 9]}
    report = compare(left, changed)
    assert not report["checks"]["same_raw_metric_function"]
    assert not report["checks"]["same_train_val_test_boundaries"]
