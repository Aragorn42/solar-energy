import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.run_solarv4_verified_support import (
    assert_aligned,
    cache_fingerprint,
    cache_is_compatible,
    capacity_clip,
    chronological_split,
    kt_to_power,
    power_to_kt,
    reserve_output_directory,
    select_target,
    split_targets_are_disjoint,
    target_timestamps,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_solarv4_full_verified.py"
SOURCE = RUNNER.read_text(encoding="utf-8")


def test_no_backward_fill():
    assert ".bfill(" not in SOURCE
    assert 'limit_direction="both"' not in SOURCE
    assert "limit_direction='both'" not in SOURCE


def test_train_only_scaler():
    index = pd.date_range("2020-01-01", periods=10, freq="h")
    df = pd.DataFrame({"x": [1] * 6 + [100] * 4}, index=index)
    _, _, _, mean, _, boundaries = chronological_split(df)
    assert mean[0] == 1
    assert boundaries == (0, 6, 7, 10)


def test_split_target_disjoint():
    index = pd.date_range("2020-01-01", periods=100, freq="h")
    df = pd.DataFrame({"x": np.arange(100)}, index=index)
    *_, boundaries = chronological_split(df)
    assert split_targets_are_disjoint(index, boundaries, seq_len=8, pred_len=4)


def test_validation_denominator():
    assert "vl /= len(val_ld)" in SOURCE
    assert "vl /= len(te_ld)" not in SOURCE


def test_target_selection():
    assert select_target(1) == "power"
    assert all(select_target(length) == "power_kt" for length in (4, 16, 72, 288))


def test_eval_timestamp_alignment():
    index = pd.date_range("2020-01-01", periods=30, freq="h")
    result = target_timestamps(index, seq_len=10, eval_idx=3, count=4)
    assert result.equals(index[13:17])


def test_power_kt_round_trip():
    power = np.array([0.0, 4.0, 20.0])
    irradiance = np.array([0.0, 20.0, 100.0])
    restored = kt_to_power(power_to_kt(power, irradiance), irradiance)
    np.testing.assert_allclose(restored, power, atol=1e-12)


def test_capacity_clipping():
    pred, true = capacity_clip([-1, 2, 20], [-2, 3, 30], capacity=10)
    np.testing.assert_array_equal(pred, [0, 2, 10])
    np.testing.assert_array_equal(true, [0, 3, 10])


def test_coordinate_configuration():
    tree = ast.parse(SOURCE)
    coords = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STATION_COORDS" for t in node.targets):
            coords = ast.literal_eval(node.value)
    assert coords is not None
    assert set(coords) == {*(f"site{i}" for i in range(1, 9)), "skippd", "gefcom_1", "gefcom_2", "gefcom_3"}
    assert all(-90 <= lat <= 90 and -180 <= lon <= 180 for lat, lon in coords.values())
    assert '"gefcom": "UTC"' in SOURCE and '"era5": "UTC"' in SOURCE


def test_old_cache_rejected(tmp_path):
    metadata = tmp_path / "old.cache.json"
    metadata.write_text(json.dumps({"cache_fingerprint": "old"}))
    assert not cache_is_compatible(metadata, "new")


def test_cache_fingerprint_changes(tmp_path):
    data = tmp_path / "data.csv"
    code = tmp_path / "code.py"
    data.write_text("x\n1\n")
    code.write_text("x = 1\n")
    first, _ = cache_fingerprint(data, {"seed": 1}, code)
    second, _ = cache_fingerprint(data, {"seed": 2}, code)
    code.write_text("x = 2\n")
    third, _ = cache_fingerprint(data, {"seed": 1}, code)
    data.write_text("x\n2\n")
    fourth, _ = cache_fingerprint(data, {"seed": 1}, code)
    assert len({first, second, third, fourth}) == 4


def test_output_directory_collision(tmp_path):
    output = tmp_path / "run"
    reserve_output_directory(output)
    with pytest.raises(FileExistsError):
        reserve_output_directory(output)


def test_predictions_targets_timestamps_same_length():
    assert assert_aligned(np.zeros(3), np.ones(3), pd.date_range("2020", periods=3)) == 3
    with pytest.raises(ValueError):
        assert_aligned(np.zeros(2), np.ones(3), pd.date_range("2020", periods=3))


def test_test_loader_not_used_for_checkpoint_selection():
    training = SOURCE[SOURCE.index("def train_single_seed"):SOURCE.index("def compute_metrics")]
    assert "for x, y, _, _ in te_ld" not in training
    assert "torch.save(model.state_dict(), ckpt)" in training
    assert "vl < best" in training
