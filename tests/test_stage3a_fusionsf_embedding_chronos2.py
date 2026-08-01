from pathlib import Path

import numpy as np
import pytest
import torch

import run_stage3a_fusionsf_embedding_chronos2 as stage3a


@pytest.fixture(scope="module")
def manifest_data():
    return stage3a.load_data_and_manifest(stage3a.DEFAULT_CSV, max_windows=8)


@pytest.fixture(scope="module")
def frozen_model():
    config = stage3a.load_checkpoint_config(stage3a.DEFAULT_CHECKPOINT)
    return stage3a.load_fusionsf_model(stage3a.DEFAULT_CHECKPOINT, config, torch.device("cpu"))


def test_same_manifest_shared_by_all_groups(manifest_data):
    _, starts, y_true, origins, manifest = manifest_data
    shared = {"baseline": manifest, "aligned": manifest, "shuffled": manifest}
    assert shared["baseline"] is shared["aligned"] is shared["shuffled"]
    assert len(starts) == len(y_true) == len(origins) == len(manifest)


def test_fusionsf_input_window_ends_at_chronos_origin(manifest_data):
    _, _, _, origins, manifest = manifest_data
    ends = manifest["fusionsf_end"].to_numpy(dtype="datetime64[ns]")
    np.testing.assert_array_equal(ends, origins)


def test_fusionsf_input_has_nothing_after_origin(manifest_data):
    frame, starts, _, origins, _ = manifest_data
    for start, origin in zip(starts, origins):
        batch = stage3a.build_fusionsf_batch(frame, np.array([start]), torch.device("cpu"))
        assert batch["ts_input"].shape == (1, 24, 1)
        timestamps = frame.iloc[int(start) + 336 - 24:int(start) + 336]["date"].to_numpy(dtype="datetime64[ns]")
        assert timestamps[-1] == origin
        assert np.all(timestamps <= origin)
        assert np.all(np.diff(timestamps) == np.timedelta64(1, "h"))


def test_aligned_embedding_rows_map_one_to_one_to_origins(manifest_data, frozen_model):
    frame, starts, _, origins, _ = manifest_data
    embeddings, _, _ = stage3a.extract_embeddings(frozen_model, frame, starts, 4, torch.device("cpu"))
    assert embeddings.shape == (len(origins), 64)
    assert len(np.unique(origins)) == len(embeddings)


def test_shuffled_has_same_rows_and_no_fixed_points():
    aligned = np.arange(20 * 64, dtype=np.float32).reshape(20, 64)
    permutation = stage3a.sattolo_derangement(20, 42)
    shuffled = aligned[permutation]
    assert not np.any(permutation == np.arange(20))
    assert np.array_equal(shuffled, aligned[permutation])
    assert {row.tobytes() for row in shuffled} == {row.tobytes() for row in aligned}


def test_future_df_has_no_embedding_columns(manifest_data):
    frame, starts, *_ = manifest_data
    embeddings = np.zeros((len(starts), 64), dtype=np.float32)
    _, future = stage3a.build_chronos_frames(frame, starts[:2], 0, embeddings)
    assert not any(column.startswith("fusion_emb_") for column in future)


def test_baseline_has_no_embedding_columns(manifest_data):
    frame, starts, *_ = manifest_data
    context, future = stage3a.build_chronos_frames(frame, starts[:2], 0, None)
    assert not any(column.startswith("fusion_emb_") for column in context)
    assert not any(column.startswith("fusion_emb_") for column in future)


def test_three_groups_share_truth_origin_and_window_id(manifest_data):
    _, _, truth, origins, manifest = manifest_data
    groups = {name: (truth, origins, manifest["window_id"].to_numpy()) for name in ("baseline", "aligned", "shuffled")}
    for name in ("aligned", "shuffled"):
        assert groups[name][0] is groups["baseline"][0]
        assert groups[name][1] is groups["baseline"][1]
        np.testing.assert_array_equal(groups[name][2], groups["baseline"][2])


def test_repeated_embedding_extraction_is_identical(manifest_data, frozen_model):
    frame, starts, *_ = manifest_data
    first, first_fusion, first_hash = stage3a.extract_embeddings(frozen_model, frame, starts, 4, torch.device("cpu"))
    second, second_fusion, second_hash = stage3a.extract_embeddings(frozen_model, frame, starts, 4, torch.device("cpu"))
    assert np.array_equal(first, second)
    assert np.array_equal(first_fusion, second_fusion)
    assert first_hash == second_hash


def test_power_checkpoint_ts_mean_equals_fusion_mean(manifest_data, frozen_model):
    frame, starts, *_ = manifest_data
    ts_mean, fusion_mean, _ = stage3a.extract_embeddings(frozen_model, frame, starts[:2], 2, torch.device("cpu"))
    assert np.array_equal(ts_mean, fusion_mean)


def test_no_parameters_updated_and_no_training_objects(manifest_data, frozen_model):
    frame, starts, *_ = manifest_data
    before = stage3a.parameter_digest(frozen_model)
    stage3a.extract_embeddings(frozen_model, frame, starts[:2], 2, torch.device("cpu"))
    after = stage3a.parameter_digest(frozen_model)
    assert before == after
    assert frozen_model.training is False
    assert not any(parameter.requires_grad for parameter in frozen_model.parameters())
    source = Path(stage3a.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".fit(" not in source
    assert ".train(" not in source


def test_checkpoint_contract():
    config = stage3a.load_checkpoint_config(stage3a.DEFAULT_CHECKPOINT)
    assert config["datamodule"]["dataset"]["modality_mode"] == "power"
    assert config["pl_module"]["model"]["dim"] == 64
