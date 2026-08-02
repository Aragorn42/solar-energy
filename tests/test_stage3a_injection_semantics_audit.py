from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import run_stage3a_fusionsf_embedding_chronos2 as stage3a
import run_stage3a_injection_semantics_audit as audit


@pytest.fixture(scope="module")
def saved():
    return audit.load_saved_inputs(n_windows=8)


def test_saved_embedding_controls(saved):
    _, aligned, shuffled, zero, permutation, *_ = saved
    assert aligned.shape == shuffled.shape == zero.shape == (8, 64)
    assert np.count_nonzero(zero) == 0
    full_aligned = np.load(audit.STAGE_ROOT / "aligned_embeddings.npy")
    np.testing.assert_array_equal(shuffled, full_aligned[permutation])
    full_permutation = np.load(audit.STAGE_ROOT / "shuffle_permutation.npy")
    assert not np.any(full_permutation == np.arange(len(full_permutation)))


def test_saved_manifest_alignment_and_no_future_input(saved):
    manifest, *_, y_true, origins = saved
    frame, starts = audit.load_frame_and_verify(manifest, y_true, origins)
    assert len(manifest) == len(origins) == len(y_true) == 8
    for start, origin in zip(starts, origins):
        timestamps = frame.iloc[start:start + 336]["date"].to_numpy(dtype="datetime64[ns]")
        assert timestamps[-1] == origin
        assert np.all(timestamps <= origin)


def test_dataframe_column_and_repetition_contract(saved):
    manifest, aligned, shuffled, zero, _, y_true, origins = saved
    frame, starts = audit.load_frame_and_verify(manifest, y_true, origins)
    column_sets = []
    for embedding in (aligned, shuffled, zero):
        context, future = stage3a.build_chronos_frames(frame, starts, 0, embedding)
        column_sets.append(list(context.columns))
        assert not any(c.startswith("fusion_emb_") for c in future.columns)
        for _, group in context.groupby("item_id", sort=False):
            assert np.max(group[stage3a.EMBEDDING_COLUMNS].std(ddof=0).to_numpy()) == 0
    assert column_sets[0] == column_sets[1] == column_sets[2]
    baseline, _ = stage3a.build_chronos_frames(frame, starts, 0, None)
    assert not any(c.startswith("fusion_emb_") for c in baseline.columns)


def test_actual_chronos_instance_norm_result_is_measured_not_assumed(saved):
    _, aligned, shuffled, zero, *_ = saved
    norm = __import__("chronos.chronos_bolt", fromlist=["InstanceNorm"]).InstanceNorm()
    outputs = []
    for embedding in (aligned, shuffled, zero):
        repeated = torch.from_numpy(np.repeat(embedding[:, :, None], 336, axis=2)).reshape(-1, 336)
        normalized, _ = norm(repeated)
        outputs.append(normalized.numpy())
        assert np.isfinite(normalized.numpy()).all()
        assert float(normalized.std(dim=-1).max()) == 0
    assert float(np.max(np.abs(outputs[2]))) == 0
    assert bool(np.allclose(outputs[0], outputs[1], atol=audit.ATOL, rtol=0)) is False
    assert bool(np.allclose(outputs[0], outputs[2], atol=audit.ATOL, rtol=0)) is False


def test_no_training_code_present():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    for forbidden in ("torch.optim", ".fit(", ".backward(", ".train("):
        assert forbidden not in source
