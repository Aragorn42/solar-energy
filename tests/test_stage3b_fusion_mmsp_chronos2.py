import json
from pathlib import Path

import numpy as np

import run_stage3b_fusion_mmsp_chronos2 as stage


def test_saved_full_audit_passes_and_is_frozen():
    root = stage.OUTPUT / "full"
    audit = json.loads((root / "audit.json").read_text())
    assert audit["audit_passed"]
    assert audit["nwp_satellite_perturbation_check"]["passed"]
    assert audit["fusion_ts_elementwise_equal"] is False
    assert audit["shuffle_fixed_points"] == 0
    assert audit["fit_called"] is False
    assert audit["backward_called"] is False
    assert audit["optimizer_created"] is False
    assert audit["trainable_module_added"] is False


def test_saved_groups_are_row_aligned_and_shuffle_is_exact():
    root = stage.OUTPUT / "full"
    fusion = np.load(root / "fusion_embeddings.npy")
    shuffled = np.load(root / "shuffled_fusion_embeddings.npy")
    permutation = np.load(root / "shuffle_permutation.npy")
    true = np.load(root / "y_true.npy")
    assert fusion.shape == shuffled.shape == (25450, 64)
    assert true.shape == (25450, 24)
    np.testing.assert_array_equal(shuffled, fusion[permutation])
    assert not np.any(permutation == np.arange(len(permutation)))


def test_saved_negative_result_is_not_silently_changed():
    root = stage.OUTPUT / "full"
    baseline = np.load(root / "baseline/y_pred.npy")
    ts = np.load(root / "ts_aligned/y_pred.npy")
    aligned = np.load(root / "fusion_aligned/y_pred.npy")
    shuffled = np.load(root / "fusion_shuffled/y_pred.npy")
    true = np.load(root / "y_true.npy")
    np.testing.assert_array_equal(aligned, shuffled)
    assert np.mean(np.abs(aligned - true)) > np.mean(np.abs(ts - true))
    assert np.mean(np.abs(aligned - true)) > np.mean(np.abs(baseline - true))


def test_availability_evidence_records_limitations():
    evidence = json.loads((stage.OUTPUT / "full/nwp_availability_evidence.json").read_text())
    assert evidence["all_available_not_after_origin"]
    assert evidence["all_valid_times_match_target_horizon"]
    assert evidence["lead_time_hours"] == list(range(1, 25))
    assert "not encoded" in evidence["provider_or_cycle"]
    assert evidence["future_satellite_used"] is False
