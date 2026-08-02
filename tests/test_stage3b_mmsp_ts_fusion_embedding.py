from pathlib import Path

import pytest

import run_stage3b_mmsp_ts_fusion_embedding_chronos2 as stage3b


def test_checkpoint_is_full_three_modal_cross_site():
    audit = stage3b.checkpoint_audit()
    assert audit["enabled_modalities"] == ["historical_power", "future_nwp", "historical_satellite"]
    assert audit["training_sites"] == list(range(10, 20))
    assert audit["unseen_test_sites"] == list(range(10))
    assert audit["seq_len"] == audit["pred_len"] == 24
    assert audit["embedding_dim"] == 64


def test_checkpoint_is_not_power_only_or_fusionsf_solar():
    audit = stage3b.checkpoint_audit()
    assert "zeroshot_train10_19_test0_9" in audit["selected_checkpoint"]
    assert "FusionSFSolar" not in audit["fusion"]["model_class"]
    assert audit["checkpoint_sha256"] == "fb576b03efa1d0ad53b2cdcdfd519b541e03cafdda6af6039436df3e1572c6cc"


def test_satellite_is_historical_but_nwp_availability_is_unverified():
    audit = stage3b.availability_audit()
    assert audit["satellite_history_slice_used_by_dataset"] is True
    assert audit["satellite_target_interval_used"] is False
    assert audit["nwp_future_horizon_used_by_dataset"] is True
    assert audit["nwp_issue_or_publication_columns"] == []
    assert audit["nwp_available_at_forecast_origin_proven"] is False
    assert audit["block_fusion_embedding_experiment"] is True


def test_preflight_fails_closed_before_model_or_windows(tmp_path):
    audit = stage3b.run_preflight(tmp_path / "audit")
    assert audit["conclusion"] == "E"
    assert audit["blocked_at_execution_step"] == 2
    assert audit["window_manifest_constructed"] is False
    assert audit["embeddings_extracted"] is False
    assert audit["chronos_loaded_or_run"] is False
    assert audit["fusionsf_loaded_or_run"] is False


def test_source_contains_no_training_or_chronos_inference_calls():
    source = Path(stage3b.__file__).read_text(encoding="utf-8")
    for forbidden in (".fit(", ".predict(", ".backward(", "torch.optim", "load_fusionsf_model", "load_chronos_pipeline"):
        assert forbidden not in source
    assert "trainable_module_added\": False" in source
