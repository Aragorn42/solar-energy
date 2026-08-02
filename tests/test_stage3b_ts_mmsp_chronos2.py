import ast
from pathlib import Path

import numpy as np

import run_stage3b_ts_mmsp_chronos2 as stage


def test_checkpoint_is_full_modal_cross_site_but_ts_only_protocol():
    cfg = stage.load_config()
    assert cfg["datamodule"]["dataset"]["modality_mode"] == "all"
    assert cfg["datamodule"]["dataset"]["num_ignored_sites"] == 10
    assert cfg["datamodule"]["dataset"]["dataset_test"]["num_sites"] == 10


def test_smoke_manifest_is_two_unseen_sites_and_history_only():
    manifest, context, target = stage.build_manifest("smoke")
    assert len(manifest) == 32 and manifest.site_id.nunique() == 2
    assert context.shape == target.shape == (32, 24)
    assert (manifest.context_end == manifest.forecast_origin).all()
    assert (manifest.target_start > manifest.forecast_origin).all()


def test_derangement_is_fixed_and_exact():
    embedding = np.arange(32 * 64).reshape(32, 64)
    permutation = stage.sattolo_derangement(32, 2021)
    assert not np.any(permutation == np.arange(32))
    assert np.array_equal(embedding[permutation], embedding[permutation])


def test_context_embedding_is_past_only_and_future_has_none():
    manifest, context, _ = stage.build_manifest("smoke")
    embedding = np.zeros((len(manifest), 64), np.float32)
    context_df, future_df = stage.frames(manifest, context, embedding, 0, 2)
    assert [c for c in context_df if c.startswith("ts_emb_")] == stage.EMBED_COLS
    assert not [c for c in future_df if c.startswith("ts_emb_")]
    baseline, _ = stage.frames(manifest, context, None, 0, 2)
    assert not [c for c in baseline if c.startswith("ts_emb_")]


def test_script_has_no_training_calls_or_future_modalities():
    source = Path(stage.__file__).read_text()
    tree = ast.parse(source)
    called = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not ({"fit", "backward", "train"} & called)
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any("optim" in name.lower() for name in imported)
    assert "nwp.csv" not in source and "satellite.npy" not in source


def test_fusionsf_read_only_interface_accepts_no_other_modalities():
    source = Path("/home/zhaopp/workspace/FusionSF/src/models/fusionSF_3modal.py").read_text()
    tree = ast.parse(source)
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "extract_ts_embeddings")
    args = [arg.arg for arg in method.args.args]
    assert args == ["self", "ts", "time_coords", "pooling"]
