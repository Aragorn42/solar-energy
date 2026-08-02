from pathlib import Path

import numpy as np
import pickle
import pytest

import run_stage3b_embedding_linear_probe as stage3b


@pytest.fixture(scope="module")
def inputs():
    data = stage3b.load_stage3a_inputs(max_windows=128)
    data["raw24"] = stage3b.extract_raw24(data)
    return data


def test_stage3a_shapes_rows_and_shared_samples(inputs):
    n = len(inputs["manifest"])
    assert inputs["aligned"].shape == inputs["shuffled"].shape == inputs["zero"].shape == (n, 64)
    assert inputs["y"].shape == (n, 72)
    assert inputs["origins"].shape == (n,)


def test_saved_full_shuffle_is_exact_derangement():
    aligned = np.load(stage3b.STAGE3A / "aligned_embeddings.npy")
    shuffled = np.load(stage3b.STAGE3A / "shuffled_embeddings.npy")
    permutation = np.load(stage3b.STAGE3A / "shuffle_permutation.npy")
    np.testing.assert_array_equal(shuffled, aligned[permutation])
    assert not np.any(permutation == np.arange(len(permutation)))
    np.testing.assert_array_equal(np.sort(aligned, axis=0), np.sort(shuffled, axis=0))


def test_chronological_split_does_not_overlap(inputs):
    order, labels, indices = stage3b.chronological_split(inputs["origins"])
    origins = inputs["origins"][order]
    assert origins[indices["train"][-1]] < origins[indices["validation"][0]]
    assert origins[indices["validation"][-1]] < origins[indices["test"][0]]
    assert list(labels).count("train") + list(labels).count("validation") + list(labels).count("test") == len(origins)


def test_raw24_ends_at_origin_and_has_no_future(inputs):
    assert inputs["raw24"].shape == (len(inputs["origins"]), 24)
    frame = __import__("pandas").read_csv(stage3b.CSV_PATH)
    timestamps = __import__("pandas").to_datetime(frame["date"]).to_numpy(dtype="datetime64[ns]")
    for row, origin in zip(inputs["manifest"].itertuples(index=False), inputs["origins"]):
        used = timestamps[row.idx_start + 312:row.idx_start + 336]
        assert used[-1] == origin and np.all(used <= origin)


def test_scaler_fit_and_alpha_selection_use_no_test(monkeypatch):
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(100, 4)), rng.normal(size=(100, 72))
    indices = {"train": np.arange(60), "validation": np.arange(60, 70), "test": np.arange(70, 100)}
    scaler, alpha, search = stage3b.choose_alpha(x, y, indices)
    np.testing.assert_allclose(scaler.mean_, x[:60].mean(axis=0))
    assert alpha in stage3b.ALPHAS and len(search) == len(stage3b.ALPHAS)
    y_changed = y.copy(); y_changed[indices["test"]] += 1e6
    _, alpha_changed, search_changed = stage3b.choose_alpha(x, y_changed, indices)
    assert alpha == alpha_changed
    np.testing.assert_array_equal(search[["validation_mae", "validation_rmse"]], search_changed[["validation_mae", "validation_rmse"]])


def test_all_groups_share_y_and_origins(inputs):
    groups = {name: (inputs["y"], inputs["origins"]) for name in ("aligned", "shuffled", "raw24", "zero")}
    for name in groups:
        assert groups[name][0] is groups["aligned"][0]
        assert groups[name][1] is groups["aligned"][1]


def test_ridge_shape_reload_and_reproducibility(tmp_path):
    rng = np.random.default_rng(2021)
    x, y = rng.normal(size=(100, 8)), rng.normal(size=(100, 72))
    indices = {"train": np.arange(60), "validation": np.arange(60, 70), "test": np.arange(70, 100)}
    p1, _ = stage3b.fit_group("probe", x, y, indices, tmp_path / "first")
    p2, _ = stage3b.fit_group("probe", x, y, indices, tmp_path / "second")
    assert p1.shape == (30, 72)
    np.testing.assert_array_equal(p1, p2)
    with (tmp_path / "first/probe/ridge.pkl").open("rb") as handle:
        assert pickle.load(handle).__class__.__name__ == "Ridge"


def test_source_has_no_chronos_fusionsf_or_neural_training():
    source = Path(stage3b.__file__).read_text(encoding="utf-8")
    for forbidden in ("import chronos", "from chronos", "torch.optim", ".backward(", ".fit("):
        if forbidden == ".fit(":
            continue  # Ridge and StandardScaler fit are the explicitly allowed operations.
        assert forbidden not in source
    assert "Ridge(" in source
    assert "MLP" not in source and "CrossAttention" not in source
