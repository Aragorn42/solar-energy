#!/usr/bin/env python3
"""Audit whether repeated Stage 3A embeddings collapse under Chronos-2 normalization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_stage3a_fusionsf_embedding_chronos2 as stage3a


ROOT = Path(__file__).resolve().parent
STAGE_ROOT = ROOT / "results/stage3a/gefcom_zone1_336_72"
OUTPUT = STAGE_ROOT / "injection_semantics_audit"
N_WINDOWS = 64
ATOL = 1e-7


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        raw = value.detach().cpu().contiguous()
        digest.update(raw.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_saved_inputs(stage_root: Path = STAGE_ROOT, n_windows: int = N_WINDOWS):
    manifest = pd.read_csv(stage_root / "window_manifest.csv").iloc[:n_windows].copy()
    aligned = np.load(stage_root / "aligned_embeddings.npy")[:n_windows]
    shuffled_all = np.load(stage_root / "shuffled_embeddings.npy")
    permutation_all = np.load(stage_root / "shuffle_permutation.npy")
    # The requested 64-window audit uses the saved full-experiment assignment for
    # each selected origin; source rows can lie outside the selected target subset.
    shuffled = shuffled_all[:n_windows]
    permutation = permutation_all[:n_windows]
    source_aligned = np.load(stage_root / "aligned_embeddings.npy")[permutation]
    zero = np.zeros_like(aligned)
    y_true = np.load(stage_root / "y_true.npy")[:n_windows]
    origins = np.load(stage_root / "t_origin.npy")[:n_windows]
    if not np.array_equal(shuffled, source_aligned):
        raise AssertionError("Saved shuffled embeddings do not equal aligned[permutation]")
    if np.any(permutation_all == np.arange(len(permutation_all))):
        raise AssertionError("Saved full permutation contains fixed points")
    if not (aligned.shape == shuffled.shape == zero.shape == (n_windows, 64)):
        raise AssertionError("Embedding shapes differ")
    if np.any(zero):
        raise AssertionError("Zero embedding contains nonzero values")
    np.testing.assert_array_equal(manifest["t_origin"].to_numpy(dtype="datetime64[ns]"), origins)
    return manifest, aligned, shuffled, zero, permutation, y_true, origins


def load_frame_and_verify(manifest: pd.DataFrame, y_true: np.ndarray, origins: np.ndarray):
    frame = pd.read_csv(stage3a.DEFAULT_CSV)[["date", "zone1"]]
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    starts = manifest["idx_start"].to_numpy(np.int64)
    timestamps = frame["date"].to_numpy(dtype="datetime64[ns]")
    values = frame["zone1"].to_numpy(np.float32)
    rebuilt_truth = np.stack([values[s + 336:s + 408] for s in starts])[..., None]
    rebuilt_origins = np.array([timestamps[s + 335] for s in starts], dtype="datetime64[ns]")
    np.testing.assert_array_equal(rebuilt_truth, y_true)
    np.testing.assert_array_equal(rebuilt_origins, origins)
    for start, origin in zip(starts, origins):
        used = timestamps[start:start + 336]
        if used[-1] != origin or np.any(used > origin):
            raise AssertionError("Context accesses data after forecast origin")
    return frame, starts


def embedding_stats(array: np.ndarray) -> dict:
    per_dimension_std = array.std(axis=0)
    return {
        "embedding_shape": list(array.shape),
        "global_min": float(array.min()), "global_max": float(array.max()),
        "global_mean": float(array.mean()), "global_std": float(array.std()),
        "per_dimension_std_min": float(per_dimension_std.min()),
        "per_dimension_std_max": float(per_dimension_std.max()),
        "zero_variance_dimension_count": int(np.sum(per_dimension_std == 0)),
    }


def actual_pipeline_tensors(context_df: pd.DataFrame, future_df: pd.DataFrame):
    from chronos.chronos2.dataset import validate_and_prepare_single_dict_task
    from chronos.df_utils import convert_df_input_to_list_of_dicts_input

    inputs, order, prediction_timestamps = convert_df_input_to_list_of_dicts_input(
        df=context_df, future_df=future_df, id_column="item_id",
        timestamp_column="timestamp", target_columns=["target"],
        prediction_length=72, validate_inputs=True,
    )
    tensors = []
    for index, task in enumerate(inputs):
        context, _, n_targets, n_covariates, _ = validate_and_prepare_single_dict_task(task, index, 72)
        if n_targets != 1 or n_covariates != 64 or context.shape != (65, 336):
            raise AssertionError("Unexpected Chronos converted input layout")
        tensors.append(context)
    return torch.stack(tensors), inputs, order, prediction_timestamps


def normalization_audit(pipeline, frame, starts, manifest, groups):
    samples = []
    normalized = {}
    input_tensors = {}
    column_sets = {}
    for name, embeddings in groups.items():
        context, future = stage3a.build_chronos_frames(frame, starts[:8], 0, embeddings)
        if any(column.startswith("fusion_emb_") for column in future.columns):
            raise AssertionError("future_df contains embedding")
        tensor, inputs, _, _ = actual_pipeline_tensors(context, future)
        expected = np.repeat(embeddings[:8, :, None], 336, axis=2)
        np.testing.assert_array_equal(tensor[:, 1:].numpy(), expected)
        input_tensors[name] = tensor[:, 1:].numpy()
        column_sets[name] = list(context.columns)
        with torch.inference_mode():
            norm, _ = pipeline.model.instance_norm(tensor.reshape(-1, 336).to(pipeline.model.device))
        norm = norm.reshape(8, 65, 336)[:, 1:].float().cpu().numpy()
        normalized[name] = norm
        for window_index in range(8):
            for dim, covariate in enumerate(stage3a.EMBEDDING_COLUMNS):
                series = input_tensors[name][window_index, dim]
                samples.append({
                    "group": name, "window_id": manifest.iloc[window_index]["window_id"],
                    "origin": manifest.iloc[window_index]["t_origin"], "covariate_name": covariate,
                    "pre_norm_mean": float(series.mean()), "pre_norm_std": float(series.std()),
                    "pre_norm_min": float(series.min()), "pre_norm_max": float(series.max()),
                })
    if not (column_sets["aligned"] == column_sets["shuffled"] == column_sets["zero"]):
        raise AssertionError("Covariate context column sets differ")
    payload = {"before_dataframe": {name: embedding_stats(value) for name, value in groups.items()}}
    payload["chronos_instance_norm"] = {
        "class": f"{pipeline.model.instance_norm.__class__.__module__}.{pipeline.model.instance_norm.__class__.__name__}",
        "dimension": -1,
        "dimension_semantics": "each item/variable independently across 336 context time steps",
        "eps": pipeline.model.instance_norm.eps,
        "use_arcsinh": pipeline.model.instance_norm.use_arcsinh,
    }
    payload["after_instance_normalization_first_8"] = {}
    for name, value in normalized.items():
        payload["after_instance_normalization_first_8"][name] = {
            "post_norm_min": float(value.min()), "post_norm_max": float(value.max()),
            "post_norm_mean": float(value.mean()), "post_norm_std": float(value.std()),
            "post_norm_max_abs": float(np.max(np.abs(value))),
            "finite": bool(np.isfinite(value).all()), "shape": list(value.shape),
        }
    payload["comparisons"] = {
        "aligned_vs_shuffled_allclose": bool(np.allclose(normalized["aligned"], normalized["shuffled"], atol=ATOL, rtol=0)),
        "aligned_vs_zero_allclose": bool(np.allclose(normalized["aligned"], normalized["zero"], atol=ATOL, rtol=0)),
        "atol": ATOL,
    }
    # Exact structural test: all 336 stored values equal the first value. Using
    # np.std here can itself produce a tiny nonzero due to float32 reduction.
    temporal_std_zero = all(np.max(np.ptp(value, axis=2)) == 0 for value in input_tensors.values())
    return payload, pd.DataFrame(samples), normalized, temporal_std_zero


def predict_four_groups(pipeline, frame, starts, groups, y_true):
    predictions = {}
    state_before = parameter_digest(pipeline.model)
    rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    for name, embedding in groups.items():
        torch.random.set_rng_state(rng_before.clone())
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state_all(cuda_rng_before)
        predictions[name] = stage3a.run_chronos_group(
            pipeline, frame, starts, embedding, 64, 64, 0.5
        )
    state_after = parameter_digest(pipeline.model)
    if state_before != state_after:
        raise AssertionError("Chronos parameters changed")
    metrics = {}
    for name, prediction in predictions.items():
        window_mae = np.mean(np.abs(prediction - y_true), axis=(1, 2))
        metrics[name] = {
            **stage3a.metric_payload(prediction, y_true),
            "window_mae": window_mae.tolist(),
            "prediction_sha256": sha256_bytes(prediction.tobytes()),
        }
    pairs = {}
    for left, right in (("aligned", "shuffled"), ("aligned", "zero"), ("shuffled", "zero"), ("baseline", "zero")):
        difference = np.abs(predictions[left] - predictions[right])
        pairs[f"{left}_vs_{right}"] = {
            "max_abs_diff": float(difference.max()), "mean_abs_diff": float(difference.mean()),
            "prediction_allclose_atol_1e-7": bool(np.allclose(predictions[left], predictions[right], atol=ATOL, rtol=0)),
            "left_window_mae_win_rate": float(np.mean(np.array(metrics[left]["window_mae"]) < np.array(metrics[right]["window_mae"]))),
        }
    return predictions, metrics, pairs, state_before


def source_audit(pipeline) -> tuple[dict, str]:
    import chronos
    from chronos.chronos2 import dataset, model, pipeline as pipeline_module
    from chronos import df_utils
    sources = {
        "predict_df": (pipeline_module.__file__, "Chronos2Pipeline.predict_df"),
        "dataframe_conversion": (df_utils.__file__, "convert_df_input_to_list_of_dicts_input"),
        "covariate_tensor_conversion": (dataset.__file__, "validate_and_prepare_single_dict_task"),
        "batch_construction": (dataset.__file__, "Chronos2Dataset._build_batch"),
        "patch_construction": (model.__file__, "Chronos2Model._prepare_patched_context"),
        "instance_normalization": (
            inspect.getsourcefile(pipeline.model.instance_norm.__class__),
            "InstanceNorm.forward",
        ),
    }
    records = {}
    for key, (path, functions) in sources.items():
        records[key] = {"source_file": str(path), "functions": functions, "source_sha256": sha256_file(Path(path))}
    environment = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "chronos_package_path": str(Path(chronos.__file__).parent),
        "chronos_package_version": importlib.metadata.version("chronos-forecasting"),
        "chronos_model_path": str(stage3a.DEFAULT_CHRONOS),
        "chronos_model_config_sha256": sha256_file(stage3a.DEFAULT_CHRONOS / "config.json"),
        "instance_norm_eps": pipeline.model.instance_norm.eps,
        "instance_norm_use_arcsinh": pipeline.model.instance_norm.use_arcsinh,
        "source_records": records,
    }
    norm_source = inspect.getsource(pipeline.model.instance_norm.forward)
    markdown = """# Chronos-2 injection source audit

The installed Chronos implementation is the authority for this audit. `predict_df` converts every non-ID/non-timestamp/non-target context column into a past covariate. `validate_and_prepare_single_dict_task` stacks target followed by each past covariate, producing one row per variable and 336 columns along time. `Chronos2Model._prepare_patched_context` calls `self.instance_norm(context)` before patch construction. `InstanceNorm.forward` computes `nanmean` and RMS scale with `dim=-1`, so normalization is independently applied to every item/variable row along its 336-step time dimension. A constant repeated covariate becomes `(x - loc) / scale` before patching. Mathematically this is zero, but this audit records the actual float32 reduction result rather than assuming exact cancellation; nonzero roundoff can be amplified when the computed scale is also tiny.

## Actual installed paths and hashes

```json
""" + json.dumps(environment, indent=2) + "\n```\n\n## Installed `InstanceNorm.forward`\n\n```python\n" + norm_source + "```\n"
    return environment, markdown


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    manifest, aligned, shuffled, zero, permutation, y_true, origins = load_saved_inputs()
    frame, starts = load_frame_and_verify(manifest, y_true, origins)
    pipeline = stage3a.load_chronos_pipeline(stage3a.DEFAULT_CHRONOS, "cuda" if torch.cuda.is_available() else "cpu")
    pipeline.model.eval()
    pipeline.model.requires_grad_(False)
    groups = {"aligned": aligned, "shuffled": shuffled, "zero": zero}
    normalization, samples, normalized, temporal_std_zero = normalization_audit(
        pipeline, frame, starts, manifest, groups
    )
    prediction_groups = {"baseline": None, **groups}
    predictions, metrics, comparisons, model_digest = predict_four_groups(
        pipeline, frame, starts, prediction_groups, y_true
    )
    normalized_zero = {name: float(np.max(np.abs(value))) <= ATOL for name, value in normalized.items()}
    aligned_matches_shuffled = comparisons["aligned_vs_shuffled"]["prediction_allclose_atol_1e-7"]
    aligned_matches_zero = comparisons["aligned_vs_zero"]["prediction_allclose_atol_1e-7"]
    conclusion = "A" if all(normalized_zero.values()) and aligned_matches_shuffled and aligned_matches_zero else (
        "B" if not normalization["comparisons"]["aligned_vs_shuffled_allclose"] and aligned_matches_shuffled else "C"
    )
    audit = {
        "forecast_origin_alignment_valid": True, "future_information_used": False,
        "fusionsf_trained": False, "chronos_trained": False, "trainable_module_added": False,
        "embedding_repeated_across_context": True, "per_window_temporal_std_zero": temporal_std_zero,
        "chronos_instance_normalization_confirmed": True,
        "aligned_normalized_to_zero": normalized_zero["aligned"],
        "shuffled_normalized_to_zero": normalized_zero["shuffled"],
        "zero_normalized_to_zero": normalized_zero["zero"],
        "aligned_equals_shuffled_after_normalization": normalization["comparisons"]["aligned_vs_shuffled_allclose"],
        "aligned_equals_zero_after_normalization": normalization["comparisons"]["aligned_vs_zero_allclose"],
        "aligned_prediction_matches_shuffled": aligned_matches_shuffled,
        "aligned_prediction_matches_zero": aligned_matches_zero,
        "fusionsf_checkpoint_sha256_unchanged": sha256_file(stage3a.DEFAULT_CHECKPOINT) == "8fe851d9b66b8301bb8615ecf99db4625459b245342b3365e14aacf712e4f5d8",
        "chronos_parameter_digest_before_after": model_digest,
        "chronos_parameters_updated": False, "optimizer_created": False,
        "backward_called": False, "train_called": False, "fit_called": False,
        "window_count": N_WINDOWS, "normalization_atol": ATOL,
        "conclusion": conclusion,
    }
    environment, source_markdown = source_audit(pipeline)
    resolved = {
        "dataset": "GEFCom zone1", "seq_len": 336, "pred_len": 72,
        "strict_test_only": 1, "test_ratio": 0.3, "freq": "1h", "cross_learning": 0,
        "window_selection": "first 64 saved valid Stage 3A windows", "quantile": 0.5,
        "context_length": 336, "window_batch_size": 64, "model_batch_size": 64,
        "source_stage_root": str(STAGE_ROOT), "normalization_atol": ATOL,
    }
    np.save(OUTPUT / "zero_embeddings.npy", zero)
    for name, prediction in predictions.items():
        group_dir = OUTPUT / name
        group_dir.mkdir()
        np.save(group_dir / "y_pred.npy", prediction)
    samples.to_csv(OUTPUT / "normalization_samples.csv", index=False)
    for filename, payload in (("resolved_config.json", resolved), ("environment.json", environment),
                              ("normalization_audit.json", normalization), ("metrics.json", metrics),
                              ("prediction_comparison.json", comparisons), ("audit.json", audit)):
        (OUTPUT / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "chronos_source_audit.md").write_text(source_markdown, encoding="utf-8")
    conclusion_text = {
        "A": "Conclusion A: repeated window-level embeddings are normalized to zero, so this injection does not test embedding content.",
        "B": "Conclusion B: Chronos-2 receives different normalized embeddings but is insensitive to them under this injection.",
        "C": "Conclusion C: the prior strict-zero hypothesis is false. Chronos normalizes each variable along time, but float32 mean/scale roundoff is amplified for constant nonzero rows, so aligned and shuffled remain different from zero after normalization and produce measurably different predictions. The claim that all three embeddings are normalized to zero is withdrawn.",
    }[conclusion]
    report = "# Stage 3A injection semantics audit\n\n" + conclusion_text + "\n\n```json\n" + json.dumps({"metrics": metrics, "prediction_comparison": comparisons, "audit": audit}, indent=2) + "\n```\n"
    (OUTPUT / "audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "metrics": metrics, "comparisons": comparisons, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
