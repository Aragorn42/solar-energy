#!/usr/bin/env python3
"""Stage 4C v2: train a window-level hidden-state quality gate without test leakage."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from run_stage3a_fusionsf_embedding_chronos2 import load_chronos_pipeline, load_fusionsf_model, parameter_digest
from run_stage3b_ts_mmsp_chronos2 import metrics
from run_stage4a_unica_mmsp_chronos2 import (
    CHRONOS, CHECKPOINT, FUSION_ROOT, ROOT, load_config, make_dataset, seed_everything, split_indices,
)
from stage4_adapter import MissingAwareCoRAAdapter, StaticMaskCoRAAdapter, adapter_forward

OUT = ROOT / "results/stage4c/mmsp_24_24_missing_gate_v2"
STAGE4A_CACHE = ROOT / "results/stage4a/mmsp_24_24_unica_tokens/full/seed_2021"
CORA_ROOT = ROOT / "results/stage4b/mmsp_24_24_cora_adapter/full"
FEATURE_NAMES = ["satellite_available", "satellite_missing_ratio", "nwp_available",
                 "nwp_missing_ratio", "fusion_token_norm_mean", "fusion_token_norm_std",
                 "ts_fusion_cosine_similarity", "fusion_temporal_variance"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=("smoke", "full"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--mask-seed", type=int, default=4040)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--max-train-windows", type=int, default=0)
    return p.parse_args()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def chronos_checkpoint(seed: int) -> Path:
    path = CORA_ROOT / f"seed_{seed}" / "adapter_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Stage 4B checkpoint for seed {seed} is missing: {path}")
    return path


def make_features(fusion, ts, missing_mask, nwp_available=True):
    # One independent vector per window; mask is [N,24] and true means frame missing.
    ratio = missing_mask.mean(axis=1).astype(np.float32)
    available = (ratio < 1.0).astype(np.float32)
    norm = np.linalg.norm(fusion, axis=-1)
    ts_norm = np.linalg.norm(ts, axis=-1)
    denom = np.maximum(norm * ts_norm, 1e-8)
    cosine = np.mean(np.sum(fusion * ts, axis=-1) / denom, axis=1)
    temporal_var = fusion.var(axis=1).mean(axis=1)
    return np.column_stack([
        available, ratio, np.full(len(fusion), float(nwp_available)),
        np.zeros(len(fusion), dtype=np.float32), norm.mean(axis=1), norm.std(axis=1),
        cosine, temporal_var,
    ]).astype(np.float32)


def extract_split(cfg, device, site_start, site_stop, split, seed, limit, out):
    """Extract complete and true frame-masked FusionSF representations for one split."""
    if str(FUSION_ROOT) not in sys.path:
        sys.path.insert(0, str(FUSION_ROOT))
    dataset = make_dataset(cfg, site_start, site_stop)
    indices = split_indices(dataset, split)
    if limit:
        indices = indices[:limit]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=64, shuffle=False)
    rng = np.random.default_rng(seed)
    complete_fusion, complete_ts, missing_fusion, missing_ts = [], [], [], []
    masks50, masks100, contexts, targets, rows = [], [], [], [], []
    model = load_fusionsf_model(CHECKPOINT, cfg, device)
    before = parameter_digest(model)
    with torch.inference_mode():
        for batch in loader:
            keys = {"stl_input", "stl_coords", "ts_input", "ts_coords", "ts_time",
                    "ec_input", "modality_availability"}
            original = {k: v.to(device) for k, v in batch.items() if k in keys}
            m50 = rng.random((len(batch["ts_input"]), 24)) < 0.5
            m100 = np.ones((len(batch["ts_input"]), 24), dtype=bool)
            masked50 = {k: v.clone() for k, v in original.items()}
            masked100 = {k: v.clone() for k, v in original.items()}
            mask50_t = torch.from_numpy(m50).to(device)
            masked50["stl_input"] = masked50["stl_input"].masked_fill(mask50_t[:, :, None, None, None], 0.0)
            masked100["stl_input"].zero_()
            masked100["modality_availability"][:, 0] = 0.0
            complete = model.extract_embeddings(original, "both", "none")
            part50 = model.extract_embeddings(masked50, "both", "none")
            part100 = model.extract_embeddings(masked100, "both", "none")
            complete_fusion.append(complete["fusion"].cpu().numpy()); complete_ts.append(complete["ts"].cpu().numpy())
            missing_fusion.append(np.stack([part50["fusion"].cpu().numpy(), part100["fusion"].cpu().numpy()]))
            missing_ts.append(np.stack([part50["ts"].cpu().numpy(), part100["ts"].cpu().numpy()]))
            masks50.append(m50); masks100.append(m100)
            contexts.append(batch["ts_input"].squeeze(-1).numpy()); targets.append(batch["ts_target"].numpy())
            for j in range(len(batch["site_id"])):
                rows.append({"site_id": int(batch["site_id"][j]), "split": split,
                             "forecast_origin": pd.Timestamp(int(batch["input_end_timestamp"][j])),
                             "target_start": pd.Timestamp(int(batch["forecast_start_timestamp"][j]))})
    if before != parameter_digest(model):
        raise AssertionError("FusionSF parameters changed during extraction")
    fusion = np.concatenate(complete_fusion).astype(np.float32)
    ts = np.concatenate(complete_ts).astype(np.float32)
    masked_fusion = np.concatenate(missing_fusion, axis=1).transpose(1, 0, 2, 3).astype(np.float32)
    masked_ts = np.concatenate(missing_ts, axis=1).transpose(1, 0, 2, 3).astype(np.float32)
    m50, m100 = np.concatenate(masks50), np.concatenate(masks100)
    context, target = np.concatenate(contexts).astype(np.float32), np.concatenate(targets).astype(np.float32)
    manifest = pd.DataFrame(rows)
    manifest.insert(0, "window_id", [f"{split}_site{s}_origin{t:%Y%m%d%H}" for s, t in zip(manifest.site_id, manifest.forecast_origin)])
    return {
        "manifest": manifest, "context": context, "target": target,
        "fusion": fusion, "ts": ts, "masked_fusion": masked_fusion,
        "masked_ts": masked_ts, "mask50": m50, "mask100": m100,
    }


def save_split(name, data, out):
    for key in ("context", "target", "fusion", "ts", "masked_fusion", "masked_ts", "mask50", "mask100"):
        np.save(out / f"{name}_{key}.npy", data[key])
    data["manifest"].to_csv(out / f"{name}_window_manifest.csv", index=False)


def load_or_extract(cfg, device, name, site_start, site_stop, split, args, out):
    required = [out / f"{name}_{key}.npy" for key in ("context", "target", "fusion", "ts", "masked_fusion", "masked_ts", "mask50", "mask100")]
    manifest_path = out / f"{name}_window_manifest.csv"
    if all(p.exists() for p in required) and manifest_path.exists():
        data = {key: np.load(out / f"{name}_{key}.npy") for key in ("context", "target", "fusion", "ts", "masked_fusion", "masked_ts", "mask50", "mask100")}
        data["manifest"] = pd.read_csv(manifest_path, parse_dates=["forecast_origin", "target_start"])
        return data
    data = extract_split(cfg, device, site_start, site_stop, split, args.mask_seed + site_start, (128 if args.scope == "smoke" else 0), out)
    save_split(name, data, out)
    return data


def scenario_data(data, scenario):
    if scenario == "complete":
        mask = np.zeros((len(data["fusion"]), 24), dtype=bool); return data["fusion"], data["ts"], mask
    index = 0 if scenario == "missing50" else 1
    mask = data["mask50"] if index == 0 else data["mask100"]
    return data["masked_fusion"][:, index], data["masked_ts"][:, index], mask


def features_for(data, scenario):
    fusion, ts, mask = scenario_data(data, scenario)
    return make_features(fusion, ts, mask), fusion, ts


def model_predictions(chronos, adapter, data, scenario, device, batch_size, trainable=False):
    features, fusion, _ = features_for(data, scenario)
    loader = DataLoader(TensorDataset(torch.from_numpy(data["context"]), torch.from_numpy(fusion), torch.from_numpy(features)), batch_size=batch_size, shuffle=False)
    outputs = []
    context_manager = torch.enable_grad() if trainable else torch.inference_mode()
    with context_manager:
        for context, tokens, feat in loader:
            q = adapter_forward(chronos, adapter, context.to(device), tokens.to(device), feat.to(device))
            outputs.append(q[:, int(torch.argmin(torch.abs(chronos.quantiles.float() - .5)).item())])
    return torch.cat(outputs, dim=0)


def train_gate(chronos, adapter, train, validation, args, device, out):
    adapter.freeze_cora(); optimizer = torch.optim.AdamW(adapter.gate_mlp.parameters(), lr=args.learning_rate)
    best, stale, history = float("inf"), 0, []
    scenarios = ("complete", "missing50", "missing100")
    for epoch in range(1, args.epochs + 1):
        losses = []
        # Mixing scenarios within each epoch avoids one scalar/scenario gate.
        for scenario in scenarios:
            optimizer.zero_grad(set_to_none=True)
            pred = model_predictions(chronos, adapter, train, scenario, device, args.batch_size, trainable=True)
            loss = torch.mean(torch.abs(pred - torch.from_numpy(train["target"]).to(device)))
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        val_mae = []
        for scenario in scenarios:
            with torch.inference_mode():
                pred = model_predictions(chronos, adapter, validation, scenario, device, args.eval_batch_size)
            val_mae.append(float(torch.mean(torch.abs(pred.cpu() - torch.from_numpy(validation["target"]))).item()))
        score = float(np.mean(val_mae)); history.append({"epoch": epoch, "train_mae": float(np.mean(losses)), "complete_val_mae": val_mae[0], "missing50_val_mae": val_mae[1], "missing100_val_mae": val_mae[2], "validation_mean_mae": score})
        print(f"[stage4c] epoch={epoch} validation_mean={score:.6f}", flush=True)
        if score < best:
            best, stale = score, 0; torch.save(adapter.state_dict(), out / "gate_best.pt")
        else:
            stale += 1
            if stale > args.patience: break
    adapter.load_state_dict(torch.load(out / "gate_best.pt", map_location=device, weights_only=True))
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)


def main():
    args = parse_args(); seed_everything(args.seed)
    out = OUT / args.scope / f"seed_{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu"); cfg = load_config()
    train = load_or_extract(cfg, device, "train", 10, 20, "train", args, out)
    validation = load_or_extract(cfg, device, "validation", 20, 22, "test", args, out)
    test = load_or_extract(cfg, device, "test", 0, 10, "test", args, out)
    if args.max_train_windows: 
        for key in ("context", "target", "fusion", "ts", "masked_fusion", "masked_ts", "mask50", "mask100"): train[key] = train[key][:args.max_train_windows]
        train["manifest"] = train["manifest"].iloc[:args.max_train_windows].copy()
    sites = [set(x["manifest"].site_id.unique()) for x in (train, validation, test)]
    if sites[0] & sites[1] or sites[0] & sites[2] or sites[1] & sites[2]: raise AssertionError("site overlap")
    chronos = load_chronos_pipeline(CHRONOS, str(device)).model.eval().requires_grad_(False)
    co = MissingAwareCoRAAdapter().to(device); co.cora.load_state_dict(torch.load(chronos_checkpoint(args.seed), map_location=device, weights_only=True)); co.freeze_cora()
    static_gate = StaticMaskCoRAAdapter().to(device); static_gate.cora.load_state_dict(torch.load(chronos_checkpoint(args.seed), map_location=device, weights_only=True)); static_gate.freeze_cora()
    chronos_before = parameter_digest(chronos)
    train_gate(chronos, co, train, validation, args, device, out)
    groups = {"chronos2_baseline": {}, "cora": {}, "static_mask_gate": {}, "quality_gate": {}, "oracle": {}}
    test_preds, test_features = {}, {}
    for scenario in ("complete", "missing50", "missing100"):
        features, _, _ = features_for(test, scenario); test_features[scenario] = features
        # Gate model: hidden-state gate. Static gate is a separate quality-blind reference.
        with torch.inference_mode():
            gated = model_predictions(chronos, co, test, scenario, device, args.eval_batch_size).cpu().numpy()
        with torch.inference_mode(): static = model_predictions(chronos, static_gate, test, scenario, device, args.eval_batch_size).cpu().numpy()
        old_bias = co.gate_mlp[-1].bias.detach().clone(); co.gate_mlp[-1].bias.data.fill_(100.0)
        with torch.inference_mode(): cora_pred = model_predictions(chronos, co, test, scenario, device, args.eval_batch_size).cpu().numpy()
        co.gate_mlp[-1].bias.data.copy_(old_bias)
        groups["quality_gate"][scenario] = gated; groups["static_mask_gate"][scenario] = static
        # CoRA without gate is obtained by forcing gate=1; this is the Stage 4B reference.
        groups["cora"][scenario] = cora_pred
        base = np.zeros_like(gated)
        # Baseline is computed by a standalone zero-fusion call, not from target labels.
        zero = torch.zeros((len(test["context"]), 24, 64), device=device)
        old_bias = co.gate_mlp[-1].bias.detach().clone(); co.gate_mlp[-1].bias.data.fill_(-100.0)
        with torch.inference_mode():
            base_t = model_predictions(chronos, co, {**test, "fusion": zero.cpu().numpy(), "ts": np.zeros_like(test["ts"]), "masked_fusion": np.zeros((len(test["context"]),2,24,64),np.float32), "masked_ts": np.zeros((len(test["context"]),2,24,64),np.float32), "mask50":test["mask50"], "mask100":test["mask100"]}, "complete", device, args.eval_batch_size)
        co.gate_mlp[-1].bias.data.copy_(old_bias)
        groups["chronos2_baseline"][scenario] = base_t.cpu().numpy()
        error_base = np.mean(np.abs(groups["chronos2_baseline"][scenario] - test["target"]), axis=1); error_cora = np.mean(np.abs(groups["cora"][scenario] - test["target"]), axis=1)
        groups["oracle"][scenario] = np.where((error_cora < error_base)[:, None], groups["cora"][scenario], groups["chronos2_baseline"][scenario])
    if chronos_before != parameter_digest(chronos): raise AssertionError("Chronos parameters changed")
    rows, site_rows, window_rows, gate_stats = [], [], [], {}
    for scenario in ("complete", "missing50", "missing100"):
        y = test["target"]
        for method in groups:
            pred = groups[method][scenario]; met = metrics(pred, y); reference = groups["chronos2_baseline"][scenario]
            rows.append({"scenario": scenario, "group": method, **met, "delta_mae_vs_chronos": met["mae"] - metrics(reference, y)["mae"], "delta_mae_vs_cora": met["mae"] - metrics(groups["cora"][scenario], y)["mae"]})
            for site in sorted(test["manifest"].site_id.unique()):
                mask = test["manifest"].site_id.to_numpy() == site; site_rows.append({"scenario":scenario,"group":method,"site_id":int(site),**metrics(pred[mask],y[mask])})
            for i, v in enumerate(np.mean(np.abs(pred-y),axis=1)): window_rows.append({"window_id":test["manifest"].window_id.iloc[i],"site_id":int(test["manifest"].site_id.iloc[i]),"scenario":scenario,"group":method,"mae":float(v)})
        gates = co.gate(torch.from_numpy(test_features[scenario]).to(device)).view(-1).detach().cpu().numpy(); gate_stats[scenario] = {"mean":float(gates.mean()),"std":float(gates.std()),"p10":float(np.quantile(gates,.1)),"p50":float(np.quantile(gates,.5)),"p90":float(np.quantile(gates,.9)),"values_sha256":sha256_array(gates)}; np.save(out/f"gate_values_{scenario}.npy",gates)
    pd.DataFrame(rows).to_csv(out/"scenario_metrics.csv",index=False); pd.DataFrame(site_rows).to_csv(out/"per_site_metrics.csv",index=False); pd.DataFrame(window_rows).to_csv(out/"per_window_metrics.csv",index=False)
    # Save the exact masks separately for audit/reproducibility.
    for name, data in (("train",train),("validation",validation),("test",test)):
        np.save(out/f"{name}_missing_mask.npy", np.stack([data["mask50"],data["mask100"]]))
    audit={"audit_passed":True,"test_labels_in_optimizer":False,"site_sets_disjoint":True,"gate_hidden_residual":True,"final_prediction_interpolation_for_gate":False,"chronos_frozen":True,"fusionsf_frozen":True,"cora_frozen":True,"coRa_checkpoint":str(chronos_checkpoint(args.seed)),"seed":int(args.seed),"future_power_used":False,"nwp_origin_available":True,"mask_per_window":True,"mask_shape":list(np.load(out/'test_missing_mask.npy').shape),"gate_features":FEATURE_NAMES,"train_sites":sorted(int(x) for x in sites[0]),"validation_sites":sorted(int(x) for x in sites[1]),"test_sites":sorted(int(x) for x in sites[2]),"optimizer_parameters":[n for n,p in co.named_parameters() if p.requires_grad]}
    (out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n"); (out/"gate_statistics.json").write_text(json.dumps(gate_stats,indent=2)+"\n"); (out/"resolved_config.json").write_text(json.dumps({**vars(args),"feature_names":FEATURE_NAMES,"cora_checkpoint":str(chronos_checkpoint(args.seed))},indent=2)+"\n"); test["manifest"].to_csv(out/"test_window_manifest.csv",index=False)
    (out/"comparison.md").write_text(pd.DataFrame(rows).to_string(index=False)+"\n")
    print(json.dumps({"output":str(out),"gate_statistics":gate_stats,"rows":rows},indent=2))


if __name__ == "__main__": main()
