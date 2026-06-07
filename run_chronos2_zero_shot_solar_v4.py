#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chronos-2 zero-shot rolling forecast (Solar) — robust version

Fixes:
1) Always call Chronos2Pipeline.predict_df with the correct signature: predict_df(df, future_df=..., ...)
2) Use consistent column names: item_id / timestamp / target
3) Handle irregular timestamps by skipping irregular windows (default), based on --freq or inferred mode delta
4) Force timestamp dtype to datetime64[ns] and sort by (item_id, timestamp)
"""

import os
import json
import re
import math
import argparse
import warnings
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from chronos import BaseChronosPipeline, Chronos2Pipeline


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--date_col", type=str, default="date")
    p.add_argument("--target_col", type=str, default="OT")
    p.add_argument("--dataset_tag", type=str, required=True)

    # rolling windows
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--pred_len", type=int, required=True)
    p.add_argument("--test_ratio", type=float, default=0.3)
    p.add_argument("--strict_test_only", type=int, default=1)

    # model
    p.add_argument("--chronos_model_dir", type=str, required=True)
    p.add_argument("--device_map", type=str, default="cuda")
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--cross_learning", type=int, default=0)
    p.add_argument("--model_batch_size", type=int, default=64)   # predict_df(batch_size=...)
    p.add_argument("--window_batch_size", type=int, default=256) # windows per predict_df call

    # covariates
    p.add_argument("--past_covariate_cols", type=str, default="")
    p.add_argument("--future_covariate_cols", type=str, default="")
    p.add_argument("--normalize_colnames", type=int, default=1,
                   help="If 1, strip and collapse whitespace in CSV headers and covariate names to avoid mismatches.")

    # output quantiles
    p.add_argument("--quantile_levels", type=str, default="0.1,0.5,0.9")
    p.add_argument("--save_all_requested_quantiles", type=int, default=0)

    # freq handling
    p.add_argument("--freq", type=str, default="")  # e.g. "15min", "H"
    p.add_argument("--skip_irregular_windows", type=int, default=1)

    # results
    p.add_argument("--results_root", type=str, default="./results")
    p.add_argument("--run_name", type=str, default="")

    return p.parse_args()


def _split_cols(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [c.strip() for c in s.split(",") if c.strip()]



def _norm_colname(x: str) -> str:
    # Collapse consecutive whitespace and strip ends
    return re.sub(r"\s+", " ", str(x)).strip()

def _ensure_datetime64_ns(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    return ts.astype("datetime64[ns]")


def _infer_expected_delta(timestamps: np.ndarray) -> Optional[np.timedelta64]:
    if len(timestamps) < 3:
        return None
    diffs = np.diff(timestamps.astype("datetime64[ns]"))
    diffs = diffs[diffs > np.timedelta64(0, "ns")]
    if len(diffs) == 0:
        return None
    vals, counts = np.unique(diffs, return_counts=True)
    return vals[np.argmax(counts)]


def _delta_from_freq(freq: str) -> Optional[np.timedelta64]:
    if not freq:
        return None
    try:
        td = pd.to_timedelta(freq)
        return np.timedelta64(int(td.value), "ns")
    except Exception:
        try:
            off = pd.tseries.frequencies.to_offset(freq)
            # Use .nanos to avoid deprecated .delta
            td = pd.Timedelta(off.nanos, unit='ns')
            return np.timedelta64(int(td.value), "ns")
        except Exception:
            return None


def build_windows(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    seq_len: int,
    pred_len: int,
    test_ratio: float,
    strict_test_only: int,
    expected_delta: Optional[np.timedelta64],
    skip_irregular_windows: bool,
) -> Dict[str, np.ndarray]:
    n = len(df)
    n_test = int(math.ceil(n * test_ratio))
    test_start = n - n_test

    max_i = n - (seq_len + pred_len)
    if max_i < 0:
        raise ValueError(f"Not enough data: n={n}, need >= {seq_len+pred_len}")

    starts = []
    for i in range(0, max_i + 1):
        hist_l = i
        hist_r = i + seq_len
        fut_l = hist_r
        fut_r = fut_l + pred_len

        if strict_test_only:
            if hist_l < test_start:
                continue
        else:
            if fut_l < test_start:
                continue
        if fut_r > n:
            continue

        if skip_irregular_windows and expected_delta is not None:
            ts_span = df[date_col].values[hist_l:fut_r].astype("datetime64[ns]")
            d = np.diff(ts_span)
            if len(d) == 0 or np.any(d != expected_delta):
                continue

        starts.append(i)

    starts = np.array(starts, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError("No valid windows. Try relaxing strict_test_only or disable skip_irregular_windows.")

    y = df[target_col].values.astype(np.float32)
    y_true = np.stack([y[i+seq_len:i+seq_len+pred_len] for i in starts], axis=0)
    t_origin = df[date_col].values[starts + seq_len - 1].astype("datetime64[ns]")

    return {"idx_start": starts, "y_true": y_true, "t_origin": t_origin}


def main():
    args = parse_args()
    os.makedirs(args.results_root, exist_ok=True)

    past_cov_cols = _split_cols(args.past_covariate_cols)
    future_cov_cols = _split_cols(args.future_covariate_cols)

    df = pd.read_csv(args.csv)
    if int(args.normalize_colnames):
        # normalize CSV headers (strip + collapse spaces)
        df = df.rename(columns={c: _norm_colname(c) for c in df.columns})
        args.date_col = _norm_colname(args.date_col)
        args.target_col = _norm_colname(args.target_col)
        past_cov_cols[:] = [_norm_colname(c) for c in past_cov_cols]
        future_cov_cols[:] = [_norm_colname(c) for c in future_cov_cols]
    for c in [args.date_col, args.target_col]:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in CSV")
    for c in past_cov_cols + future_cov_cols:
        if c and c not in df.columns:
            cols_repr = ', '.join([repr(x) for x in df.columns.tolist()])
            raise KeyError(
                f"Covariate column not found: {repr(c)}. "
                f"Tip: CSV headers often contain trailing/multiple spaces. "
                f"Re-run with --normalize_colnames 1 (default). "
                f"CSV columns are: [{cols_repr}]"
            )

    df[args.date_col] = _ensure_datetime64_ns(df[args.date_col])
    df = df.sort_values(args.date_col).reset_index(drop=True)

    # drop duplicate timestamps
    if df[args.date_col].duplicated().any():
        warnings.warn("Duplicate timestamps found; dropping duplicates (keep='last').")
        df = df.drop_duplicates(subset=[args.date_col], keep="last").reset_index(drop=True)

    expected_delta = _delta_from_freq(args.freq) or _infer_expected_delta(df[args.date_col].values)
    if expected_delta is None:
        warnings.warn("Could not infer expected delta. Irregular-window skipping will be disabled.")
    else:
        print(f"[Chronos2] expected_delta = {expected_delta} (ns)")

    windows = build_windows(
        df=df,
        date_col=args.date_col,
        target_col=args.target_col,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        test_ratio=args.test_ratio,
        strict_test_only=int(args.strict_test_only),
        expected_delta=expected_delta,
        skip_irregular_windows=bool(args.skip_irregular_windows),
    )

    starts = windows["idx_start"]
    y_true = windows["y_true"].astype(np.float32)[..., None]
    t_origin = windows["t_origin"]
    N = len(starts)
    print(f"[Chronos2] windows constructed: N={N}, seq_len={args.seq_len}, pred_len={args.pred_len}")

    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.chronos_model_dir,
        device_map=args.device_map,
    )

    quantile_levels = [float(x) for x in args.quantile_levels.split(",") if x.strip()]
    if not quantile_levels:
        quantile_levels = [0.5]

    y_pred = np.zeros((N, args.pred_len, 1), dtype=np.float32)
    save_q = bool(args.save_all_requested_quantiles)
    all_q = None
    if save_q:
        all_q = np.zeros((N, args.pred_len, len(quantile_levels)), dtype=np.float32)

    num_batches = math.ceil(N / args.window_batch_size)

    for bi in range(num_batches):
        l = bi * args.window_batch_size
        r = min((bi + 1) * args.window_batch_size, N)
        batch_starts = starts[l:r]

        rows_ctx = []
        rows_fut = []

        for j, s in enumerate(batch_starts):
            wid = f"w{l+j}"  # stable id for this batch range
            hist_l = int(s)
            hist_r = hist_l + args.seq_len
            fut_l = hist_r
            fut_r = fut_l + args.pred_len

            hist_slice = df.iloc[hist_l:hist_r]
            fut_slice = df.iloc[fut_l:fut_r]

            # context rows: item_id, timestamp, target (+ past cov)
            ts_hist = hist_slice[args.date_col].values.astype("datetime64[ns]")
            y_hist = hist_slice[args.target_col].values.astype(np.float32)
            for k in range(args.seq_len):
                row = {"item_id": wid, "timestamp": ts_hist[k], "target": float(y_hist[k])}
                # past covariates
                for c in past_cov_cols:
                    row[c] = float(hist_slice[c].values.astype(np.float32)[k])
                # future covariates must also exist in df columns (Chronos validation requirement)
                for c in future_cov_cols:
                    # row[c] = np.nan
                    row[c] = float(hist_slice[c].values.astype(np.float32)[k])
                rows_ctx.append(row)

            # future rows: item_id, timestamp (+ future cov)
            ts_fut = fut_slice[args.date_col].values.astype("datetime64[ns]")
            for k in range(args.pred_len):
                row = {"item_id": wid, "timestamp": ts_fut[k]}
                # keep column set consistent with df
                for c in future_cov_cols:
                    row[c] = float(fut_slice[c].values.astype(np.float32)[k])
                rows_fut.append(row)

        context_df = pd.DataFrame(rows_ctx)
        future_df = pd.DataFrame(rows_fut)

        # Safety: Chronos requires future_df columns ⊆ df columns
        for c in past_cov_cols + future_cov_cols:
            if c not in context_df.columns:
                context_df[c] = np.nan

        for c in future_cov_cols:               
            if c not in future_df.columns:
                future_df[c] = np.nan

        # enforce dtype + order
        context_df["timestamp"] = _ensure_datetime64_ns(context_df["timestamp"])
        context_df = context_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
        future_df["timestamp"] = _ensure_datetime64_ns(future_df["timestamp"])
        future_df = future_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

        kwargs = dict(
            future_df=future_df,
            prediction_length=args.pred_len,
            quantile_levels=quantile_levels,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            batch_size=args.model_batch_size,
            validate_inputs=True,
            cross_learning=bool(args.cross_learning),
        )
        if args.context_length is not None:
            kwargs["context_length"] = int(args.context_length)

        # IMPORTANT: correct signature -> first arg is df
        pred_df = pipeline.predict_df(context_df, **kwargs)

        # Quantile columns in output are often like "0.1", "0.5", "0.9"
        qcols = [c for c in pred_df.columns if c not in ["item_id", "timestamp", "target_name", "predictions"]]
        col_for_q = {}
        for q in quantile_levels:
            k = str(q)
            if k in qcols:
                col_for_q[q] = k
        point_col = col_for_q.get(0.5, qcols[0] if qcols else "0.5")

        for j in range(l, r):
            wid = f"w{j}"
            sub = pred_df[pred_df["item_id"] == wid].sort_values("timestamp")
            vals = sub[point_col].values.astype(np.float32)
            if len(vals) != args.pred_len:
                raise RuntimeError(f"Pred length mismatch for {wid}: got {len(vals)} expected {args.pred_len}")
            y_pred[j, :, 0] = vals
            if save_q and all_q is not None:
                for qi, q in enumerate(quantile_levels):
                    col = col_for_q.get(q, point_col)
                    all_q[j, :, qi] = sub[col].values.astype(np.float32)

        print(f"[Chronos2] batch {bi+1}/{num_batches} done: windows {l}-{r-1}")

    if args.run_name:
        out_dir = os.path.join(args.results_root, args.run_name)
    else:
        out_dir = os.path.join(args.results_root, f"{args.dataset_tag}_sl{args.seq_len}_pl{args.pred_len}")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(out_dir, "y_true.npy"), y_true)
    np.save(os.path.join(out_dir, "t_origin.npy"), t_origin.astype("datetime64[ns]"))
    if save_q and all_q is not None:
        np.save(os.path.join(out_dir, "pred_quantiles.npy"), all_q)

    meta = {
        "csv": args.csv,
        "dataset_tag": args.dataset_tag,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "test_ratio": args.test_ratio,
        "strict_test_only": int(args.strict_test_only),
        "freq_arg": args.freq,
        "expected_delta_ns": int(expected_delta.astype("timedelta64[ns]").astype(np.int64)) if expected_delta is not None else None,
        "skip_irregular_windows": int(args.skip_irregular_windows),
        "past_covariate_cols": past_cov_cols,
        "future_covariate_cols": future_cov_cols,
        "quantile_levels": quantile_levels,
        "cross_learning": int(args.cross_learning),
        "model_batch_size": args.model_batch_size,
        "window_batch_size": args.window_batch_size,
        "chronos_model_dir": args.chronos_model_dir,
        "device_map": args.device_map,
        "context_length": args.context_length,
        "num_windows": int(N),
        "run_name": args.run_name,
        "saved_dir": out_dir,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Chronos2] Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
