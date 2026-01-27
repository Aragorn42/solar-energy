#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch

from chronos import BaseChronosPipeline, Chronos2Pipeline

CAP = 30.1


def build_setting(
    dataset_tag: str,
    seq_len: int,
    pred_len: int,
    model_tag: str = "Chronos2",
) -> str:
    
    return f"{dataset_tag}_{seq_len}_{pred_len}_{model_tag}__zero_shot_Exp_0"


def parse_quantiles(s: str):
    qs = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not qs:
        raise ValueError("quantiles is empty")
    for q in qs:
        if not (0.0 < q < 1.0):
            raise ValueError(f"Invalid quantile {q}, must be in (0,1)")
    return qs


def load_df_from_csv(
    csv_path: str,
    date_col: str,
    target_col: str,
    freq: str = "15min",
    fill_method: str = "interpolate",
) -> pd.DataFrame:
    """
    读取单变量时序 CSV，并强制对齐到规则频率网格，避免 Chronos infer_freq 失败：
    - 去重 timestamp
    - reindex 到 date_range(freq=...)
    - 对缺失 target 进行填充（默认插值）
    """
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"date_col='{date_col}' not found in columns: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' not found in columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # 同一 timestamp 多条记录会导致频率推断失败：只保留最后一条
    df = df.drop_duplicates(subset=[date_col], keep="last")

    # 对齐到规则网格
    df = df.set_index(date_col)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_idx)

    # 填充 target
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    if fill_method == "interpolate":
        df[target_col] = df[target_col].interpolate(limit_direction="both")
    elif fill_method == "ffill":
        df[target_col] = df[target_col].ffill().bfill()
    elif fill_method == "zero":
        df[target_col] = df[target_col].fillna(0.0)
    else:
        raise ValueError(f"Unknown fill_method={fill_method}. Use interpolate/ffill/zero")

    df = df.reset_index().rename(columns={"index": date_col})
    return df


def make_test_windows(
    y: np.ndarray,
    ts: np.ndarray,
    seq_len: int,
    pred_len: int,
    test_ratio: float = 0.3,
    strict_test_only: bool = False,
):
    """
    strict_test_only=True:
      - 输入窗口和预测窗口都必须完全落在最后 test_ratio 段内
    strict_test_only=False:
      - 预测点仍在最后 test_ratio 内，但输入窗口允许向前借 seq_len 历史
    """
    n = len(y)
    test_start = int(n * (1 - test_ratio))

    if strict_test_only:
        start = test_start
    else:
        start = max(0, test_start - seq_len)

    end = n  # 右开

    x_slices, y_slices, t0s, future_ts_slices = [], [], [], []

    for s in range(start, end - seq_len - pred_len + 1):
        t_begin = s + seq_len
        t_end = t_begin + pred_len
        if t_begin < test_start:
            continue

        x_slices.append((s, s + seq_len))
        y_slices.append((t_begin, t_end))
        t0s.append(ts[t_begin])
        future_ts_slices.append(ts[t_begin:t_end])

    if not x_slices:
        raise RuntimeError(
            f"No test windows produced. n={n}, test_start={test_start}, "
            f"seq_len={seq_len}, pred_len={pred_len}, strict_test_only={strict_test_only}"
        )

    t0 = np.array(t0s)
    future_ts = np.stack(future_ts_slices, axis=0)  # [N, pred_len]
    return x_slices, y_slices, t0, future_ts


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    acc_mae = 1.0 - mae / CAP
    acc_rmse = 1.0 - rmse / CAP
    return mae, rmse, acc_mae, acc_rmse


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--date_col", type=str, default="date")
    parser.add_argument("--target_col", type=str, default="POWER")
    parser.add_argument("--dataset_tag", type=str, required=True)

    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--test_ratio", type=float, default=0.30)
    parser.add_argument("--strict_test_only", type=int, default=1)

    # Chronos-2
    parser.add_argument("--model_name", type=str, default="/home/zhaopp/chronos-forecasting-main/chronos-2")
    parser.add_argument("--device_map", type=str, default="cuda")
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")

    # frequency fix
    parser.add_argument("--freq", type=str, default="15min", help="Resample freq, e.g. 15min/H/D")
    parser.add_argument("--fill_method", type=str, default="interpolate", help="interpolate/ffill/zero")

    parser.add_argument("--results_root", type=str, default="./results/chronos2")
    args = parser.parse_args()

    quantiles = parse_quantiles(args.quantiles)

    # 1) 读数据并强制规则频率
    df = load_df_from_csv(
        args.csv, args.date_col, args.target_col,
        freq=args.freq,
        fill_method=args.fill_method,
    )

    covariate_cols = [c for c in df.columns if c not in (args.date_col, args.target_col)]
    ts = df[args.date_col].to_numpy()
    y = df[args.target_col].to_numpy(dtype=np.float32)

    # 快速自检：频率是否可推断
    inferred = pd.infer_freq(pd.to_datetime(df[args.date_col]))
    print("[CHECK] infer_freq:", inferred, "| dup_ts:", int(df[args.date_col].duplicated().sum()))

    # 2) 生成 test windows
    x_slices, y_slices, t0, future_ts = make_test_windows(
        y=y,
        ts=ts,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        test_ratio=args.test_ratio,
        strict_test_only=bool(args.strict_test_only),
    )
    N = len(x_slices)

    # 3) 组装 context_df / future_df（每个滑窗一个 id）
    window_ids = [f"w{i:06d}" for i in range(N)]
    context_records = []
    future_records = []

    for i, wid in enumerate(window_ids):
        xs, xe = x_slices[i]
        ys_, ye_ = y_slices[i]

        ctx_part = df.iloc[xs:xe]
        for row in ctx_part.itertuples(index=False):
            rec = {
                "id": wid,
                "timestamp": getattr(row, args.date_col),
                "target": float(getattr(row, args.target_col)),
            }
            for c in covariate_cols:
                rec[c] = getattr(row, c)
            context_records.append(rec)

        fut_part = df.iloc[ys_:ye_]
        for row in fut_part.itertuples(index=False):
            rec = {
                "id": wid,
                "timestamp": getattr(row, args.date_col),
            }
            for c in covariate_cols:
                rec[c] = getattr(row, c)
            future_records.append(rec)

    context_df = pd.DataFrame.from_records(context_records)
    future_df = pd.DataFrame.from_records(future_records)

    # 关键：规避 Chronos 的 dtype/校验坑
    context_df["id"] = context_df["id"].astype("object")
    future_df["id"] = future_df["id"].astype("object")
    context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])
    future_df["timestamp"] = pd.to_datetime(future_df["timestamp"])
    context_df = context_df.sort_values(["id", "timestamp"]).drop_duplicates(["id", "timestamp"])
    future_df = future_df.sort_values(["id", "timestamp"]).drop_duplicates(["id", "timestamp"])

    # 4) 初始化并预测
    torch.set_float32_matmul_precision("high")
    pipeline = Chronos2Pipeline.from_pretrained(
    args.model_name,              # 这里传本地目录，比如 /home/zhaopp/chronos-forecasting-main/chronos-2
    device_map=args.device_map,
    local_files_only=True,
    )

    pred_df = pipeline.predict_df(
        context_df,                  # 必须是第一个参数 df（位置参数）
        future_df=future_df,
        prediction_length=args.pred_len,
        quantile_levels=quantiles,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    # 5) reshape -> [N, pred_len] / [N, pred_len, Q]
    pred_df = pred_df.sort_values(["id", "timestamp"]).reset_index(drop=True)

    y_pred = np.zeros((N, args.pred_len), dtype=np.float32)
    y_true = np.zeros((N, args.pred_len), dtype=np.float32)
    y_quantile = np.zeros((N, args.pred_len, len(quantiles)), dtype=np.float32)

    for i in range(N):
        ys_, ye_ = y_slices[i]
        y_true[i, :] = y[ys_:ye_]

    q_cols = [str(q) for q in quantiles]
    grouped = pred_df.groupby("id", sort=False)

    for i, wid in enumerate(window_ids):
        sub = grouped.get_group(wid).sort_values("timestamp").tail(args.pred_len)

        if len(sub) != args.pred_len:
            raise RuntimeError(f"Pred length mismatch for {wid}: got {len(sub)} rows, expected {args.pred_len}")

        if "predictions" not in sub.columns:
            raise RuntimeError(f"'predictions' column not found. Columns: {list(sub.columns)}")

        y_pred[i, :] = sub["predictions"].to_numpy(dtype=np.float32)

        for qi, qc in enumerate(q_cols):
            if qc not in sub.columns:
                raise RuntimeError(f"Quantile column '{qc}' not found. Columns: {list(sub.columns)}")
            y_quantile[i, :, qi] = sub[qc].to_numpy(dtype=np.float32)

    # 6) 指标
    mae, rmse, acc_mae, acc_rmse = compute_metrics(y_true, y_pred)
    print(f"[METRIC] MAE={mae:.6f} RMSE={rmse:.6f} acc_mae={acc_mae:.6f} acc_rmse={acc_rmse:.6f}")

    # 7) 保存
    setting = build_setting(
        dataset_tag=args.dataset_tag,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        model_tag="Chronos2",
    )

    out_dir = os.path.join(args.results_root, setting)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred[:, :, None])   # [N, pred_len, 1]
    np.save(os.path.join(out_dir, "y_true.npy"), y_true[:, :, None])   # [N, pred_len, 1]
    np.save(os.path.join(out_dir, "t0.npy"), t0)                       # [N]
    np.save(os.path.join(out_dir, "y_quantile.npy"), y_quantile)       # [N, pred_len, Q]
    np.save(os.path.join(out_dir, "future_ts.npy"), future_ts)         # [N, pred_len]

    meta = {
        "setting": setting,
        "cap": CAP,
        "N": int(N),
        "seq_len": int(args.seq_len),
        "pred_len": int(args.pred_len),
        "test_ratio": float(args.test_ratio),
        "strict_test_only": bool(args.strict_test_only),
        "date_col": args.date_col,
        "target_col": args.target_col,
        "covariate_cols": covariate_cols,
        "model_name": args.model_name,
        "device_map": args.device_map,
        "quantiles": quantiles,
        "freq": args.freq,
        "fill_method": args.fill_method,
        "metrics": {"mae": mae, "rmse": rmse, "acc_mae": acc_mae, "acc_rmse": acc_rmse},
        "t0_first": str(t0[0]),
        "t0_last": str(t0[-1]),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 8) 记录到 chronos2_result.txt（追加）
    os.makedirs(args.results_root, exist_ok=True)
    result_path = os.path.join(args.results_root, "chronos2_result.txt")
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(
            f"{setting}\t"
            f"MAE={mae:.6f}\tRMSE={rmse:.6f}\t"
            f"acc_mae={acc_mae:.6f}\tacc_rmse={acc_rmse:.6f}\t"
        ) 
    

if __name__ == "__main__":
    main()
