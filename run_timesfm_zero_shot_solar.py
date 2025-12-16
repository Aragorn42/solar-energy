#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
import timesfm



def build_setting(dataset_tag: str, seq_len: int, pred_len: int, model_tag: str = "TimesFM") -> str:
    # 指标解析用的前缀：DATASET_SEQ_PRED_MODEL
    return f"{dataset_tag}_{seq_len}_{pred_len}_{model_tag}_zero_shot_Exp_0"


def load_series_from_csv(csv_path: str, date_col: str, target_col: str):
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"date_col='{date_col}' not found in columns: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' not found in columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    ts = df[date_col].to_numpy()
    y = df[target_col].to_numpy(dtype=np.float32)
    return ts, y


def make_test_windows(y: np.ndarray, seq_len: int, pred_len: int, test_ratio: float = 0.3, strict_test_only: bool = False):
    """
    strict_test_only=True:
      - 输入窗口和预测窗口都必须完全落在最后30% test段内
      - 代价：可用窗口更少

    strict_test_only=False:
      - 预测点仍在最后30%内，但输入窗口允许向前借 seq_len 历史（更常见的评测方式）
    """
    n = len(y)
    test_start = int(n * (1 - test_ratio))

    if strict_test_only:
        start = test_start
    else:
        start = max(0, test_start - seq_len)

    end = n  # 右开

    # 生成所有可用的滑窗起点 s，使得：
    #  input:  y[s : s+seq_len]
    #  target: y[s+seq_len : s+seq_len+pred_len]
    # 并且 target 段必须落在 test 段内（>= test_start）
    xs, ys = [], []
    for s in range(start, end - seq_len - pred_len + 1):
        t_begin = s + seq_len
        t_end = t_begin + pred_len
        if t_begin < test_start:
            continue
        xs.append(y[s:s + seq_len])
        ys.append(y[t_begin:t_end])

    if not xs:
        raise RuntimeError(
            f"No test windows produced. n={n}, test_start={test_start}, "
            f"seq_len={seq_len}, pred_len={pred_len}, strict_test_only={strict_test_only}"
        )

    x = np.stack(xs, axis=0).astype(np.float32)  # [N, seq_len]
    t = np.stack(ys, axis=0).astype(np.float32)  # [N, pred_len]
    return x, t


@torch.no_grad()
def forecast_timesfm(model, x_hist: np.ndarray, pred_len: int, batch_size: int = 64):
    """
    x_hist: [N, seq_len] float32
    return: y_pred [N, pred_len] float32
    """
    N = x_hist.shape[0]
    out = np.zeros((N, pred_len), dtype=np.float32)

    for i in range(0, N, batch_size):
        chunk = x_hist[i:i + batch_size]
        inputs = [chunk[j].astype(np.float32) for j in range(chunk.shape[0])]
        point_fcst, _ = model.forecast(horizon=pred_len, inputs=inputs)  # [B, pred_len]
        out[i:i + chunk.shape[0]] = point_fcst.astype(np.float32)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV (must include date and target columns)")
    parser.add_argument("--date_col", type=str, default="date", help="Datetime column name")
    parser.add_argument("--target_col", type=str, default="POWER", help="Target column name (PV power)")
    parser.add_argument("--dataset_tag", type=str, required=True, help="e.g. GEFCom_TASK15 or CSGS1 etc. Used in folder name prefix")
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--test_ratio", type=float, default=0.30)
    parser.add_argument("--strict_test_only", type=int, default=1, help="1: input+target both in last30%; 0: target in last30%, input can look back")
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--timesfm_dir", type=str, required=True, help="Local HF snapshot dir of timesfm-2.5-200m-pytorch")
    parser.add_argument("--results_root", type=str, default="./results/solar", help="Root output dir")
    args = parser.parse_args()

    # 1) 读数据（不做 StandardScaler）
    ts, y = load_series_from_csv(args.csv, args.date_col, args.target_col)

    # 2) 生成 test windows（最后30%）
    x_hist, y_true = make_test_windows(
        y=y,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        test_ratio=args.test_ratio,
        strict_test_only=bool(args.strict_test_only),
    )

    # 3) 初始化 TimesFM（离线）
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.timesfm_dir, local_files_only=True)
    model.compile(timesfm.ForecastConfig(
        max_context=max(1024, args.seq_len),
        max_horizon=max(512, args.pred_len),
        normalize_inputs=True,   # 原始功率输入 -> 建议 True
    ))

    # 4) 预测
    y_pred = forecast_timesfm(model, x_hist, args.pred_len, batch_size=args.batch_size)

    # 5) 保存 npy
    setting = build_setting(args.dataset_tag, args.seq_len, args.pred_len, model_tag="TimesFM")
    out_dir = os.path.join(args.results_root, setting)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred[:, :, None].astype(np.float32))  # [N, pred_len, 1]
    np.save(os.path.join(out_dir, "y_true.npy"), y_true[:, :, None].astype(np.float32))  # [N, pred_len, 1]

    print(f"Saved to: {out_dir}")
    print("y_pred:", (y_pred[:, :, None]).shape, "y_true:", (y_true[:, :, None]).shape)
    print("strict_test_only:", bool(args.strict_test_only), "test_ratio:", args.test_ratio)


if __name__ == "__main__":
    main()