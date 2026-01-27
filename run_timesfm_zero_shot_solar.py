#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
import timesfm

CAP = 30.1

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


def make_test_windows(y: np.ndarray, ts:np.ndarray, seq_len: int, pred_len: int, test_ratio: float = 0.3, strict_test_only: bool = False):
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
    xs, ys, t0s = [], [], []
    for s in range(start, end - seq_len - pred_len + 1):
        t_begin = s + seq_len
        t_end = t_begin + pred_len
        if t_begin < test_start:
            continue
        xs.append(y[s:s + seq_len])
        ys.append(y[t_begin:t_end])
        t0s.append(ts[t_begin])

    if not xs:
        raise RuntimeError(
            f"No test windows produced. n={n}, test_start={test_start}, "
            f"seq_len={seq_len}, pred_len={pred_len}, strict_test_only={strict_test_only}"
        )

    x = np.stack(xs, axis=0).astype(np.float32)  # [N, seq_len]
    t = np.stack(ys, axis=0).astype(np.float32)  # [N, pred_len]
    t0 = np.array(t0s)
    return x, t, t0


@torch.no_grad()
def forecast_timesfm(model, x_hist: np.ndarray, pred_len: int, batch_size: int = 64):
    """
    x_hist: [N, seq_len] float32
    return: y_pred [N, pred_len] float32
    """
    N = x_hist.shape[0]
    out = np.zeros((N, pred_len), dtype=np.float32)

    # 一次性打包成list
    inputs_all = [x_hist[i] for i in range(N)]
    q_list = []
    for i in range(0, N, batch_size):
        inputs = inputs_all[i:i + batch_size]
        point_fcst, quantile_forecast = model.forecast(horizon=pred_len, inputs=inputs)  # [B, pred_len]
        
        if i == 0:
            print("point_fcst:", np.asarray(point_fcst).shape)
            print("quantile_forecast:", np.asarray(quantile_forecast).shape, type(quantile_forecast))

        out[i:i + len(inputs)] = point_fcst.astype(np.float32)
        q_list.append(np.asarray(quantile_forecast))
    quantile_out = np.concatenate(q_list,axis=0) if len(q_list) > 0 else None
    return out, quantile_out

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    acc_mae = 1.0 - mae / CAP
    acc_rmse = 1.0 - rmse / CAP
    return mae, rmse, acc_mae, acc_rmse

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
    parser.add_argument("--batch_size", type=int, default=1024)

    parser.add_argument("--timesfm_dir", type=str, required=True, help="Local HF snapshot dir of timesfm-2.5-200m-pytorch")
    parser.add_argument("--results_root", type=str, default="./results/timesfm", help="Root output dir")

    parser.add_argument("--normalize_inputs", type=int, default=1 )
    parser.add_argument("--use_continuous_quantile_head", type=int, default=1 )
    parser.add_argument("--force_flip_invariance", type=int, default=1 )
    parser.add_argument("--infer_is_positive", type=int, default=1 )
    parser.add_argument("--fix_quantile_crossing", type=int, default=1 )
   
    args = parser.parse_args()

    # 1) 读数据（不做 StandardScaler）
    ts, y = load_series_from_csv(args.csv, args.date_col, args.target_col)

    # 2) 生成 test windows（最后30%）
    x_hist, y_true, t0 = make_test_windows(
        y=y,
        ts=ts,
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
        normalize_inputs=bool(args.normalize_inputs),  

        use_continuous_quantile_head=bool(args.use_continuous_quantile_head),
        force_flip_invariance=bool(args.force_flip_invariance),
        infer_is_positive=bool(args.infer_is_positive),
        fix_quantile_crossing=bool(args.fix_quantile_crossing),
    ))
    
    if torch.cuda.is_available():
        try:
            model = model.to("cuda")
        except Exception:
            pass

    # 4) 预测
    y_pred, y_quantile = forecast_timesfm(model, x_hist, args.pred_len, batch_size=args.batch_size)

    # 5) 计算指标
    mae, rmse, acc_mae, acc_rmse = compute_metrics(y_true, y_pred)
    print(f"[METRIC] MAE={mae:.6f} RMSE={rmse:.6f} acc_mae={acc_mae:.6f} acc_rmse={acc_rmse:.6f}")

    # 6) 保存 npy
    setting = build_setting(args.dataset_tag, args.seq_len, args.pred_len, model_tag="TimesFM")

    out_dir = os.path.join(args.results_root, setting)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred[:, :, None])  # [N, pred_len, 1]
    np.save(os.path.join(out_dir, "y_true.npy"), y_true[:, :, None])  # [N, pred_len, 1]
    np.save(os.path.join(out_dir,"t0.npy"), t0)
    np.save(os.path.join(out_dir,"y_quantile.npy"), y_quantile)

    # 7) 记录到 timesfm_result.txt（追加）
    os.makedirs(args.results_root, exist_ok=True)
    result_path = os.path.join(args.results_root, "timesfm_result.txt")
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(
            f"{setting}\t"
            f"MAE={mae:.6f}\tRMSE={rmse:.6f}\t"
            f"acc_mae={acc_mae:.6f}\tacc_rmse={acc_rmse:.6f}\t"
            f"cap={CAP}\t"
            f"N={x_hist.shape[0]}\tseq_len={args.seq_len}\tpred_len={args.pred_len}\t"
            f"batch_size={args.batch_size}\tstrict_test_only={bool(args.strict_test_only)}\t"
            f"test_ratio={args.test_ratio}\t"
            f"t0_first={str(t0[0])}\tt0_last={str(t0[-1])}\n"
        ) 
    
    print(f"Saved to: {out_dir}")
    print("y_pred:", (y_pred[:, :, None]).shape, "y_true:", (y_true[:, :, None]).shape)
    print("strict_test_only:", bool(args.strict_test_only), "test_ratio:", args.test_ratio)


if __name__ == "__main__":
    main()