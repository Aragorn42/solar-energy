#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline


def build_setting(args, model_tag: str = "Chronos2") -> str:
    # 避免不同参数覆盖同目录
    qtag = args.quantiles.replace(",", "-")
    return (
        f"{args.dataset_tag}_{args.seq_len}_{args.pred_len}_{model_tag}_zero_shot"
        f"_q-{qtag}"
        f"_strict-{int(bool(args.strict_test_only))}"
        f"_cal-{int(args.do_calib)}-r{args.calib_ratio}"
        f"_freq-{args.freq}"
        f"_fill-{args.fill_method}"
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


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
    读取单变量时序 CSV，并对齐到规则频率网格：
    - 去重 timestamp
    - reindex 到 date_range(freq=...)
    - target 缺失填充（默认插值）
    """
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"date_col='{date_col}' not found in columns: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' not found in columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # 同一 timestamp 多条记录 -> 只保留最后一条
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


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, cap: float):
    err = (y_pred - y_true).astype(np.float64)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    acc_mae = 1.0 - mae / cap
    acc_rmse = 1.0 - rmse / cap
    return mae, rmse, acc_mae, acc_rmse


def fit_affine_ls(yhat: np.ndarray, ytru: np.ndarray):
    """最小二乘拟合 y ≈ a*x + b"""
    x = yhat.astype(np.float64).reshape(-1)
    y = ytru.astype(np.float64).reshape(-1)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def pick_q_colname(df_cols, q: float):
    """
    兼容 float/str 两种列名，避免不同版本/输出格式导致取不到列
    """
    # 1) float 直接匹配
    if q in df_cols:
        return q
    # 2) str(q)
    s1 = str(q)
    if s1 in df_cols:
        return s1
    # 3) 紧凑格式
    s2 = f"{q:g}"
    if s2 in df_cols:
        return s2
    # 4) 试一下 1 位/2 位小数
    s3 = f"{q:.1f}"
    if s3 in df_cols:
        return s3
    s4 = f"{q:.2f}"
    if s4 in df_cols:
        return s4
    raise KeyError(f"Quantile column not found for q={q}. Available cols: {list(df_cols)[:30]}...")


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

    # affine calibration
    parser.add_argument("--cap", type=float, default=30.1)
    parser.add_argument("--do_calib", type=int, default=0, help="1: do affine calibration on test windows")
    parser.add_argument("--calib_ratio", type=float, default=0.5,
                        help="fraction of test windows used for calibration (time-ordered, first part)")
    parser.add_argument("--calib_on", type=str, default="rmse", choices=["rmse"],
                        help="rmse: fit a,b by least squares")

    # optional: append a csv log for easy comparison
    parser.add_argument("--log_csv", type=str, default="",
                        help="If set, append one row per run to this CSV (recommended).")

    args = parser.parse_args()

    cap = float(args.cap)
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

    inferred = pd.infer_freq(pd.to_datetime(df[args.date_col]))
    print("[CHECK] infer_freq:", inferred, "| dup_ts:", int(df[args.date_col].duplicated().sum()))
    print("[CHECK] covariate_cols:", covariate_cols)

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
    print("[INFO] num_windows:", N)

    # 3) 组装 context_df / future_df（每个滑窗一个 id）
    window_ids = [f"w{i:06d}" for i in range(N)]
    context_records = []
    future_records = []

    # NOTE: 这一步对大 N 会非常慢/占内存（你已知），用于验证方法OK
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

    # dtype / 排序 / 去重
    context_df["id"] = context_df["id"].astype("object")
    future_df["id"] = future_df["id"].astype("object")
    context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])
    future_df["timestamp"] = pd.to_datetime(future_df["timestamp"])

    context_df = context_df.sort_values(["id", "timestamp"]).drop_duplicates(["id", "timestamp"], keep="last")
    future_df = future_df.sort_values(["id", "timestamp"]).drop_duplicates(["id", "timestamp"], keep="last")

    # 4) 初始化并预测
    torch.set_float32_matmul_precision("high")
    pipeline = Chronos2Pipeline.from_pretrained(
        args.model_name,
        device_map=args.device_map,
        local_files_only=True,
    )

    pred_df = pipeline.predict_df(
        context_df,  # 位置参数
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

    if "predictions" not in pred_df.columns:
        raise RuntimeError(f"'predictions' column not found. Columns: {list(pred_df.columns)}")

    # 分位数列名兼容
    q_cols = [pick_q_colname(pred_df.columns, q) for q in quantiles]

    grouped = pred_df.groupby("id", sort=False)

    for i, wid in enumerate(window_ids):
        sub = grouped.get_group(wid).sort_values("timestamp").head(args.pred_len)

        if len(sub) != args.pred_len:
            raise RuntimeError(f"Pred length mismatch for {wid}: got {len(sub)} rows, expected {args.pred_len}")

        y_pred[i, :] = sub["predictions"].to_numpy(dtype=np.float32)

        for qi, qc in enumerate(q_cols):
            y_quantile[i, :, qi] = sub[qc].to_numpy(dtype=np.float32)

    # 6) 指标（raw 全部 test windows）
    mae_raw_all, rmse_raw_all, acc_mae_raw_all, acc_rmse_raw_all = compute_metrics(y_pred, y_true, cap)
    print(f"[RAW-ALL] MAE={mae_raw_all:.6f} RMSE={rmse_raw_all:.6f} acc_mae={acc_mae_raw_all:.6f} acc_rmse={acc_rmse_raw_all:.6f}")

    # 7) affine calibration（时间排序后，前半拟合、后半评测）
    a_cal = b_cal = None
    mae_raw_test = rmse_raw_test = acc_mae_raw_test = acc_rmse_raw_test = None
    mae_cal_test = rmse_cal_test = acc_mae_cal_test = acc_rmse_cal_test = None

    if int(args.do_calib) == 1:
        order = np.argsort(pd.to_datetime(t0).astype("datetime64[ns]"))
        yhat_s = y_pred[order]
        ytru_s = y_true[order]

        n = len(yhat_s)
        n_cal = int(n * float(args.calib_ratio))
        n_cal = max(1, min(n - 1, n_cal))

        yhat_cal, ytru_cal = yhat_s[:n_cal], ytru_s[:n_cal]
        yhat_test, ytru_test = yhat_s[n_cal:], ytru_s[n_cal:]

        a_cal, b_cal = fit_affine_ls(yhat_cal, ytru_cal)

        mae_raw_test, rmse_raw_test, acc_mae_raw_test, acc_rmse_raw_test = compute_metrics(yhat_test, ytru_test, cap)
        yhat_test_cal = a_cal * yhat_test + b_cal
        mae_cal_test, rmse_cal_test, acc_mae_cal_test, acc_rmse_cal_test = compute_metrics(yhat_test_cal, ytru_test, cap)

        print(f"[CAL] fitted a={a_cal:.6f}, b={b_cal:.6f}, calib_ratio={args.calib_ratio}")
        print(f"[RAW-TEST] MAE={mae_raw_test:.6f} RMSE={rmse_raw_test:.6f} acc_mae={acc_mae_raw_test:.6f} acc_rmse={acc_rmse_raw_test:.6f}")
        print(f"[CAL-TEST] MAE={mae_cal_test:.6f} RMSE={rmse_cal_test:.6f} acc_mae={acc_mae_cal_test:.6f} acc_rmse={acc_rmse_cal_test:.6f}")

    # 8) 保存
    setting = build_setting(args, model_tag="Chronos2")
    out_dir = os.path.join(args.results_root, setting)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred[:, :, None])         # [N, pred_len, 1]
    np.save(os.path.join(out_dir, "y_true.npy"), y_true[:, :, None])         # [N, pred_len, 1]
    np.save(os.path.join(out_dir, "t0.npy"), t0)                             # [N]
    np.save(os.path.join(out_dir, "y_quantile.npy"), y_quantile)             # [N, pred_len, Q]
    np.save(os.path.join(out_dir, "future_ts.npy"), future_ts)               # [N, pred_len]

    meta = {
        "setting": setting,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cap": cap,
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

        "RAW_ALL": {"mae": mae_raw_all, "rmse": rmse_raw_all, "acc_mae": acc_mae_raw_all, "acc_rmse": acc_rmse_raw_all},

        "do_calib": int(args.do_calib),
        "calib_ratio": float(args.calib_ratio),
        "a_cal": "" if a_cal is None else a_cal,
        "b_cal": "" if b_cal is None else b_cal,

        "RAW_TEST": "" if mae_raw_test is None else {"mae": mae_raw_test, "rmse": rmse_raw_test, "acc_mae": acc_mae_raw_test, "acc_rmse": acc_rmse_raw_test},
        "CAL_TEST": "" if mae_cal_test is None else {"mae": mae_cal_test, "rmse": rmse_cal_test, "acc_mae": acc_mae_cal_test, "acc_rmse": acc_rmse_cal_test},
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 9) 追加记录到 result.txt（含校准对比）
    os.makedirs(args.results_root, exist_ok=True)
    result_path = os.path.join(args.results_root, "chronos2_result.txt")
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(
            f"{setting}\t"
            f"RAW_ALL_RMSE={rmse_raw_all:.6f}\tRAW_ALL_acc_rmse={acc_rmse_raw_all:.6f}\t"
        )
        if int(args.do_calib) == 1:
            f.write(
                f"RAW_TEST_RMSE={rmse_raw_test:.6f}\tRAW_TEST_acc_rmse={acc_rmse_raw_test:.6f}\t"
                f"CAL_TEST_RMSE={rmse_cal_test:.6f}\tCAL_TEST_acc_rmse={acc_rmse_cal_test:.6f}\t"
                f"a={a_cal:.6f}\tb={b_cal:.6f}\t"
            )
        f.write("\n")

    # 10) 可选：追加一行到 CSV（强烈推荐方便排序）
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
        row = {
            "time": meta["time"],
            "setting": setting,
            "dataset_tag": args.dataset_tag,
            "csv": os.path.basename(args.csv),
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "strict_test_only": int(bool(args.strict_test_only)),
            "test_ratio": args.test_ratio,
            "freq": args.freq,
            "fill_method": args.fill_method,
            "quantiles": args.quantiles,
            "cap": cap,

            "RAW_ALL_MAE": mae_raw_all,
            "RAW_ALL_RMSE": rmse_raw_all,
            "RAW_ALL_acc_rmse": acc_rmse_raw_all,

            "do_calib": int(args.do_calib),
            "calib_ratio": args.calib_ratio,
            "a_cal": "" if a_cal is None else a_cal,
            "b_cal": "" if b_cal is None else b_cal,

            "RAW_TEST_RMSE": "" if rmse_raw_test is None else rmse_raw_test,
            "RAW_TEST_acc_rmse": "" if acc_rmse_raw_test is None else acc_rmse_raw_test,
            "CAL_TEST_RMSE": "" if rmse_cal_test is None else rmse_cal_test,
            "CAL_TEST_acc_rmse": "" if acc_rmse_cal_test is None else acc_rmse_cal_test,

            "out_dir": out_dir,
        }
        df_row = pd.DataFrame([row])
        if os.path.exists(args.log_csv):
            df_row.to_csv(args.log_csv, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df_row.to_csv(args.log_csv, mode="w", header=True, index=False, encoding="utf-8")
        print("Appended to log_csv:", args.log_csv)

    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
