import os, glob, warnings, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

print("=" * 70)
print("v12.2_all_stable_fixed - 三数据集短临单变量（修正版）")
print("修正：1.三分数据集 2.使用真实时间戳")
print("StateGrid(8站点) + SKIPPD + GEFCom2014(3区域) = 12测试点")
print("=" * 70)

GUOWANG_DIR = r"/home/zhaopp/workspace/solar-energy/dataset/csg_solar"
SKIPPD_DIR = r"/home/zhaopp/workspace/solar-energy/dataset"
GEFCOM_DIR = r"/home/zhaopp/workspace/solar-energy/dataset/GEFCom"
OUTPUT_DIR = r"/home/zhaopp/workspace/solar-energy/Hyper"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

SEEDS = [42, 123]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"使用2种子集成: {SEEDS}")
print("每个站点训练2个模型取平均，确保稳定性\n")

SEQ_LEN = 672
PRED_LEN = 1
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
PATIENCE = 30
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1

print(f"统一配置:")
print(f"  SEQ_LEN: {SEQ_LEN}")
print(f"  EPOCHS: {EPOCHS}")
print(f"  LR: {LR}")
print(f"  PATIENCE: {PATIENCE}")
print(f"  数据划分: train={TRAIN_RATIO}, val={VAL_RATIO}, test={1-TRAIN_RATIO-VAL_RATIO}\n")

CAPS = {
    1: 50, 2: 130, 3: 30, 4: 130, 5: 110, 6: 35, 7: 30, 8: 30,
    'skippd': 30,
    'gef1': 1.0, 'gef2': 1.0, 'gef3': 1.0
}

def load_guowang(site_num):
    files = glob.glob(os.path.join(GUOWANG_DIR, f"*site_{site_num}_*.csv"))
    if not files:
        return None, None
    df = pd.read_csv(files[0], index_col=0, parse_dates=True)
    power_col = None
    for col in df.columns:
        if "power" in col.lower():
            power_col = col
            break
    if power_col is None:
        return None, None
    power = df[power_col].clip(lower=0)
    power = power.resample("15min").mean().ffill().bfill()
    return power.values, power.index

def load_skippd():
    target = os.path.join(SKIPPD_DIR, "skippd.csv")
    print(f"  [SKIPPD] 使用文件: {os.path.basename(target)}")
    df = pd.read_csv(target, index_col=0, parse_dates=True)
    power_col = None
    for col in df.columns:
        if "power" in col.lower() or "pv" in col.lower():
            power_col = col
            break
    if power_col is None:
        power_col = df.columns[0]
    power = df[power_col].clip(lower=0)
    if len(power) > 150000:
        power = power.resample("15min").mean()
    power = power.ffill().bfill()
    return power.values, power.index

def load_gefcom(zone_id):
    task_dirs = sorted(glob.glob(os.path.join(GEFCOM_DIR, "Task*")))
    if not task_dirs:
        return None, None

    def parse_time(d):
        if "TIMESTAMP" in d.columns:
            d["time"] = pd.to_datetime(d["TIMESTAMP"])
        elif "timestamp" in d.columns:
            d["time"] = pd.to_datetime(d["timestamp"])
        else:
            for cols in [["YEAR","MONTH","DAY","HOUR"],["year","month","day","hour"]]:
                if all(c in d.columns for c in cols):
                    d["time"] = pd.to_datetime(
                        d[cols[0]].astype(str) + "-" +
                        d[cols[1]].astype(str).str.zfill(2) + "-" +
                        d[cols[2]].astype(str).str.zfill(2) + " " +
                        d[cols[3]].astype(str).str.zfill(2) + ":00:00"
                    )
                    break
        return d

    all_train = []
    for d in task_dirs:
        n = os.path.basename(d).replace("Task", "").strip()
        f = os.path.join(d, f"train{n}.csv")
        if os.path.exists(f):
            all_train.append(pd.read_csv(f))

    if not all_train:
        return None, None

    tr = parse_time(pd.concat(all_train, ignore_index=True).drop_duplicates())
    if "ZONEID" not in tr.columns:
        return None, None
    tr = tr[tr["ZONEID"] == zone_id].set_index("time").sort_index()
    pc = [c for c in tr.columns if "power" in c.lower() or c == "POWER"]
    if not pc:
        return None, None
    df = tr[[pc[0]]].rename(columns={pc[0]: "power"})
    df["power"] = df["power"].clip(lower=0)
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample("1h").mean().ffill().bfill()
    return df["power"].values, df.index

class PowerDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
    def forward(self, x):
        front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x_pad = torch.cat([front, x, end], dim=1)
        x_pad = x_pad.unsqueeze(1)
        avg = self.avg(x_pad)
        return avg.squeeze(1)

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, kernel_size=49):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.moving_avg = MovingAvg(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)
    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        trend_out = self.linear_trend(trend)
        seasonal_out = self.linear_seasonal(seasonal)
        return trend_out + seasonal_out

def train_one_seed(train_norm, val_norm, seed):
    set_seed(seed)

    train_dataset = PowerDataset(train_norm, SEQ_LEN, PRED_LEN)
    val_dataset = PowerDataset(val_norm, SEQ_LEN, PRED_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DLinear(SEQ_LEN, PRED_LEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience_count = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_count = 0
            best_model = model.state_dict()
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            break

    model.load_state_dict(best_model)
    return model, best_loss

def train_ensemble(train_data, val_data):
    train_mean = train_data.mean()
    train_std = train_data.std()

    train_norm = (train_data - train_mean) / train_std
    val_norm = (val_data - train_mean) / train_std

    print(f"  标准化: mean={train_mean:.2f}, std={train_std:.2f}")

    models = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"  [{i}/{len(SEEDS)}] 训练seed={seed}...", end=" ")
        model, best_loss = train_one_seed(train_norm, val_norm, seed)
        models.append(model)
        print(f"loss={best_loss:.4f}")

    return models, train_mean, train_std

def evaluate_ensemble(models, test_data, test_ts, capacity, train_mean, train_std):
    test_norm = (test_data - train_mean) / train_std
    test_dataset = PowerDataset(test_norm, SEQ_LEN, PRED_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_model_preds = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                pred = model(x)
                preds.append(pred.cpu().numpy())
        all_model_preds.append(np.concatenate(preds))

    ensemble_pred = np.mean(all_model_preds, axis=0)

    trues = []
    for x, y in test_loader:
        trues.append(y.numpy())
    true_norm = np.concatenate(trues)

    pred_power = ensemble_pred[:, 0] * train_std + train_mean
    true_power = true_norm[:, 0] * train_std + train_mean

    pred_power = np.clip(pred_power, 0, None)
    true_power = np.clip(true_power, 0, None)

    valid_ts = test_ts[SEQ_LEN : SEQ_LEN + len(pred_power)]
    months = pd.to_datetime(valid_ts).to_period('M')

    df_results = pd.DataFrame({
        'pred': pred_power,
        'true': true_power,
        'month': months
    })

    monthly = df_results.groupby('month').apply(
        lambda x: pd.Series({
            'mae': np.abs(x['pred'] - x['true']).mean(),
            'rmse': np.sqrt(((x['pred'] - x['true']) ** 2).mean())
        })
    )

    mae_acc = (1 - monthly['mae'] / capacity).mean() * 100
    rmse_acc = (1 - monthly['rmse'] / capacity).mean() * 100

    return mae_acc, rmse_acc

def main():
    results = []
    total_start = time.time()

    print("\n" + "=" * 70)
    print("【1/3】StateGrid 国网数据集（8站点）")
    print("=" * 70)

    for site_num in [1, 2, 3, 4, 5, 6, 7, 8]:
        print(f"\n── Site {site_num} (容量={CAPS[site_num]}MW) ──")

        power, timestamps = load_guowang(site_num)
        if power is None:
            print(f"  跳过 Site {site_num}（数据加载失败）")
            continue

        print(f"  数据长度: {len(power)} 样本")
        print(f"  时间范围: {timestamps[0]} 至 {timestamps[-1]}")

        train_idx = int(len(power) * TRAIN_RATIO)
        val_idx = int(len(power) * (TRAIN_RATIO + VAL_RATIO))

        train_data = power[:train_idx]
        val_data = power[train_idx:val_idx]
        test_data = power[val_idx:]
        test_ts = timestamps[val_idx:]

        print(f"  数据划分: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        models, train_mean, train_std = train_ensemble(train_data, val_data)

        mae_acc, rmse_acc = evaluate_ensemble(
            models, test_data, test_ts,
            CAPS[site_num], train_mean, train_std
        )

        print(f"\n   MAE={mae_acc:.1f}%  RMSE={rmse_acc:.1f}%")

        results.append({
            '数据集': 'StateGrid',
            '站点': f'Site {site_num}',
            '容量': f'{CAPS[site_num]}MW',
            'MAE': round(mae_acc, 1),
            'RMSE': round(rmse_acc, 1),
            '达标': '' if rmse_acc >= 96.0 else ''
        })

    print("\n" + "=" * 70)
    print("【2/3】SKIPPD 数据集")
    print("=" * 70)

    power_sk, timestamps_sk = load_skippd()
    if power_sk is not None:
        print(f"  数据长度: {len(power_sk)} 样本")
        print(f"  时间范围: {timestamps_sk[0]} 至 {timestamps_sk[-1]}")

        train_idx = int(len(power_sk) * TRAIN_RATIO)
        val_idx = int(len(power_sk) * (TRAIN_RATIO + VAL_RATIO))

        train_data = power_sk[:train_idx]
        val_data = power_sk[train_idx:val_idx]
        test_data = power_sk[val_idx:]
        test_ts = timestamps_sk[val_idx:]

        print(f"  数据划分: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        models, train_mean, train_std = train_ensemble(train_data, val_data)

        mae_acc, rmse_acc = evaluate_ensemble(
            models, test_data, test_ts,
            CAPS['skippd'], train_mean, train_std
        )

        print(f"\n   MAE={mae_acc:.1f}%  RMSE={rmse_acc:.1f}%")

        results.append({
            '数据集': 'SKIPPD',
            '站点': 'SKIPPD',
            '容量': f'{CAPS["skippd"]}kW',
            'MAE': round(mae_acc, 1),
            'RMSE': round(rmse_acc, 1),
            '达标': '' if rmse_acc >= 96.0 else ''
        })
    else:
        print("  SKIPPD数据加载失败")

    print("\n" + "=" * 70)
    print("【3/3】GEFCom2014 数据集（3区域）")
    print("=" * 70)

    for zone_id in [1, 2, 3]:
        print(f"\n── Zone {zone_id} (容量={CAPS[f'gef{zone_id}']}) ──")

        power_gef, timestamps_gef = load_gefcom(zone_id)
        if power_gef is None:
            print(f"  跳过 Zone {zone_id}（数据加载失败）")
            continue

        print(f"  数据长度: {len(power_gef)} 样本")
        print(f"  时间范围: {timestamps_gef[0]} 至 {timestamps_gef[-1]}")

        train_idx = int(len(power_gef) * TRAIN_RATIO)
        val_idx = int(len(power_gef) * (TRAIN_RATIO + VAL_RATIO))

        train_data = power_gef[:train_idx]
        val_data = power_gef[train_idx:val_idx]
        test_data = power_gef[val_idx:]
        test_ts = timestamps_gef[val_idx:]

        print(f"  数据划分: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        models, train_mean, train_std = train_ensemble(train_data, val_data)

        mae_acc, rmse_acc = evaluate_ensemble(
            models, test_data, test_ts,
            CAPS[f'gef{zone_id}'], train_mean, train_std
        )

        print(f"\n   MAE={mae_acc:.1f}%  RMSE={rmse_acc:.1f}%")

        results.append({
            '数据集': 'GEFCom2014',
            '站点': f'Zone {zone_id}',
            '容量': f'{CAPS[f"gef{zone_id}"]}',
            'MAE': round(mae_acc, 1),
            'RMSE': round(rmse_acc, 1),
            '达标': '' if rmse_acc >= 96.0 else ''
        })

    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("三数据集短临单变量测试 - 完整结果汇总（修正版）")
    print("=" * 70)

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    print("\n" + "=" * 70)
    print("统计分析")
    print("=" * 70)

    达标数 = (df_results['RMSE'] >= 96.0).sum()
    总数 = len(df_results)
    平均RMSE = df_results['RMSE'].mean()

    print(f"总测试点数: {总数}")
    print(f"达标数量:   {达标数}/{总数} ({达标数/总数*100:.1f}%)")
    print(f"平均RMSE:   {平均RMSE:.1f}%")
    print(f"最高RMSE:   {df_results['RMSE'].max():.1f}%")
    print(f"最低RMSE:   {df_results['RMSE'].min():.1f}%")

    print("\n按数据集统计:")
    for dataset in ['StateGrid', 'SKIPPD', 'GEFCom2014']:
        df_ds = df_results[df_results['数据集'] == dataset]
        if len(df_ds) > 0:
            达标 = (df_ds['RMSE'] >= 96.0).sum()
            平均 = df_ds['RMSE'].mean()
            print(f"  {dataset}: 平均={平均:.1f}%, 达标={达标}/{len(df_ds)}")

    print(f"\n总运行时间: {total_time/3600:.2f}小时")

    csv_path = os.path.join(OUTPUT_DIR, "results_v12.2_all_stable_fixed.csv")
    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {csv_path}")

    print("\n" + "=" * 70)
    print(" 全部完成！（修正版：三分数据+真实时间戳）")
    print("=" * 70)

if __name__ == "__main__":
    main()