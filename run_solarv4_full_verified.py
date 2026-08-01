import os, glob, random, warnings, math, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pvlib
from pathlib import Path
from utils.run_solarv4_verified_support import (
    assert_aligned,
    cache_fingerprint,
    cache_is_compatible,
    capacity_clip,
    chronological_split,
    kt_to_power,
    power_to_kt,
    reserve_output_directory,
    select_target,
    target_timestamps,
)

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

print("[verified-full] 混合模型策略：短临PatchTST + 超短期DLinear + 短期NLinear")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GUOWANG_DIR = os.path.join(ROOT_DIR, "dataset", "csg_solar")
SKIPPD_DIR = os.path.join(ROOT_DIR, "dataset")
GEFCOM_DIR = os.path.join(ROOT_DIR, "dataset", "GEFCom")
RUN_TAG = os.environ.get("SOLAR_RUN_TAG")
if not RUN_TAG:
    raise RuntimeError("SOLAR_RUN_TAG is required to prevent output-directory overwrite")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "verified", "run_solarv4_full", RUN_TAG)
ERA5_DIR = os.path.join(ROOT_DIR, "dataset", "ERA5")
MODEL_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FEATURE_DIR = os.path.join(OUTPUT_DIR, "engineered_features")
SAVE_ENGINEERED_FEATURES = False

reserve_output_directory(OUTPUT_DIR)
os.makedirs(MODEL_DIR,  exist_ok=True)
if SAVE_ENGINEERED_FEATURES:
    os.makedirs(FEATURE_DIR, exist_ok=True)

STATION_COORDS = {
    "site1":    (36.1,   103.8),
    "site2":    (36.6,   101.8),
    "site3":    (32.1,   118.8),
    "site4":    (36.7,   117.0),
    "site5":    (36.7,   117.0),
    "site6":    (25.0,   102.7),
    "site7":    (26.6,   106.7),
    "site8":    (34.3,   108.9),
    "skippd":   (37.427,-122.174),
    "gefcom_1": (-34.9,   138.6),
    "gefcom_2": (-35.2,   139.5),
    "gefcom_3": (-34.5,   140.5),
}

# ============================================================
# Time-zone configuration
# ============================================================
# SOURCE_TIMEZONES describes the clock convention used by each raw file
# when its timestamps do not already carry a timezone.
SOURCE_TIMEZONES = {
    "guowang": "Asia/Shanghai",
    "skippd": "America/Los_Angeles",
    "gefcom": "UTC",
    "era5": "UTC",
}

# LOCAL_TIMEZONES is used only to build local calendar/time features.
# All dataframe indexes are stored internally in UTC.
LOCAL_TIMEZONES = {
    "guowang": "Asia/Shanghai",
    "skippd": "America/Los_Angeles",
    "gefcom": "Australia/Adelaide",
}

RUN_GUOWANG = os.environ.get("SOLAR_RUN_STATEGRID", "1") == "1"
RUN_SKIPPD = os.environ.get("SOLAR_RUN_SKIPPD", "1") == "1"
RUN_GEFCOM = os.environ.get("SOLAR_RUN_GEFCOM", "1") == "1"

ENSEMBLE_SEEDS = [int(value) for value in os.environ.get("SOLAR_SEEDS", "42,123").split(",")]
USE_AMP = True
USE_CACHE = os.environ.get("SOLAR_USE_CACHE", "0") == "1"
print(f"[优化] 集成种子数: {len(ENSEMBLE_SEEDS)}")
print(f"[优化] 混合精度训练: {'启用' if USE_AMP else '禁用'}")
print(f"[优化] 模型缓存: {'启用' if USE_CACHE else '禁用'}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SEQ_LEN      = 336
BATCH_SIZE   = 64
EPOCHS       = int(os.environ.get("SOLAR_EPOCHS", "100"))
LR           = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1

D_MODEL  = 256
N_HEADS  = 8
N_LAYERS = 3
D_FF     = 512
DROPOUT  = 0.1

PATCH_LEN = 16
STRIDE = 8
D_MODEL_PATCH = 128
N_HEADS_PATCH = 4
N_LAYERS_PATCH = 2

PATIENCE = {
    "nowcasting_patchtst":  20,
    "ultra_short_dlinear":  20,
    "short_term_nlinear":   20,
}
print(f"[优化] 早停patience: 短临PatchTST={PATIENCE['nowcasting_patchtst']}, "
      f"超短期DLinear={PATIENCE['ultra_short_dlinear']}, 短期NLinear={PATIENCE['short_term_nlinear']}")

TASKS_15 = {
    "nowcasting":  {"pred_len": 1,   "eval_idx": 0,  "label": "短临"},
    "ultra_short": {"pred_len": 16,  "eval_idx": 15, "label": "超短期"},
    "short_term":  {"pred_len": 288, "eval_idx": 95, "label": "短期"},
}
TASKS_1H = {
    "nowcasting":  {"pred_len": 1,   "eval_idx": 0,  "label": "短临"},
    "ultra_short": {"pred_len": 4,  "eval_idx": 3,  "label": "超短期"},
    "short_term":  {"pred_len": 72, "eval_idx": 23, "label": "短期"},
}
CAPS = {
    "site1":50,"site2":130,"site3":30,"site4":130,
    "site5":110,"site6":35,"site7":30,"site8":30,
    "skippd":30,
    "gefcom_1":1.0,"gefcom_2":1.0,"gefcom_3":1.0,
}

def ensure_utc_index(index, source_tz, name="data"):
    """Return a timezone-aware UTC DatetimeIndex.

    If ``index`` is naive, its clock values are first interpreted in
    ``source_tz`` and then converted to UTC. If it is already timezone-aware,
    only a timezone conversion is performed.
    """
    index = pd.DatetimeIndex(pd.to_datetime(index))

    if index.tz is None:
        try:
            index = index.tz_localize(
                source_tz,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        except Exception as exc:
            raise ValueError(
                f"{name} timezone localization failed: source_tz={source_tz}, "
                f"range={index.min()} ~ {index.max()}"
            ) from exc

    return index.tz_convert("UTC")

def save_engineered_features(df, dataset_name, station_name, local_tz):
    """Save the dataframe returned by prepare() without changing training data.

    The CSV index is timestamp_utc and an additional timestamp_local column is
    included for inspection. All engineered feature columns are preserved.
    """
    if not SAVE_ENGINEERED_FEATURES:
        return None

    if df is None or df.empty:
        print(f"  [特征保存] 跳过空数据: {dataset_name}-{station_name}")
        return None

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError(
            f"保存特征失败：{dataset_name}-{station_name} 的索引没有时区"
        )

    safe_dataset = str(dataset_name).replace(" ", "_").replace("/", "_")
    safe_station = str(station_name).replace(" ", "_").replace("/", "_")
    filename = f"{safe_dataset}_{safe_station}_features.csv"
    output_path = os.path.join(FEATURE_DIR, filename)
    temp_path = output_path + ".tmp"

    save_df = df.copy()
    save_df.index = idx.tz_convert("UTC")
    save_df.index.name = "timestamp_utc"

    local_timestamp = idx.tz_convert(local_tz)
    if "timestamp_local" in save_df.columns:
        save_df = save_df.drop(columns=["timestamp_local"])
    save_df.insert(0, "timestamp_local", local_timestamp.astype(str))

    # Write to a temporary file first, then replace atomically.
    save_df.to_csv(
        temp_path,
        index=True,
        index_label="timestamp_utc",
        encoding="utf-8-sig",
    )
    os.replace(temp_path, output_path)

    print(
        f"  [特征保存] {output_path} | "
        f"shape={save_df.shape} | 特征列={len(df.columns)}"
    )
    return output_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def add_solar_features(df, lat, lon, local_tz):
    """Add solar-geometry and clear-sky features using timezone-aware times."""
    df = df.copy()
    times = pd.DatetimeIndex(df.index)

    if times.tz is None:
        raise ValueError(
            "add_solar_features received a timezone-naive index; "
            "loaders must call ensure_utc_index first."
        )

    times_utc = times.tz_convert("UTC")
    loc = pvlib.location.Location(
        latitude=lat,
        longitude=lon,
        tz=local_tz,
    )

    sp = loc.get_solarposition(times_utc)
    df["solar_elevation"] = sp["elevation"].to_numpy()
    df["solar_azimuth"] = sp["azimuth"].to_numpy()
    df["cos_zenith"] = np.cos(
        np.radians(sp["zenith"].to_numpy())
    ).clip(0)
    df["is_day"] = (sp["elevation"].to_numpy() > 0).astype(float)

    cs = loc.get_clearsky(times_utc, model="ineichen")
    df["cs_ghi"] = cs["ghi"].to_numpy()
    df["cs_dni"] = cs["dni"].to_numpy()
    df["cs_dhi"] = cs["dhi"].to_numpy()

    ghi_col = next(
        (c for c in ["total_irr", "ghi", "era5_ghi"] if c in df.columns),
        None,
    )
    if ghi_col is not None:
        df["kt"] = (
            df[ghi_col] / df["cs_ghi"].clip(lower=10)
        ).clip(0, 1.5)
    else:
        df["kt"] = df["cos_zenith"]

    df.loc[df["is_day"] == 0, "kt"] = 0.0
    df["kt_diff"] = df["kt"].diff().fillna(0)
    df["power_kt"] = power_to_kt(df["power"].to_numpy(), df["cs_ghi"].to_numpy())
    return df


def add_lag_rolling(df, freq="15min"):
    df   = df.copy()
    sph  = 4 if freq == "15min" else 1
    lags = [1, 4, sph, 4*sph, 16*sph, 96*sph]
    for col in ["power_kt", "power"]:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    for c in ["kt","cs_ghi"]:
        if c in df.columns:
            for lag in lags:
                df[f"{c}_lag{lag}"] = df[c].shift(lag)
    for w_h, w_name in [(1,"1h"),(4,"4h"),(24,"24h")]:
        w = w_h * sph
        df[f"power_kt_rmean_{w_name}"] = df["power_kt"].rolling(w, min_periods=1).mean()
        df[f"power_kt_rstd_{w_name}"]  = df["power_kt"].rolling(w, min_periods=1).std().fillna(0)
    df["power_kt_ramp"] = df["power_kt"].diff().fillna(0)
    return df

def add_time_features(df, freq="15min", local_tz="UTC"):
    """Add calendar features in the station's local timezone.

    The dataframe index itself remains UTC so that joins and resampling across
    data sources are unambiguous.
    """
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)

    if idx.tz is None:
        raise ValueError("add_time_features received a timezone-naive index")

    local_idx = idx.tz_convert(local_tz)
    df["tf_hour_sin"] = np.sin(2 * np.pi * local_idx.hour / 24)
    df["tf_hour_cos"] = np.cos(2 * np.pi * local_idx.hour / 24)
    df["tf_doy_sin"] = np.sin(2 * np.pi * local_idx.dayofyear / 365.25)
    df["tf_doy_cos"] = np.cos(2 * np.pi * local_idx.dayofyear / 365.25)
    df["tf_month_sin"] = np.sin(2 * np.pi * local_idx.month / 12)
    df["tf_month_cos"] = np.cos(2 * np.pi * local_idx.month / 12)
    df["tf_dow_sin"] = np.sin(2 * np.pi * local_idx.dayofweek / 7)
    df["tf_dow_cos"] = np.cos(2 * np.pi * local_idx.dayofweek / 7)

    if freq == "15min":
        slot = local_idx.hour * 4 + local_idx.minute // 15
        df["tf_slot_sin"] = np.sin(2 * np.pi * slot / 96)
        df["tf_slot_cos"] = np.cos(2 * np.pi * slot / 96)

    return df


def load_era5_csv(station_key):
    path = os.path.join(ERA5_DIR, f"{station_key}_all.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df.index = ensure_utc_index(
        df.index,
        SOURCE_TIMEZONES["era5"],
        name=f"ERA5-{station_key}",
    )
    rename = {
        "temperature_2m":"era5_temp","surface_pressure":"era5_pressure",
        "dew_point_2m":"era5_dewpoint","shortwave_radiation":"era5_ghi",
        "direct_normal_irradiance":"era5_dni","cloud_cover":"era5_cloud",
        "wind_speed_10m":"era5_wind","precipitation":"era5_precip",
        "relative_humidity_2m":"era5_rh",
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    df = df[[c for c in df.columns if c.startswith("era5_")]]
    return df.ffill(limit=4)

def era5_resample(era5_df, freq):
    if freq == "15min":
        idx = pd.date_range(era5_df.index[0], era5_df.index[-1], freq="15min")
        return era5_df.reindex(idx).interpolate(method="time").ffill(limit=4)
    return era5_df.resample("1h").mean().ffill(limit=4)

def load_guowang(site_num):
    files = glob.glob(os.path.join(GUOWANG_DIR, f"*site_{site_num}_*.csv"))
    if not files: return None
    df = pd.read_csv(files[0], index_col=0, parse_dates=True)
    df = df.sort_index()
    df.index = ensure_utc_index(
        df.index,
        SOURCE_TIMEZONES["guowang"],
        name=f"StateGrid-site{site_num}",
    )
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "power" in cl: col_map[c] = "power"
        elif "total solar" in cl: col_map[c] = "total_irr"
        elif "direct normal"in cl: col_map[c] = "dni"
        elif "global horiz" in cl: col_map[c] = "ghi"
        elif "air temp" in cl or "temperature" in cl: col_map[c] = "temperature"
        elif "atmosphere" in cl or "pressure" in cl: col_map[c] = "pressure"
        elif "humidity" in cl: col_map[c] = "humidity"
    df = df.rename(columns=col_map)
    if "power" not in df.columns: return None
    df["power"] = df["power"].clip(lower=0)
    return df.resample("15min").mean().ffill(limit=4)

def load_skippd():
    target = os.path.join(SKIPPD_DIR,"skippd.csv")
    print(f"  [SKIPPD] 使用: {os.path.basename(target)}")
    df = pd.read_csv(target, index_col=0, parse_dates=True)
    df = df.sort_index()
    # The file contains one copy of the 01:00 hour during the autumn DST
    # transition, so pandas cannot infer which fold it represents. Interpret
    # that single hour as standard time; this preserves a unique UTC index and
    # avoids inventing or duplicating observations.
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is None:
        idx = idx.tz_localize(
            SOURCE_TIMEZONES["skippd"],
            ambiguous=False,
            nonexistent="shift_forward",
        )
    df.index = idx.tz_convert("UTC")
    if df.index.duplicated().any():
        raise ValueError("SKIPPD timezone conversion produced duplicate UTC timestamps")
    pc = [c for c in df.columns if "power" in c.lower() or "pv" in c.lower()]
    col = pc[0] if pc else df.columns[0]
    df = df[[col]].rename(columns={col:"power"})
    df["power"] = df["power"].clip(lower=0)
    if len(df) > 150000: df = df.resample("15min").mean()
    return df.ffill(limit=4)

def load_gefcom(zone_id):
    task_dirs = sorted(glob.glob(os.path.join(GEFCOM_DIR,"Task*")))
    if not task_dirs: return None
    def parse_time(d):
        if "TIMESTAMP" in d.columns:
            d["time"] = pd.to_datetime(d["TIMESTAMP"])
        elif "timestamp" in d.columns:
            d["time"] = pd.to_datetime(d["timestamp"])
        else:
            for cols in [["YEAR","MONTH","DAY","HOUR"],["year","month","day","hour"]]:
                if all(c in d.columns for c in cols):
                    d["time"] = pd.to_datetime(
                        d[cols[0]].astype(str)+"-"+d[cols[1]].astype(str).str.zfill(2)+
                        "-"+d[cols[2]].astype(str).str.zfill(2)+" "+
                        d[cols[3]].astype(str).str.zfill(2)+":00:00")
                    break
        return d
    all_train = []
    for d in task_dirs:
        n = os.path.basename(d).replace("Task","").strip()
        f = os.path.join(d, f"train{n}.csv")
        if os.path.exists(f): all_train.append(pd.read_csv(f))
    if not all_train: return None
    tr = parse_time(pd.concat(all_train,ignore_index=True).drop_duplicates())
    if "ZONEID" not in tr.columns: return None
    tr = tr[tr["ZONEID"] == zone_id].set_index("time").sort_index()
    tr.index = ensure_utc_index(
        tr.index,
        SOURCE_TIMEZONES["gefcom"],
        name=f"GEFCom-zone{zone_id}-train",
    )
    pc = [c for c in tr.columns if "power" in c.lower() or c=="POWER"]
    if not pc: return None
    df = tr[[pc[0]]].rename(columns={pc[0]:"power"})
    df["power"] = df["power"].clip(lower=0)
    df = df[~df.index.duplicated(keep="first")]
    all_preds = []
    for d in task_dirs:
        n = os.path.basename(d).replace("Task","").strip()
        f = os.path.join(d, f"predictors{n}.csv")
        if os.path.exists(f): all_preds.append(pd.read_csv(f))
    if all_preds:
        pr = pd.concat(all_preds,ignore_index=True).drop_duplicates()
        if "ZONEID" in pr.columns: pr = pr[pr["ZONEID"]==zone_id]
        pr = parse_time(pr).set_index("time").sort_index()
        pr.index = ensure_utc_index(
            pr.index,
            SOURCE_TIMEZONES["gefcom"],
            name=f"GEFCom-zone{zone_id}-predictors",
        )
        pr = pr[~pr.index.duplicated(keep="first")]
        wcols = [c for c in pr.columns
                 if c not in ["ZONEID","TIMESTAMP","timestamp","time","POWER","power"]]
        if wcols:
            pr = pr[wcols]
            for c in ["VAR169","VAR175","VAR178"]:
                if c in pr.columns:
                    pr[c] = pr[c].diff().clip(lower=0)/3600
            df = df.join(pr, how="left")
    return df.resample("1h").mean().ffill(limit=4)

def prepare(
    raw_df,
    era5_df,
    freq="15min",
    lat=None,
    lon=None,
    local_tz="UTC",
):
    df = raw_df.copy()
    df.index = ensure_utc_index(df.index, "UTC", name="raw_df")

    if era5_df is not None:
        era5 = era5_resample(era5_df, freq)
        era5.index = ensure_utc_index(
            era5.index,
            "UTC",
            name="ERA5-resampled",
        )
        df = df.join(era5, how="left")

    # This step is intentionally left unchanged in the timezone-only revision.
    # Non-causal filling will be repaired separately.
    df = df.ffill(limit=4)

    if lat is not None and lon is not None:
        df = add_solar_features(
            df,
            lat,
            lon,
            local_tz=local_tz,
        )

    df = add_time_features(
        df,
        freq=freq,
        local_tz=local_tz,
    )
    df = add_lag_rolling(df, freq=freq)
    df = df.ffill(limit=4).dropna()
    expected_delta = pd.Timedelta(freq)
    deltas = pd.Series(df.index).diff().dropna()
    if not deltas.eq(expected_delta).all():
        counts = deltas.value_counts().sort_index().to_dict()
        raise ValueError(
            f"Prepared data contains time gaps after missing-value filtering: {counts}"
        )

    cols = ["power_kt", "power", "cs_ghi"] + [
        c
        for c in df.columns
        if c not in ["power_kt", "power", "cs_ghi"]
    ]
    return df[cols]


class PVDataset(Dataset):
    def __init__(
        self,
        arr,
        seq_len,
        pred_len,
        power_kt_idx,
        power_idx,
        cs_ghi_idx,
        mean,
        std,
        predict_power_kt=True
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.power_kt_idx = power_kt_idx
        self.power_idx = power_idx
        self.cs_ghi_idx = cs_ghi_idx
        self.predict_power_kt = predict_power_kt

        # 模型输入和训练标签使用标准化后的数据
        self.data = ((arr - mean) / std).astype(np.float32)

        # 指标计算使用未经标准化的原始数据
        self.cs_ghi_raw = arr[:, cs_ghi_idx].astype(np.float32)
        self.power_raw = arr[:, power_idx].astype(np.float32)

    def __len__(self):
        return max(
            0,
            len(self.data) - self.seq_len - self.pred_len + 1
        )

    def __getitem__(self, i):
        input_start = i
        input_end = i + self.seq_len

        target_start = input_end
        target_end = target_start + self.pred_len

        x = self.data[input_start:input_end]

        if self.predict_power_kt:
            y = self.data[
                target_start:target_end,
                self.power_kt_idx
            ]
        else:
            y = self.data[
                target_start:target_end,
                self.power_idx
            ]

        # 未来晴空辐照度，用于将预测的 power_kt 转回功率
        cs_ghi_future = self.cs_ghi_raw[target_start:target_end]

        # 未来原始真实功率，直接作为指标真值
        power_future_raw = self.power_raw[target_start:target_end]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(cs_ghi_future),
            torch.from_numpy(power_future_raw)
        )

def make_loaders(df, seq_len, pred_len):
    train, validation, test, mean, std, _ = chronological_split(
        df, TRAIN_RATIO, VAL_RATIO
    )
    tr = train.values.astype(np.float32)
    val = validation.values.astype(np.float32)
    te = test.values.astype(np.float32)
    power_kt_idx = list(df.columns).index("power_kt")
    power_idx = list(df.columns).index("power")
    cs_ghi_idx = list(df.columns).index("cs_ghi")
    predict_power_kt = (pred_len > 1)
    tr_ld = DataLoader(
        PVDataset(tr, seq_len, pred_len, power_kt_idx, power_idx, cs_ghi_idx,
                 mean, std, predict_power_kt),
        BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0, pin_memory=True
    )
    te_ld = DataLoader(
        PVDataset(te, seq_len, pred_len, power_kt_idx, power_idx, cs_ghi_idx,
                 mean, std, predict_power_kt),
        BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0, pin_memory=True
    )
    val_ld = DataLoader(
        PVDataset(val, seq_len, pred_len, power_kt_idx, power_idx, cs_ghi_idx,
                 mean, std, predict_power_kt),
        BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0, pin_memory=True
    )
    return tr_ld, val_ld, te_ld, mean, std, power_kt_idx, power_idx, cs_ghi_idx, predict_power_kt

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, kernel_size=25, target_idx=0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomp = SeriesDecomp(kernel_size)
        self.trend_proj = nn.Linear(seq_len, pred_len)
        self.season_proj = nn.Linear(seq_len, pred_len)
        self.n_vars = n_vars
        self.target_idx = target_idx

    def forward(self, x):
        batch_size = x.shape[0]
        seasonal, trend = self.decomp(x)
        seasonal = seasonal.permute(0, 2, 1).reshape(batch_size * self.n_vars, self.seq_len)
        trend = trend.permute(0, 2, 1).reshape(batch_size * self.n_vars, self.seq_len)
        seasonal_pred = self.season_proj(seasonal)
        trend_pred = self.trend_proj(trend)
        pred = seasonal_pred + trend_pred
        pred = pred.reshape(batch_size, self.n_vars, self.pred_len).permute(0, 2, 1)
        return pred[:, :, self.target_idx]

class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, target_idx=0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.target_idx = target_idx
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = x.permute(0, 2, 1).reshape(batch_size * self.n_vars, self.seq_len)
        pred = self.linear(x)
        pred = pred.reshape(batch_size, self.n_vars, self.pred_len).permute(0, 2, 1)
        pred = pred + seq_last
        return pred[:, :, self.target_idx]

class RevIN(nn.Module):
    def __init__(self, n, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.w = nn.Parameter(torch.ones(n))
            self.b = nn.Parameter(torch.zeros(n))

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self._m = x.mean(1, keepdim=True).detach()
            self._s = x.std(1, keepdim=True).detach() + self.eps
            x = (x - self._m) / self._s
            if self.affine:
                x = x * self.w + self.b
        else:
            if self.affine:
                x = (x - self.b) / (self.w + self.eps)
            n = x.shape[-1]
            x = x * self._s[..., :n] + self._m[..., :n]
        return x

class PatchTST(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars,
                 patch_len=PATCH_LEN, stride=STRIDE,
                 d_model=D_MODEL_PATCH, n_heads=N_HEADS_PATCH,
                 n_layers=N_LAYERS_PATCH, dropout=DROPOUT,
                 target_idx=0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.target_idx = target_idx
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.revin = RevIN(n_vars)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * self.num_patches, pred_len)
        self.dropout = nn.Dropout(dropout)

    def create_patches(self, x):
        batch_size, seq_len, n_vars = x.shape
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :]
            patches.append(patch)
        patches = torch.stack(patches, dim=1)
        patches = patches.permute(0, 3, 1, 2)
        return patches

    def forward(self, x):
        batch_size, seq_len, n_vars = x.shape
        x = self.revin(x, mode="norm")
        patches = self.create_patches(x)
        patches = patches.reshape(batch_size * n_vars, self.num_patches, self.patch_len)
        patch_embed = self.patch_embedding(patches)
        patch_embed = self.dropout(patch_embed)
        patch_embed = patch_embed + self.pos_embed
        transformer_out = self.transformer(patch_embed)
        flattened = transformer_out.reshape(batch_size * n_vars, -1)
        pred = self.head(flattened)
        pred = pred.reshape(batch_size, n_vars, self.pred_len)
        pred = pred.permute(0, 2, 1)
        pred = self.revin(pred, mode="denorm")
        return pred[:, :, self.target_idx]

class Block(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.n1(x + self.drop(a))
        return self.n2(x + self.drop(self.ff(x)))

class iTransformer(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars,
                 d=D_MODEL, h=N_HEADS, layers=N_LAYERS, ff=D_FF, drop=DROPOUT,
                 target_idx=0):
        super().__init__()
        self.target_idx = target_idx
        self.revin = RevIN(n_vars)
        self.embed = nn.Sequential(nn.Linear(seq_len, d), nn.Dropout(drop))
        self.blocks = nn.ModuleList([Block(d, h, ff, drop) for _ in range(layers)])
        self.project = nn.Linear(d, pred_len)

    def forward(self, x):
        x = self.revin(x, "norm")
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.project(x)
        x = x.permute(0, 2, 1)
        x = self.revin(x, "denorm")
        return x[:, :, self.target_idx]

def get_best_model_type(pred_len, task_name):

    if pred_len == 1:
        return 'patchtst'
    elif pred_len <= 16:
        return 'dlinear'
    else:
        return 'nlinear'

def get_patience(model_type, task_name):

    if model_type == 'patchtst' and task_name == 'nowcasting':
        return PATIENCE['nowcasting_patchtst']
    elif model_type == 'dlinear' and task_name == 'ultra_short':
        return PATIENCE['ultra_short_dlinear']
    elif model_type == 'nlinear' and task_name == 'short_term':
        return PATIENCE['short_term_nlinear']
    else:
        return 15

def get_model(model_type, seq_len, pred_len, n_vars, target_idx=0):

    if model_type == 'dlinear':
        return DLinear(seq_len, pred_len, n_vars, target_idx=target_idx).to(DEVICE)
    elif model_type == 'nlinear':
        return NLinear(seq_len, pred_len, n_vars, target_idx=target_idx).to(DEVICE)
    elif model_type == 'patchtst':
        return PatchTST(seq_len, pred_len, n_vars, target_idx=target_idx).to(DEVICE)
    elif model_type == 'itransformer':
        return iTransformer(seq_len, pred_len, n_vars, target_idx=target_idx).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_single_seed(df, task_cfg, model_key, seed, model_type='dlinear'):

    pred_len = task_cfg["pred_len"]
    tname = task_cfg.get("tname", "ultra_short")
    patience = get_patience(model_type, tname)
    seq_len = min(SEQ_LEN, len(df) // 6)

    if len(df) < seq_len + pred_len + 200:
        return None

    source_data = task_cfg["source_data_path"]
    fingerprint, fingerprint_inputs = cache_fingerprint(
        source_data,
        {
            "model_type": model_type, "pred_len": pred_len, "seq_len": seq_len,
            "seed": seed, "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "columns": list(df.columns),
        },
        __file__,
    )
    ckpt = os.path.join(MODEL_DIR, f"{model_key}_{model_type}_seed{seed}_{fingerprint[:12]}.pt")
    cache_metadata = ckpt + ".cache.json"
    if USE_CACHE and os.path.exists(ckpt) and cache_is_compatible(cache_metadata, fingerprint):
        print(f"        [{model_type}_seed{seed}] 使用缓存: {ckpt}")
        return ckpt

    set_seed(seed)

    tr_ld, val_ld, te_ld, mean, std, power_kt_idx, power_idx, cs_ghi_idx, predict_power_kt = \
        make_loaders(df, seq_len, pred_len)

    if len(tr_ld) == 0 or len(val_ld) == 0:
        print(f"        [跳过] 数据窗口不足: train_batches={len(tr_ld)}, val_batches={len(val_ld)}")
        return None

    target_idx = power_kt_idx if predict_power_kt else power_idx
    model = get_model(model_type, seq_len, pred_len, df.shape[1], target_idx=target_idx)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    loss_fn = nn.HuberLoss(delta=1.0)

    best, pat, best_epoch = float("inf"), 0, None
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and DEVICE.type == 'cuda' else None

    for ep in range(EPOCHS):
        model.train()
        tl = 0.0
        for x, y, _, _ in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss = loss_fn(model(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

            tl += loss.item()
        tl /= len(tr_ld)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y, _, _ in val_ld:
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        vl += loss_fn(model(x.to(DEVICE)), y.to(DEVICE)).item()
                else:
                    vl += loss_fn(model(x.to(DEVICE)), y.to(DEVICE)).item()
        vl /= len(val_ld)
        sch.step(vl)

        if vl < best:
            best, pat, best_epoch = vl, 0, ep + 1
            torch.save(model.state_dict(), ckpt)
        else:
            pat += 1

        # 打印日志
        if (ep + 1) % 20 == 0:
            cur_lr = opt.param_groups[0]["lr"]
            print(f"        [{model_type}_seed{seed}] ep{ep + 1:3d}  tr={tl:.4f}  vl={vl:.4f}  best={best:.4f}  lr={cur_lr:.2e}")

        if pat >= patience:
            print(f"        [{model_type}_seed{seed}] 早停 @ ep{ep + 1}")
            break

    with open(cache_metadata, "w", encoding="utf-8") as handle:
        json.dump({
            "cache_fingerprint": fingerprint,
            "fingerprint_inputs": fingerprint_inputs,
            "checkpoint_selected_by": "validation_loss",
            "best_validation_loss": best,
            "best_epoch": best_epoch,
        }, handle, indent=2)
    return ckpt

def compute_metrics(
    pred_norm,
    cs_ghi_arr,
    true_power_raw_arr,
    eval_idx,
    target_mean,
    target_std,
    cap,
    test_ts,
    seq_len,
    is_power_kt
):
    """
    pred_norm:
        模型输出，标准化空间，shape = [N, pred_len]

    cs_ghi_arr:
        原始未来晴空辐照度，shape = [N, pred_len]

    true_power_raw_arr:
        原始未来真实功率，shape = [N, pred_len]
    """

    # 1. 将模型输出从标准化空间恢复到目标变量原始尺度
    pred_target = (
        pred_norm[:, eval_idx] * target_std
        + target_mean
    )

    # 2. 直接读取原始真实功率
    true_power = true_power_raw_arr[:, eval_idx]

    if is_power_kt:
        cs_ghi = cs_ghi_arr[:, eval_idx]

        # 必须与 power_kt 构造时的分母完全一致
        denominator = np.maximum(cs_ghi, 10.0)

        pred_power = kt_to_power(pred_target, cs_ghi)
    else:
        # 模型直接预测功率
        pred_power = pred_target

    # 3. 物理约束：功率在 [0, 装机容量] 范围内
    pred_power, true_power = capacity_clip(pred_power, true_power, cap)

    # 4. 对齐目标时间
    max_ts_samples = len(test_ts) - seq_len - eval_idx

    n = min(
        len(pred_power),
        len(true_power),
        max_ts_samples
    )

    if n <= 0:
        raise ValueError(
            "测试集时间戳不足，无法进行指标计算："
            f"len(test_ts)={len(test_ts)}, "
            f"seq_len={seq_len}, "
            f"eval_idx={eval_idx}"
        )

    pred_power = pred_power[:n]
    true_power = true_power[:n]

    target_ts = target_timestamps(test_ts, seq_len, eval_idx, n)
    assert_aligned(pred_power, true_power, target_ts)

    # 5. 计算误差
    ae = np.abs(pred_power - true_power)
    se = (pred_power - true_power) ** 2

    months = pd.DatetimeIndex(target_ts).to_period("M")

    tmp = pd.DataFrame({
        "ae": ae,
        "se": se,
        "month": months
    })

    monthly = tmp.groupby("month")

    monthly_mae = monthly["ae"].mean()

    monthly_rmse = monthly["se"].apply(
        lambda values: float(np.sqrt(values.mean()))
    )

    mae_acc = float(
        (1.0 - monthly_mae / cap).mean() * 100.0
    )

    rmse_acc = float(
        (1.0 - monthly_rmse / cap).mean() * 100.0
    )

    return {
        "mae_acc": round(mae_acc, 1),
        "rmse_acc": round(rmse_acc, 1),
        "sample_count": int(n),
        "predictions_finite": bool(np.isfinite(pred_power).all()),
        "targets_finite": bool(np.isfinite(true_power).all()),
        "timestamps_finite": bool(pd.DatetimeIndex(target_ts).notna().all()),
        "target_timestamp_start": str(target_ts[0]),
        "target_timestamp_end": str(target_ts[-1]),
    }

def ensemble_predict_and_eval(
    df,
    task_cfg,
    cap,
    model_key,
    model_paths,
    model_type="dlinear"
):
    pred_len = task_cfg["pred_len"]
    eval_idx = task_cfg["eval_idx"]

    seq_len = min(
        SEQ_LEN,
        len(df) // 6
    )

    if len(df) < seq_len + pred_len + 200:
        return None

    (
        tr_ld,
        val_ld,
        te_ld,
        mean,
        std,
        power_kt_idx,
        power_idx,
        cs_ghi_idx,
        predict_power_kt
    ) = make_loaders(
        df,
        seq_len,
        pred_len
    )

    if len(te_ld) == 0:
        print(
            f"      [跳过] 测试集窗口不足: "
            f"test_batches={len(te_ld)}"
        )
        return None

    models = []

    for ckpt in model_paths:
        if ckpt is None or not os.path.exists(ckpt):
            continue

        target_idx = (
            power_kt_idx
            if predict_power_kt
            else power_idx
        )

        model = get_model(
            model_type,
            seq_len,
            pred_len,
            df.shape[1],
            target_idx=target_idx
        )

        state_dict = torch.load(
            ckpt,
            map_location=DEVICE
        )

        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    if len(models) == 0:
        return None

    print(
        f"      [Ensemble] 使用 {len(models)} 个"
        f"{model_type}模型进行集成预测"
    )

    all_preds = []
    cs_ghis = []
    true_powers_raw = []

    with torch.no_grad():
        for x, y, cs_ghi, power_raw in te_ld:
            x_dev = x.to(DEVICE)

            batch_preds = []

            for model in models:
                pred = model(x_dev)
                batch_preds.append(
                    pred.cpu().numpy()
                )

            avg_pred = np.mean(
                batch_preds,
                axis=0
            )

            all_preds.append(avg_pred)
            cs_ghis.append(cs_ghi.numpy())
            true_powers_raw.append(power_raw.numpy())

    pred_norm = np.concatenate(
        all_preds,
        axis=0
    )

    cs_ghi_arr = np.concatenate(
        cs_ghis,
        axis=0
    )

    true_power_raw_arr = np.concatenate(
        true_powers_raw,
        axis=0
    )

    val_idx = int(
        len(df) * (TRAIN_RATIO + VAL_RATIO)
    )

    test_ts = df.index[val_idx:]

    target_idx = (
        power_kt_idx
        if predict_power_kt
        else power_idx
    )

    return compute_metrics(
        pred_norm=pred_norm,
        cs_ghi_arr=cs_ghi_arr,
        true_power_raw_arr=true_power_raw_arr,
        eval_idx=eval_idx,
        target_mean=float(mean[target_idx]),
        target_std=float(std[target_idx]),
        cap=cap,
        test_ts=test_ts,
        seq_len=seq_len,
        is_power_kt=predict_power_kt
    )

def train_and_eval_hybrid(df, task_cfg, cap, model_key):

    pred_len = task_cfg["pred_len"]
    tname = task_cfg.get("tname", "ultra_short")

    model_type = get_best_model_type(pred_len, tname)

    target_name = select_target(pred_len)

    print(f"    [混合选择] {model_type.upper()}（pred_len={pred_len}，预测{target_name}）")

    print(f"    [Ensemble] 开始训练 {len(ENSEMBLE_SEEDS)} 个{model_type}模型...")

    model_paths = []
    start_time = time.time()
    for i, seed in enumerate(ENSEMBLE_SEEDS, 1):
        print(f"      [{i}/{len(ENSEMBLE_SEEDS)}] 训练 seed={seed}")
        ckpt = train_single_seed(df, task_cfg, model_key, seed, model_type)
        model_paths.append(ckpt)

    train_time = time.time() - start_time
    print(f"    [Ensemble] 训练完成，耗时: {train_time:.1f}秒")

    print(f"    [Ensemble] 开始集成预测...")
    result = ensemble_predict_and_eval(df, task_cfg, cap, model_key, model_paths, model_type)
    if result is not None:
        result["checkpoint_paths"] = model_paths
        result["checkpoint_selected_by"] = "validation_loss"
        result["model_type"] = model_type
        result["target_name"] = target_name

    return result


def save_task_artifacts(dataset_name, site_name, df, task_cfg, result):
    """Write isolated preflight/full-run records for one task."""
    task_name = task_cfg["tname"]
    run_name = f"{dataset_name}_{site_name}_{task_name}_seed{'-'.join(map(str, ENSEMBLE_SEEDS))}"
    run_dir = reserve_output_directory(os.path.join(OUTPUT_DIR, run_name))
    train, validation, test, mean, std, boundaries = chronological_split(
        df, TRAIN_RATIO, VAL_RATIO
    )
    resolved = {
        "dataset": dataset_name,
        "site": site_name,
        "task": task_name,
        "model_type": result["model_type"],
        "target": result["target_name"],
        "seq_len": min(SEQ_LEN, len(df) // 6),
        "pred_len": task_cfg["pred_len"],
        "eval_idx": task_cfg["eval_idx"],
        "seeds": ENSEMBLE_SEEDS,
        "max_epochs": EPOCHS,
        "use_cache": USE_CACHE,
        "checkpoint_selected_by": result["checkpoint_selected_by"],
        "source_data_path": task_cfg["source_data_path"],
    }
    manifest = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "split_boundaries": {"train": [0, boundaries[1]], "validation": [boundaries[1], boundaries[2]], "test": [boundaries[2], boundaries[3]]},
        "split_timestamp_ranges": {
            "train": [str(train.index[0]), str(train.index[-1])],
            "validation": [str(validation.index[0]), str(validation.index[-1])],
            "test": [str(test.index[0]), str(test.index[-1])],
        },
        "scaler_fit_rows": [0, boundaries[1]],
        "scaler_mean": mean.tolist(),
        "scaler_scale": std.tolist(),
        "sample_count": result["sample_count"],
        "target_timestamp_start": result["target_timestamp_start"],
        "target_timestamp_end": result["target_timestamp_end"],
    }
    metrics = {key: value for key, value in result.items() if key != "checkpoint_paths"}
    metrics["checkpoint_paths"] = [str(Path(path).relative_to(ROOT_DIR)) for path in result["checkpoint_paths"]]
    for filename, payload in (("resolved_config.json", resolved), ("data_manifest.json", manifest), ("metrics.json", metrics)):
        with open(run_dir / filename, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    return str(run_dir)

records = []
total_start = time.time()

if RUN_GUOWANG:
    print("\n" + "═" * 60 + "\n  国网 StateGrid (v12.1混合策略debug)\n" + "═" * 60)
    SITES = {1: "50MW", 2: "130MW", 3: "30MW", 4: "130MW",
             5: "110MW", 6: "35MW", 7: "30MW", 8: "30MW"}
    requested_sites = {
        int(value) for value in os.environ.get("SOLAR_SITES", "1,2,3,4,5,6,7,8").split(",")
        if value
    }
    SITES = {site: capacity for site, capacity in SITES.items() if site in requested_sites}
    E5MAP = {1: "guowang_site1_50MW", 2: "guowang_site2_130MW",
             3: "guowang_site3_30MW", 4: "guowang_site4_130MW",
             5: "guowang_site5_110MW", 6: "guowang_site6_35MW",
             7: "guowang_site7_30MW", 8: "guowang_site8_30MW"}
    for sn, mw in SITES.items():
        print(f"\n── Site {sn} ({mw}) ──")
        raw = load_guowang(sn)
        if raw is None:
            continue
        lat, lon = STATION_COORDS[f"site{sn}"]
        df = prepare(
            raw,
            load_era5_csv(E5MAP[sn]),
            freq="15min",
            lat=lat,
            lon=lon,
            local_tz=LOCAL_TIMEZONES["guowang"],
        )
        save_engineered_features(
            df,
            dataset_name="StateGrid",
            station_name=f"site{sn}",
            local_tz=LOCAL_TIMEZONES["guowang"],
        )
        cap = CAPS[f"site{sn}"]
        print(f"  shape={df.shape}  Cap={cap}MW  特征数={df.shape[1]}")
        row = {"模型": "Hybrid_v12.1", "数据集": "StateGrid", "电站": sn}
        for tname, tcfg in TASKS_15.items():
            lbl = tcfg["label"]
            source_files = glob.glob(os.path.join(GUOWANG_DIR, f"*site_{sn}_*.csv"))
            tcfg_ = dict(tcfg, tname=tname, source_data_path=source_files[0])
            print(f"\n  [{lbl}] pred_len={tcfg['pred_len']}  eval_idx={tcfg['eval_idx']}")
            res = train_and_eval_hybrid(df, tcfg_, cap, f"sg{sn}_{tname}")
            if res:
                print(f"    → [Result] MAE={res['mae_acc']}%  RMSE={res['rmse_acc']}%")
                row[f"{lbl}_MAE"] = res["mae_acc"]
                row[f"{lbl}_RMSE"] = res["rmse_acc"]
                save_task_artifacts("StateGrid", f"site{sn}", df, tcfg_, res)
        records.append(row)

if RUN_SKIPPD:
    print("\n" + "═" * 60 + "\n  SKIPPD\n" + "═" * 60)
    raw_sk = load_skippd()
    if raw_sk is not None:
        lat, lon = STATION_COORDS["skippd"]
        df_sk = prepare(
            raw_sk,
            load_era5_csv("skippd_stanford"),
            freq="15min",
            lat=lat,
            lon=lon,
            local_tz=LOCAL_TIMEZONES["skippd"],
        )
        save_engineered_features(
            df_sk,
            dataset_name="SKIPPD",
            station_name="site1",
            local_tz=LOCAL_TIMEZONES["skippd"],
        )
        cap_sk = CAPS["skippd"]
        print(f"  shape={df_sk.shape}  Cap={cap_sk}kW  特征数={df_sk.shape[1]}")
        row = {"模型": "Hybrid_v12.1", "数据集": "SKIPPD", "电站": 1}
        for tname, tcfg in TASKS_15.items():
            lbl = tcfg["label"]
            tcfg_ = dict(tcfg, tname=tname, source_data_path=os.path.join(SKIPPD_DIR, "skippd.csv"))
            print(f"\n  [{lbl}] pred_len={tcfg['pred_len']}  eval_idx={tcfg['eval_idx']}")
            res = train_and_eval_hybrid(df_sk, tcfg_, cap_sk, f"skippd_{tname}")
            if res:
                print(f"    → [Result] MAE={res['mae_acc']}%  RMSE={res['rmse_acc']}%")
                row[f"{lbl}_MAE"] = res["mae_acc"]
                row[f"{lbl}_RMSE"] = res["rmse_acc"]
                save_task_artifacts("SKIPPD", "site1", df_sk, tcfg_, res)
        records.append(row)

if RUN_GEFCOM:
    print("\n" + "═" * 60 + "\n  GEFCom2014\n" + "═" * 60)
    requested_zones = [
        int(value) for value in os.environ.get("SOLAR_GEFCOM_ZONES", "1,2,3").split(",")
        if value
    ]
    for zid in requested_zones:
        print(f"\n── Zone {zid} ──")
        df_gef = load_gefcom(zid)
        if df_gef is None:
            continue
        lat, lon = STATION_COORDS[f"gefcom_{zid}"]
        df_gef = prepare(
            df_gef,
            None,
            freq="1h",
            lat=lat,
            lon=lon,
            local_tz=LOCAL_TIMEZONES["gefcom"],
        )
        save_engineered_features(
            df_gef,
            dataset_name="GEFCom2014",
            station_name=f"zone{zid}",
            local_tz=LOCAL_TIMEZONES["gefcom"],
        )
        cap_g = CAPS[f"gefcom_{zid}"]
        print(f"  shape={df_gef.shape}  特征数={df_gef.shape[1]}")
        row = {"模型": "Hybrid_v12.1", "数据集": "GEFCom2014", "电站": zid,
               "短临_MAE": "N/A", "短临_RMSE": "N/A"}
        for tname, tcfg in TASKS_1H.items():
            lbl = tcfg["label"]
            gefcom_sources = sorted(glob.glob(os.path.join(GEFCOM_DIR, "Task*", "train*.csv")))
            tcfg_ = dict(tcfg, tname=tname, source_data_path=gefcom_sources[0])
            print(f"\n  [{lbl}] pred_len={tcfg['pred_len']}  eval_idx={tcfg['eval_idx']}")
            res = train_and_eval_hybrid(df_gef, tcfg_, cap_g, f"gef{zid}_{tname}")
            if res:
                print(f"    → [Result] MAE={res['mae_acc']}%  RMSE={res['rmse_acc']}%")
                row[f"{lbl}_MAE"] = res["mae_acc"]
                row[f"{lbl}_RMSE"] = res["rmse_acc"]
                save_task_artifacts("GEFCom", f"zone{zid}", df_gef, tcfg_, res)
        records.append(row)

total_time = time.time() - total_start
print(f"\n总运行时间: {total_time/3600:.2f}小时")

DISPLAY_COLS = ["短临_MAE", "短临_RMSE", "超短期_MAE", "超短期_RMSE", "短期_MAE", "短期_RMSE"]
COL_LABELS = ["模型", "数据集", "电站",
              "短临功率\n1-MAE/Cap(%)", "短临功率\n1-RMSE/Cap(%)",
              "超短期功率\n1-MAE/Cap(%)", "超短期功率\n1-RMSE/Cap(%)",
              "短期功率\n1-MAE/Cap(%)", "短期功率\n1-RMSE/Cap(%)"]
COL_THRESH = {"短临_MAE": 97.0, "短临_RMSE": 96.0, "超短期_RMSE": 95.0, "短期_RMSE": 95.0}

df_res = pd.DataFrame(records)
for c in ["模型", "数据集", "电站"] + DISPLAY_COLS:
    if c not in df_res.columns:
        df_res[c] = "-"
df_res = df_res[["模型", "数据集", "电站"] + DISPLAY_COLS]

csv_path = os.path.join(OUTPUT_DIR, "results_hybrid_full_verified.csv")
df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n结果CSV: {csv_path}")
print(df_res.to_string(index=False))

GREEN = "#C6EFCE"
RED = "#FFC7CE"
GRAY = "#D9D9D9"
HEADER = "#2E75B6"
cell_text, cell_colors = [], []
for _, row in df_res.iterrows():
    txt, clr = [], []
    for i, col in enumerate(["模型", "数据集", "电站"] + DISPLAY_COLS):
        val = row.get(col, "-")
        txt.append(str(val))
        if i < 3:
            clr.append(GRAY)
        elif val in ("-", "N/A"):
            clr.append("white")
        elif isinstance(val, (int, float)):
            clr.append(GREEN if val >= COL_THRESH.get(col, 90.0) else RED)
        else:
            clr.append("white")
    cell_text.append(txt)
    cell_colors.append(clr)

fig, ax = plt.subplots(figsize=(24, max(5, len(df_res) * 0.6 + 3.5)))
ax.axis("off")
tbl = ax.table(cellText=cell_text, colLabels=COL_LABELS,
               cellColours=cell_colors, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.0)
for j in range(len(COL_LABELS)):
    c = tbl[0, j]
    c.set_facecolor(HEADER)
    c.set_text_props(color="white", fontweight="bold")
plt.legend(handles=[mpatches.Patch(color=GREEN, label="达标 "),
                    mpatches.Patch(color=RED, label="未达标 ")],
           loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=10)
plt.title(
    "Verified full run_solarv4 光伏功率预测结果\n"
    "短临：PatchTST  |  超短期：DLinear  |  短期：NLinear\n"
    "阈值: 短临 MAE>97%, RMSE>96%  |  超短期 RMSE>95%  |  短期 RMSE>95%",
    fontsize=11, fontweight="bold", pad=14
)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "results_table_full_verified.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"图表: {fig_path}\n 全部完成！")
