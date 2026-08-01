# FusionSFSolar fair-benchmark audit

## Scope and authoritative path

The requested baseline is `run_longExp_solarv2.py`. In this checkout it dispatches to
`exp.exp_main_solarv2.Exp_Main`, which uses `data_provider.data_factory_solarv2.data_provider`,
`Dataset_Custom_Solar`, `utils.metrics.metric`, validation-selected `checkpoint.pth`, and the
native `test()` writer. No FusionSF split, scaler, sample manifest, Chronos origin, weather,
or independent training loop is used.

The model is registered as `FusionSFSolar`, with `experiment_track=standard_power_only` and
`architecture_version=fusionsf_solar_v1`.

## Existing solarv2 Transformer scripts (as stored, not inferred)

| Dataset | data_path | features / target | freq | seq / label / pred | epochs / patience | batch / LR | loss / inverse / seed |
|---|---|---|---|---|---|---|---|
| GEFCom Task15 | `dataset/GEFCom/task15.csv` | `M` / `zone3` | default `h` | 336 / 48 / 1,4,72 | 100 / 10 | 64 / 0.0005 | MSE / false / 2021 |
| SKIPPD | `dataset/skippd.csv` | `S` / `OT` | default `h` | 336 / 48 / 1,16,288 | 100 / 10 | 64 / 0.0005 | MSE / false / 2021 |
| CSG site1 | `dataset/csg_solar/Solar_station_site_1_Nominal_capacity-50MW.csv` | `S` / `Power (MW)` | default `h` | 336 / 48 / 1,16,288 | 100 / 10 | 64 / 0.0005 | MSE / false / 2021 |

Sources: `script/huawei_solarv2/Transformer/task15.sh`, `skippd.sh`, and
`csg_solar/csgs1.sh`. Notably, the stored 15-minute dataset scripts do not pass `--freq`, so
the runner default is `h`. This audit records that behavior without silently correcting it.

## GEFCom zone1 Power-only smoke

The stored GEFCom baseline is a three-zone multivariate (`features=M`) task targeting all
three output channels; it is not a zone1-only Power task. The requested smoke therefore uses
`features=S,target=zone1,enc_in=dec_in=c_out=1`. All other data and evaluation behavior stays
native. This smoke can be compared fairly only with another model run with these exact
Power-only arguments; it is not interchangeable with the stored multivariate Task15 result.

Native protocol observations:

- chronological `Dataset_Custom_Solar` boundaries: train `[0,11960)`, validation
  `[11624,13288)`, test `[12952,18984)` (history overlap is intentional);
- `StandardScaler.fit` uses rows `[0,11960)` only, and validation/test instantiate the same
  fitted statistics from that same range;
- native test dataset has 5,625 step-1 windows for 336→72; the smoke debug cap evaluates 64;
- raw metrics come directly from `utils.metrics.metric`; no clipping is performed;
- checkpoint selection uses validation loss; the Solar implementation prints test loss per
  epoch but does not use it in `EarlyStopping`;
- `pred.npy`, `true.npy`, and `metrics.npy` are standardized-space native outputs;
  `y_pred.npy` and `y_true.npy` are the existing post-test inverse-scaled outputs.

The machine-readable audit is
`reports/fusionsf_solar_gefcom_zone1_pred72_fairness.json`. All eight protocol checks pass for
the exact zone1 Power-only configuration. Because the run is batch-limited, its result status
is `smoke_only`, not a final leaderboard entry.

## Consistency matrix

| Item | FusionSFSolar smoke vs exact Power-only comparator |
|---|---|
| Dataset class | same: `Dataset_Custom_Solar` |
| train/val/test boundaries | same |
| scaler fitting range | same pure-train rows |
| test window construction | same step=1; smoke evaluates first 2 batches only |
| target timestamps | same for evaluated windows |
| seq_len / pred_len | same: 336 / 72 |
| inverse behavior | same: raw standardized metric, extra inverse-scaled arrays |
| metric function | same: `utils.metrics.metric` |
| future target input | not read by `FusionSFSolar` |
| future weather/NWP/ERA5 | absent |

## Smoke result

Seed 2021, one epoch, two train/two validation/two test batches, batch size 32, Adam through
the native experiment, learning rate 0.001. Shapes are `pred=true=(64,72,1)`; all saved arrays
are finite. Raw standardized metrics are MAE 1.082254, MSE 1.684828, RMSE 1.298009, MAPE
1.218609, and MSPE 2.030260. These values validate execution only and support no performance
conclusion.

No large-scale training, Power+Weather track, future weather, Chronos-2 embedding fusion, or
final leaderboard registration was performed.
