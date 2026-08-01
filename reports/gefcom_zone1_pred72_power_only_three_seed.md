# GEFCom zone1 336→72 Power-only three-seed stability

## Scope

Seeds 2021, 2022, and 2023 use the same Solar-native `custom_solar` Power-only protocol.
Only `random_seed`, `model`, and the seed-bearing `model_id` differ. No future weather,
new horizon, other dataset, hyperparameter search, or Chronos-2 embedding was used.

## Per-seed native results

| Seed | Model | Best epoch | Stopped epoch | Raw MAE | Raw MSE | Raw RMSE | Physical MAE | Physical RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2021 | Transformer | 3 | 13 | 0.250246 | 0.205732 | 0.453577 | 0.063960 | 0.115930 |
| 2021 | FusionSFSolar | 5 | 15 | 0.249972 | 0.201739 | 0.449154 | 0.063890 | 0.114799 |
| 2022 | Transformer | 4 | 14 | 0.257842 | 0.210120 | 0.458389 | 0.065902 | 0.117160 |
| 2022 | FusionSFSolar | 5 | 15 | 0.247573 | 0.205563 | 0.453390 | 0.063277 | 0.115882 |
| 2023 | Transformer | 4 | 14 | 0.258966 | 0.215159 | 0.463852 | 0.066189 | 0.118556 |
| 2023 | FusionSFSolar | 5 | 15 | 0.235760 | 0.200071 | 0.447293 | 0.060258 | 0.114324 |

FusionSFSolar relative to the paired Transformer:

| Seed | Raw MAE change | Raw RMSE change | Physical MAE change | Physical RMSE change |
|---:|---:|---:|---:|---:|
| 2021 | -0.1095% | -0.9752% | -0.1095% | -0.9752% |
| 2022 | -3.9824% | -1.0905% | -3.9824% | -1.0905% |
| 2023 | -8.9610% | -3.5698% | -8.9610% | -3.5698% |

MAE improves in 3/3 seeds and RMSE improves in 3/3 seeds. The mean of the three paired
relative changes is -4.3510% for raw MAE and -1.8785% for raw RMSE.

## Mean ± sample standard deviation

| Model | Raw MAE | Raw RMSE | Physical MAE | Physical RMSE | Best epoch | Training seconds | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 0.255684 ± 0.004743 | 0.458606 ± 0.005141 | 0.065351 ± 0.001212 | 0.117215 ± 0.001314 | 3.667 ± 0.577 | 349.09 ± 21.44 | 17,874,945 |
| FusionSFSolar | 0.244435 ± 0.007608 | 0.449946 ± 0.003125 | 0.062475 ± 0.001945 | 0.115002 ± 0.000799 | 5.000 ± 0.000 | 488.98 ± 15.25 | 25,428,993 |

## Moving-block bootstrap

For each evaluated forecast origin, window MAE and MSE are calculated across all 72 physical-
space predictions. Paired Fusion-minus-Transformer time series use moving blocks of 168 origins,
10,000 resamples, and bootstrap seed 42. Negative differences favor FusionSFSolar.

| Seed | Mean MAE diff | MAE 95% CI | P(Fusion MAE better) | Mean MSE diff | MSE 95% CI | P(Fusion MSE better) |
|---:|---:|---:|---:|---:|---:|---:|
| 2021 | -0.000070 | [-0.001559, 0.001587] | 0.5118 | -0.000261 | [-0.000932, 0.000371] | 0.7602 |
| 2022 | -0.002625 | [-0.004634, -0.000812] | 0.9974 | -0.000298 | [-0.001063, 0.000406] | 0.7822 |
| 2023 | -0.005931 | [-0.007837, -0.004045] | 1.0000 | -0.000986 | [-0.001932, -0.000145] | 0.9902 |
| Three-seed average | -0.002875 | [-0.003934, -0.001811] | 1.0000 | -0.000515 | [-0.000959, -0.000072] | 0.9891 |

The aggregate bootstrap averages independently resampled paired time blocks from each seed; it
does not treat highly overlapping forecast origins as independent observations.

## 72-origin stride sensitivity

Every 72nd origin is retained, producing 78 starts per seed. This sensitivity result does not
replace native leaderboard metrics.

| Seed | Mean paired MAE diff | Mean paired MSE diff |
|---:|---:|---:|
| 2021 | -0.001108 | +0.000066 |
| 2022 | -0.005345 | -0.000835 |
| 2023 | -0.006628 | -0.001059 |
| Three-seed average | -0.004360 | -0.000609 |

## Protocol and fairness

All three same-seed audits are `fair_benchmark=true`. Seeds 2022 and 2023 additionally record
the loader fields directly in each manifest and training summary. Seed 2021 is intentionally
unchanged; its equivalent counts are reconstructed from its original Dataset and output arrays.

| Seed | drop_last | Dataset windows | Evaluated windows | Dropped windows | Fair audit |
|---:|---:|---:|---:|---:|---:|
| 2021 | true | 5,625 | 5,568 | 57 | true |
| 2022 | true | 5,625 | 5,568 | 57 | true |
| 2023 | true | 5,625 | 5,568 | 57 | true |

For seeds 2022/2023, the two resolved configs differ only in `model`. Each audit compares the
data SHA, Dataset class, all split borders, scaler fit rows/values, all 5,625 origins, every
72-step target timestamp range, loader counts, and metric source SHA.

## Decision

FusionSFSolar improves both MAE and RMSE in 3/3 seeds and its three-seed means are lower. The
average paired relative improvements exceed 1%, so the supplied decision rule recommends
expanding later to GEFCom `pred_len=1` and `4`. No such runs were started in this task.
