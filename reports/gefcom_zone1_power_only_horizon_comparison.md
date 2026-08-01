# GEFCom zone1 Power-only horizon comparison

All short-horizon runs use seed 2021. The pred_len=72 three-seed rows are means over seeds 2021/2022/2023; the seed-2021 rows are retained separately.

| Horizon | Scope | Model | Params | Best epoch | Raw MAE | Raw RMSE | Physical MAE | Physical RMSE |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | seed2021 | Transformer | 17,874,945 | 14.000 | 0.114158221 | 0.239676267 | 0.029177755 | 0.061258975 |
| 1 | seed2021 | FusionSFSolar | 25,392,641 | 8.000 | 0.141328037 | 0.244819790 | 0.036122104 | 0.062573611 |
| 4 | seed2021 | Transformer | 17,874,945 | 6.000 | 0.183714300 | 0.360636204 | 0.046955630 | 0.092175189 |
| 4 | seed2021 | FusionSFSolar | 25,394,177 | 13.000 | 0.196947753 | 0.355237395 | 0.050337971 | 0.090795298 |
| 72 | seed2021 | Transformer | 17,874,945 | 3.000 | 0.250245869 | 0.453577161 | 0.063960464 | 0.115930012 |
| 72 | seed2021 | FusionSFSolar | 25,428,993 | 5.000 | 0.249971867 | 0.449153900 | 0.063890435 | 0.114799471 |
| 72 | three_seed_mean | Transformer | 17,874,945 | 3.667 | 0.255684396 | 0.458605925 | 0.065350502 | 0.117215319 |
| 72 | three_seed_mean | FusionSFSolar | 25,428,993 | 5.000 | 0.244434968 | 0.449945798 | 0.062475255 | 0.115001872 |

## FusionSFSolar relative to Transformer

- pred_len_1_seed2021: Raw MAE +0.027169816 (+23.800%); Raw RMSE +0.005143523 (+2.146%).
- pred_len_4_seed2021: Raw MAE +0.013233453 (+7.203%); Raw RMSE -0.005398810 (-1.497%).
- pred_len_72_seed2021: Raw MAE -0.000274003 (-0.109%); Raw RMSE -0.004423261 (-0.975%).
- pred_len_72_three_seed_mean: Raw MAE -0.011249428 (-4.400%); Raw RMSE -0.008660128 (-1.888%).

## Fairness and decision

Both pred_len=1 and pred_len=4 audits passed every manifest check (`fair_benchmark=true`). Checks include file SHA256, Dataset class, splits, scaler fit range and parameters, complete origins and targets, test/evaluated/drop counts, metric source SHA, seed, and batch size.

FusionSFSolar is worse on both metrics at pred_len=1. At pred_len=4 its MAE is worse while RMSE is better. The short-horizon direction is therefore not stable; seeds 2022 and 2023 are recommended for both short horizons before drawing a conclusion.

No future weather was used. MMSP embedding and Chronos-2 fusion were not started.
