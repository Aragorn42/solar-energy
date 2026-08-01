# Stage 3A GEFCom zone1 336→72

Scope: `full`; windows: 5289. No model training or future covariates.

| Group | MAE | RMSE |
| :--- | ---: | ---: |
| baseline | 0.054101866 | 0.119046248 |
| aligned | 0.054136384 | 0.120778069 |
| shuffled | 0.054135181 | 0.120780312 |

```json
{
  "metrics": {
    "baseline": {
      "mae": 0.054101865738630295,
      "rmse": 0.11904624849557877,
      "finite": true,
      "shape": [
        5289,
        72,
        1
      ]
    },
    "aligned": {
      "mae": 0.05413638427853584,
      "rmse": 0.12077806890010834,
      "finite": true,
      "shape": [
        5289,
        72,
        1
      ]
    },
    "shuffled": {
      "mae": 0.05413518100976944,
      "rmse": 0.12078031152486801,
      "finite": true,
      "shape": [
        5289,
        72,
        1
      ]
    }
  },
  "aligned_vs_baseline": {
    "mae": {
      "absolute": 3.4518539905548096e-05,
      "relative_pct": 0.06380286416056233
    },
    "rmse": {
      "absolute": 0.0017318204045295715,
      "relative_pct": 1.4547458877663744
    }
  },
  "shuffled_vs_baseline": {
    "mae": {
      "absolute": 3.33152711391449e-05,
      "relative_pct": 0.061578784177413
    },
    "rmse": {
      "absolute": 0.0017340630292892456,
      "relative_pct": 1.4566297142523115
    }
  },
  "aligned_vs_shuffled": {
    "mae": {
      "absolute": 1.2032687664031982e-06,
      "relative_pct": 0.0022227112645768227
    },
    "rmse": {
      "absolute": -2.2426247596740723e-06,
      "relative_pct": -0.0018567800756270844
    }
  },
  "aligned_window_mae_win_rate_vs_baseline": 0.49480052940064284,
  "shuffled_window_mae_win_rate_vs_baseline": 0.49669124598222725,
  "aligned_window_mae_win_rate_vs_shuffled": 0.497447532614861,
  "baseline_aligned_max_abs_prediction_difference": 0.27099305391311646,
  "baseline_shuffled_max_abs_prediction_difference": 0.27472683787345886
}
```
