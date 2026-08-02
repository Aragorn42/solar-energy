# Stage 3B frozen embedding linear probe

Conclusion: **C**

| Input | Alpha | MAE | RMSE |
|---|---:|---:|---:|
| aligned | 1.0 | 0.251091331 | 0.327209562 |
| shuffled | 1000.0 | 0.242974788 | 0.276101291 |
| raw24 | 0.0 | 0.119809225 | 0.188403070 |
| zero | 1000.0 | 0.242051423 | 0.273781478 |

```json
{
  "comparison": {
    "aligned_vs_shuffled": {
      "mae": {
        "absolute": 0.008116543292999268,
        "relative_pct": 3.340487859445023
      },
      "rmse": {
        "absolute": 0.05110827088356018,
        "relative_pct": 18.51069608012243
      },
      "left_window_mae_win_rate": 0.4025974025974026
    },
    "aligned_vs_raw24": {
      "mae": {
        "absolute": 0.13128210604190826,
        "relative_pct": 109.57595779539993
      },
      "rmse": {
        "absolute": 0.13880649209022522,
        "relative_pct": 73.6752814644281
      },
      "left_window_mae_win_rate": 0.0
    },
    "shuffled_vs_zero": {
      "mae": {
        "absolute": 0.0009233653545379639,
        "relative_pct": 0.3814748720065187
      },
      "rmse": {
        "absolute": 0.002319812774658203,
        "relative_pct": 0.8473227583447249
      },
      "left_window_mae_win_rate": 0.45454545454545453
    }
  },
  "bootstrap": {
    "aligned_minus_shuffled_mae": {
      "mean_difference": 0.008116569031368603,
      "ci_2_5": -0.007075293058553687,
      "ci_97_5": 0.023254621270802113,
      "resamples": 2000,
      "seed": 2021,
      "sampling_unit": "forecast_window",
      "interval_crosses_zero": true
    },
    "aligned_minus_raw24_mae": {
      "mean_difference": 0.13128211223459862,
      "ci_2_5": 0.1256308316543408,
      "ci_97_5": 0.13693132288209706,
      "resamples": 2000,
      "seed": 2021,
      "sampling_unit": "forecast_window",
      "interval_crosses_zero": false
    }
  },
  "audit": {
    "stage3a_shapes_and_rows_valid": true,
    "forecast_origin_unchanged": true,
    "raw24_exactly_origin_minus_23h_through_origin": true,
    "future_information_used": false,
    "chronological_split": true,
    "split_overlap": false,
    "scalers_fit_train_only": true,
    "alpha_selected_validation_only": true,
    "test_used_once_after_selection": true,
    "saved_shuffle_used_without_regeneration": true,
    "shuffle_fixed_points": 0,
    "aligned_shuffled_same_samples_marginal_distribution_shape": true,
    "all_groups_same_y_and_test_origins": true,
    "future_weather_used": false,
    "fusionsf_trained": false,
    "chronos_imported_or_run": false,
    "neural_network_or_trainable_fusion_added": false,
    "only_trainable_model": "sklearn.linear_model.Ridge",
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 2021,
    "conclusion": "C"
  }
}
```
