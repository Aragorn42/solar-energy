# Stage 3B frozen embedding linear probe

Conclusion: **C**

| Input | Alpha | MAE | RMSE |
|---|---:|---:|---:|
| aligned | 1.0 | 0.296671391 | 0.376999050 |
| shuffled | 100.0 | 0.214074224 | 0.240951672 |
| raw24 | 1.0 | 0.107592344 | 0.144616783 |
| zero | 1000.0 | 0.213768721 | 0.240096569 |

```json
{
  "comparison": {
    "aligned_vs_shuffled": {
      "mae": {
        "absolute": 0.08259716629981995,
        "relative_pct": 38.583424321873764
      },
      "rmse": {
        "absolute": 0.1360473781824112,
        "relative_pct": 56.462516712283055
      },
      "left_window_mae_win_rate": 0.21045998739760555
    },
    "aligned_vs_raw24": {
      "mae": {
        "absolute": 0.18907904624938965,
        "relative_pct": 175.73652429228298
      },
      "rmse": {
        "absolute": 0.23238226771354675,
        "relative_pct": 160.68831253938663
      },
      "left_window_mae_win_rate": 0.027095148078134845
    },
    "shuffled_vs_zero": {
      "mae": {
        "absolute": 0.00030550360679626465,
        "relative_pct": 0.14291314739613947
      },
      "rmse": {
        "absolute": 0.0008551031351089478,
        "relative_pct": 0.3561496686321668
      },
      "left_window_mae_win_rate": 0.4763705103969754
    }
  },
  "bootstrap": {
    "aligned_minus_shuffled_mae": {
      "mean_difference": 0.08259716347827052,
      "ci_2_5": 0.07782312197412382,
      "ci_97_5": 0.08729854521060523,
      "resamples": 2000,
      "seed": 2021,
      "sampling_unit": "forecast_window",
      "interval_crosses_zero": false
    },
    "aligned_minus_raw24_mae": {
      "mean_difference": 0.18907905095588431,
      "ci_2_5": 0.18350046793592797,
      "ci_97_5": 0.19447525029030183,
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
