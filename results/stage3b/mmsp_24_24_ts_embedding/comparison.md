# Stage 3B-TS — MMSP unseen-site frozen inference

Configuration: sites 0–9, 24 hourly context points, 24-hour forecast, 25,450 test windows, Chronos-2 snapshot `29ec3766`, full-modal FusionSF checkpoint TS branch, seed 2021 derangement. NMAE/NRMSE use the MMSP normalized capacity of 1.0.

| Group | MAE | RMSE | NMAE | NRMSE |
| :-- | --: | --: | --: | --: |
| Chronos-2 baseline | 0.066125765 | 0.137007728 | 0.066125765 | 0.137007728 |
| TS aligned | 0.070518509 | 0.146575779 | 0.070518509 | 0.146575779 |
| TS shuffled | 0.070518993 | 0.146662056 | 0.070518993 | 0.146662056 |

Aligned minus baseline: MAE `+0.004392743` (`+6.6430%`), RMSE `+0.009568051` (`+6.9836%`). Aligned wins 34.986% of windows. Paired bootstrap MAE-difference CI: `[+0.004152706, +0.004647383]`.

Aligned minus shuffled: MAE `-0.000000484` (`-0.000687%`), RMSE `-0.000086278` (`-0.05883%`). Aligned wins 49.650% of windows. Paired bootstrap MAE-difference CI: `[-0.000017909, +0.000017899]`.

Conclusion: correctly aligned TS embeddings do not have a stable content advantage over the fixed shuffled control and both embedding groups underperform the baseline. This is retained as a negative result without method changes.

The complete Fusion embedding experiment remains paused because future-NWP availability at forecast origin is not proven.
