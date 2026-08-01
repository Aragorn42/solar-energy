# run_solarv4_full_verified preflight smoke

Run tag: `preflight_acceptance_20260802`. This smoke is excluded from the final result table.

| Task | Model | Target | Pred len | Samples | MAE accuracy | RMSE accuracy | Finite/aligned | Validation checkpoint |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: |
| nowcasting | patchtst | power | 1 | 5331 | 83.9% | 82.5% | PASS | PASS |
| short_term | nlinear | power_kt | 72 | 5260 | 93.8% | 87.6% | PASS | PASS |
| ultra_short | dlinear | power_kt | 4 | 5328 | 91.2% | 86.2% | PASS | PASS |
