# run_solarv4_full_verified static acceptance audit

Overall: `PASS`

| Check | Result | Evidence |
| :--- | :---: | :--- |
| `no_backward_fill` | PASS | source scan: no .bfill( token |
| `no_bidirectional_interpolation` | PASS | source scan: no bidirectional interpolation argument |
| `test_not_used_for_training_or_checkpoint` | PASS | train_single_seed checkpoint condition is validation loss; test loader has no training loop |
| `train_only_scaler` | PASS | synthetic train mean=1.0, scale=1.0, boundaries=(0, 6, 7, 10) |
| `validation_denominator` | PASS | source scan: vl /= len(val_ld) |
| `target_selection` | PASS | runtime target selection for 1/4/16/72/288 |
| `eval_timestamp_alignment` | PASS | runtime timestamps start at seq_len+eval_idx=13: ['2020-01-01 13:00:00', '2020-01-01 14:00:00', '2020-01-01 15:00:00', '2020-01-01 16:00:00'] |
| `power_kt_round_trip` | PASS | maximum absolute round-trip error=0.0 |
| `capacity_clipping` | PASS | pred=[0.0, 5.0, 10.0], true=[0.0, 6.0, 10.0] |
| `coordinates_and_time_semantics` | PASS | configured coordinate keys=['gefcom_1', 'gefcom_2', 'gefcom_3', 'site1', 'site2', 'site3', 'site4', 'site5', 'site6', 'site7', 'site8', 'skippd']; source timezone declarations scanned |
| `old_cache_rejected` | PASS | legacy fingerprint metadata rejected at runtime |
| `cache_fingerprint_changes` | PASS | seed2026=f1555f902771ae72, seed2027=bfaf9f37da2380f3; fingerprint inputs include data/config/code SHA |
| `output_directory_collision` | PASS | second reservation of identical output directory raised FileExistsError |
| `aligned_lengths` | PASS | runtime prediction/target/timestamp length assertion |
| `default_cache_disabled` | PASS | source scan: default SOLAR_USE_CACHE=0 |
