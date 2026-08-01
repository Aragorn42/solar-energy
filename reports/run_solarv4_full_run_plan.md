# run_solarv4_full_verified full-run plan

This is a plan only. The final full run has not been started.

- Sites: 12 (8 StateGrid + 1 SKIPPD + 3 GEFCom)
- Task configurations: 36
- Seeds per task: 2 (`42`, `123`)
- Total model runs: 72
- Duplicate configurations: 0
- Output rule: `results/verified/run_solarv4_full/<unique_run_tag>/<dataset>_<site>_<task>_seed42-123/`
- Collision rule: an existing run directory raises `FileExistsError`; it is never overwritten.
- Retry rule: use a new run tag and rerun only the failed dataset/site/task; cache remains disabled unless its data/config/code fingerprint matches exactly.

| Dataset | Site | Task | Model | Pred len | Seed |
| :--- | :--- | :--- | :--- | ---: | ---: |
| StateGrid | site1 | nowcasting | PatchTST | 1 | 42 |
| StateGrid | site1 | nowcasting | PatchTST | 1 | 123 |
| StateGrid | site1 | ultra_short | DLinear | 16 | 42 |
| StateGrid | site1 | ultra_short | DLinear | 16 | 123 |
| StateGrid | site1 | short_term | NLinear | 288 | 42 |
| StateGrid | site1 | short_term | NLinear | 288 | 123 |
| StateGrid | site2 | nowcasting | PatchTST | 1 | 42 |
| StateGrid | site2 | nowcasting | PatchTST | 1 | 123 |
| StateGrid | site2 | ultra_short | DLinear | 16 | 42 |
| StateGrid | site2 | ultra_short | DLinear | 16 | 123 |
| StateGrid | site2 | short_term | NLinear | 288 | 42 |
| StateGrid | site2 | short_term | NLinear | 288 | 123 |
| StateGrid | site3 | nowcasting | PatchTST | 1 | 42 |
| StateGrid | site3 | nowcasting | PatchTST | 1 | 123 |
| StateGrid | site3 | ultra_short | DLinear | 16 | 42 |
| StateGrid | site3 | ultra_short | DLinear | 16 | 123 |
| StateGrid | site3 | short_term | NLinear | 288 | 42 |
| StateGrid | site3 | short_term | NLinear | 288 | 123 |
| StateGrid | site4 | nowcasting | PatchTST | 1 | 42 |
| StateGrid | site4 | nowcasting | PatchTST | 1 | 123 |
| StateGrid | site4 | ultra_short | DLinear | 16 | 42 |
| StateGrid | site4 | ultra_short | DLinear | 16 | 123 |
| StateGrid | site4 | short_term | NLinear | 288 | 42 |
| StateGrid | site4 | short_term | NLinear | 288 | 123 |
| StateGrid | site5 | nowcasting | PatchTST | 1 | 42 |
| StateGrid | site5 | nowcasting | PatchTST | 1 | 123 |
| StateGrid | site5 | ultra_short | DLinear | 16 | 42 |
| StateGrid | site5 | ultra_short | DLinear | 16 | 123 |
| StateGrid | site5 | short_term | NLinear | 288 | 42 |
| StateGrid | site5 | short_term | NLinear | 288 | 123 |
| StateGrid | site6 | nowcasting | PatchTST | 1 | 42 |
| StateGrid | site6 | nowcasting | PatchTST | 1 | 123 |
| StateGrid | site6 | ultra_short | DLinear | 16 | 42 |
| StateGrid | site6 | ultra_short | DLinear | 16 | 123 |
| StateGrid | site6 | short_term | NLinear | 288 | 42 |
| StateGrid | site6 | short_term | NLinear | 288 | 123 |
| StateGrid | site7 | nowcasting | PatchTST | 1 | 42 |
| StateGrid | site7 | nowcasting | PatchTST | 1 | 123 |
| StateGrid | site7 | ultra_short | DLinear | 16 | 42 |
| StateGrid | site7 | ultra_short | DLinear | 16 | 123 |
| StateGrid | site7 | short_term | NLinear | 288 | 42 |
| StateGrid | site7 | short_term | NLinear | 288 | 123 |
| StateGrid | site8 | nowcasting | PatchTST | 1 | 42 |
| StateGrid | site8 | nowcasting | PatchTST | 1 | 123 |
| StateGrid | site8 | ultra_short | DLinear | 16 | 42 |
| StateGrid | site8 | ultra_short | DLinear | 16 | 123 |
| StateGrid | site8 | short_term | NLinear | 288 | 42 |
| StateGrid | site8 | short_term | NLinear | 288 | 123 |
| SKIPPD | site1 | nowcasting | PatchTST | 1 | 42 |
| SKIPPD | site1 | nowcasting | PatchTST | 1 | 123 |
| SKIPPD | site1 | ultra_short | DLinear | 16 | 42 |
| SKIPPD | site1 | ultra_short | DLinear | 16 | 123 |
| SKIPPD | site1 | short_term | NLinear | 288 | 42 |
| SKIPPD | site1 | short_term | NLinear | 288 | 123 |
| GEFCom | zone1 | nowcasting | PatchTST | 1 | 42 |
| GEFCom | zone1 | nowcasting | PatchTST | 1 | 123 |
| GEFCom | zone1 | ultra_short | DLinear | 4 | 42 |
| GEFCom | zone1 | ultra_short | DLinear | 4 | 123 |
| GEFCom | zone1 | short_term | NLinear | 72 | 42 |
| GEFCom | zone1 | short_term | NLinear | 72 | 123 |
| GEFCom | zone2 | nowcasting | PatchTST | 1 | 42 |
| GEFCom | zone2 | nowcasting | PatchTST | 1 | 123 |
| GEFCom | zone2 | ultra_short | DLinear | 4 | 42 |
| GEFCom | zone2 | ultra_short | DLinear | 4 | 123 |
| GEFCom | zone2 | short_term | NLinear | 72 | 42 |
| GEFCom | zone2 | short_term | NLinear | 72 | 123 |
| GEFCom | zone3 | nowcasting | PatchTST | 1 | 42 |
| GEFCom | zone3 | nowcasting | PatchTST | 1 | 123 |
| GEFCom | zone3 | ultra_short | DLinear | 4 | 42 |
| GEFCom | zone3 | ultra_short | DLinear | 4 | 123 |
| GEFCom | zone3 | short_term | NLinear | 72 | 42 |
| GEFCom | zone3 | short_term | NLinear | 72 | 123 |
