# Solar Energy Project Status

Last updated: 2026-08-02

## Active

- `solarv4_full_712bfcc_20260802` — **running**
  - Source commit: `712bfcc3be0a3f03be8b7bdfc0b1d3331cc8817a`
  - Isolated worktree: `/home/zhaopp/workspace/solar-energy-full-run-712bfcc`
  - tmux: `solarv4-full-712bfcc`
  - Log: `logs/full_712bfcc_20260802.log`
  - Plan: 12 sites × 3 tasks × 2 seeds = 72 model runs

## Completed / reviewed

- `stage4b_cora_mmsp_chronos2`: **completed locally; No-Go, Stage 4C not entered**
  - Seed 2021 CoRA MAE 0.050525 versus Stage 4A 0.050587; the 0.12% point improvement is not stable
  - Paired window bootstrap CI crosses zero, only 6/10 sites improve, and negative-transfer rate remains 0%; three-seed formal replication is missing
  - Output: `results/stage4b/mmsp_24_24_cora_adapter/`

- `stage4a_unica_mmsp_chronos2`: **completed locally; Go to Stage 4B**
  - Frozen full-modal FusionSF and Chronos-2; adapter trained on sites 10–19, selected on sites 20–21, tested on unseen sites 0–9
  - Seed 2021 full: aligned MAE 0.050587 versus baseline 0.066126 and shuffled 0.085302; aligned improves all 10 test sites
  - Formal three-seed replication remains pending; output: `results/stage4a/mmsp_24_24_unica_tokens/`

- `stage3b_fusion_mmsp_chronos2`: **completed locally; negative result**
  - Frozen full-modal checkpoint and Chronos-2; 25,450 shared windows on unseen MMSP sites 0–9
  - Fusion aligned MAE 0.071684 versus baseline 0.066126, TS aligned 0.070519, shuffled Fusion 0.071684
  - NWP/satellite perturbation changes the formal fusion embedding, but aligned and shuffled Chronos predictions are identical
  - Output: `results/stage3b/mmsp_24_24_fusion_embedding/`

- `stage3b_mmsp_ts_fusion_embedding`: **blocked at preflight; awaiting review**
  - Selected full-modal cross-site checkpoint: train sites 10–19, unseen test sites 0–9
  - NWP file has valid time only; issue/publication time and forecast cycle are absent
  - Conclusion E; no FusionSF/Chronos loading, embedding extraction, smoke, or full run
  - Audit: `results/stage3b/mmsp_24_24_ts_fusion_embedding/audit_report.md`
- `stage3b_embedding_linear_probe`: **completed locally; awaiting review**
  - GEFCom zone1, 336→72, all 5,289 saved Stage 3A windows
  - Frozen aligned/shuffled embeddings, raw24, and zero; multi-output Ridge only
  - Result: conclusion C; aligned did not beat shuffled or raw24
  - Output: `results/stage3b/gefcom_zone1_336_72_linear_probe/`
- `stage3a_injection_semantics_audit`: **completed locally; awaiting review**
  - GEFCom zone1, 336→72, first 64 saved Stage 3A windows
  - Chronos-2 baseline/aligned/shuffled/zero; inference only
  - Result: conclusion C; constant nonzero covariates were not numerically normalized to zero
  - Output: `results/stage3a/gefcom_zone1_336_72/injection_semantics_audit/`
- `gefcom_zone1_pred72_three_seed`: Transformer versus FusionSFSolar Power-only comparison passed review.
- `gefcom_zone1_short_horizons_seed2021`: pred_len 1 and 4 paired runs completed; short-horizon direction was not stable.
- `solarv4_full_acceptance`: static audit and GEFCom zone1 three-task preflight passed at commit `712bfcc`.
- `stage3b_ts_mmsp_chronos2`: completed frozen inference on 25,450 windows from unseen MMSP sites 0–9. Aligned TS embeddings did not outperform shuffled embeddings and underperformed the Chronos-2 baseline; retained as a negative result. Output: `results/stage3b/mmsp_24_24_ts_embedding/`.

## Planned

- Additional seeds for pred_len 1 and 4 only if explicitly approved.
- No Fusion method change is planned; retain the negative result unless a separately approved injection redesign is proposed.

## Boundaries

- Do not alter or restart the active detached-worktree full run from this development worktree.
- Do not use future weather in the standard Power-only leaderboard.
