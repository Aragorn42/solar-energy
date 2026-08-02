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

## Planned

- Additional seeds for pred_len 1 and 4 only if explicitly approved.
- No follow-up MMSP embedding → Chronos-2 method design is active; Stage 3A audit awaits review.

## Boundaries

- Do not alter or restart the active detached-worktree full run from this development worktree.
- Do not use future weather in the standard Power-only leaderboard.
