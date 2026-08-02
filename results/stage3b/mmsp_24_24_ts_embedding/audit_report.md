# Stage 3B-TS audit

- Full three-modal checkpoint: yes; only its TS encoder branch was called.
- Historical power window: exactly 24 hourly values ending at forecast origin.
- NWP/satellite/future power: not loaded or passed to the extraction interface.
- Shared protocol: all three groups use identical window IDs, sites, origins, targets, future timestamps, Chronos model, quantile, context length and inference settings.
- Shuffle: global Sattolo derangement, seed 2021, zero fixed points, saved permutation reproduced exactly.
- Embeddings: past context only; no embedding columns occur in `future_df`.
- Training: no optimizer, backward, fit, trainable mapping or parameter update.
- Parameter integrity: FusionSF and Chronos state digests were unchanged before/after inference.
- Result: negative; aligned does not stably outperform shuffled and is worse than baseline.
