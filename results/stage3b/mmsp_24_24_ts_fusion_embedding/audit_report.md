# Stage 3B MMSP TS/Fusion embedding preflight audit

## Decision

**Conclusion E — experiment audit failed before window construction.**

The selected checkpoint is the fixed_v1 clean30 full-modal cross-site run trained on sites 10–19 and evaluated on unseen sites 0–9. Satellite indexing uses only the 24 historical frames ending at forecast origin.

The experiment cannot proceed because `nwp.csv` contains only `fcst_date` (valid time). It has no issue/publication timestamp, forecast-reference time, cycle, or lead time. The dataset loader selects the 24 NWP valid times in the target horizon, but neither local metadata nor the official public description proves those values were available at forecast origin. Under the approved protocol this unresolved availability is a blocking future-information risk.

No model was loaded, no window manifest or embedding was generated, and no Chronos prediction, smoke, or full run was started.
