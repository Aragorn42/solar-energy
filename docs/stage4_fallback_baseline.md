# Stage 4 fallback baseline

This baseline contains only the reusable routing and external-confirmation preflight. It intentionally excludes experiment outputs, checkpoints, plots, and MMSP-specific inference code.

The router maps complete satellite input to `quality_gate`, partial missingness to `cora`, and fully missing input to `static_mask_gate`. Ratios must be finite and in `[0, 1]`; a small endpoint tolerance handles floating-point aggregation.

Before claiming external confirmation, the preflight requires unique candidate window IDs, valid forecast/target ordering, zero overlap with the reference manifest, and either sites disjoint from every train/validation/test site or a strictly future time range. Callers must explicitly supply the complete excluded-site set.

This is a policy baseline, not a portable trained model. Each dataset must provide compatible checkpoints and independently validate the missingness semantics before deployment.
