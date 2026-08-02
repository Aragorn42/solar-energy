#!/usr/bin/env bash
set -euo pipefail

# This preflight intentionally stops before any model inference when NWP
# issue/publication-time evidence is absent.
/home/zhaopp/miniconda3/envs/torch/bin/python -u run_stage3b_mmsp_ts_fusion_embedding_chronos2.py --preflight-only

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/zhaopp/miniconda3/envs/torch/bin/python - <<'PY'
import sys
sys.path.append('/home/zhaopp/miniconda3/envs/FusionSF/lib/python3.10/site-packages')
import pytest
raise SystemExit(pytest.main(['-q']))
PY

git diff --check
