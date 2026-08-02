#!/usr/bin/env bash
set -euo pipefail

/home/zhaopp/miniconda3/envs/torch/bin/python -u run_stage3b_embedding_linear_probe.py --run_scope smoke --smoke_windows 512
/home/zhaopp/miniconda3/envs/torch/bin/python -u run_stage3b_embedding_linear_probe.py --run_scope full

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/zhaopp/miniconda3/envs/torch/bin/python - <<'PY'
import sys
sys.path.append('/home/zhaopp/miniconda3/envs/FusionSF/lib/python3.10/site-packages')
import pytest
raise SystemExit(pytest.main(['-q']))
PY

git diff --check
