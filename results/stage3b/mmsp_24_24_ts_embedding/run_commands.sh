#!/usr/bin/env bash
set -euo pipefail

/home/zhaopp/miniconda3/envs/FusionSF/bin/python -m pytest -q tests/test_stage3b_ts_mmsp_chronos2.py

/home/zhaopp/miniconda3/envs/torch/bin/python -u run_stage3b_ts_mmsp_chronos2.py \
  --scope smoke --device cuda:0 --embedding-batch-size 32 \
  --window-batch-size 32 --model-batch-size 32

/home/zhaopp/miniconda3/envs/torch/bin/python -u run_stage3b_ts_mmsp_chronos2.py \
  --scope full --device cuda:0 --embedding-batch-size 256 \
  --window-batch-size 512 --model-batch-size 256
