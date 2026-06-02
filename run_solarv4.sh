#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

mkdir -p logs

LOG_FILE="logs/run_solar_$(date '+%Y%m%d_%H%M%S').log"


exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Log file: $LOG_FILE"
echo "======================================"

python -u run_solarv4_fixed.py
