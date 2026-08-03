#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhaopp/workspace/solar-energy"
PYTHON="/home/zhaopp/miniconda3/envs/torch/bin/python"
GPU="${GPU:-1}"
OUT_ROOT="$ROOT/results/stage4c/mmsp_24_24_missing_gate_v2/full"

usage() {
  echo "Usage: $0 {start|status|monitor} {2022|2023}"
  echo "  GPU=1 $0 start 2022"
  echo "  $0 status 2022"
  echo "  $0 monitor 2022"
}

[[ $# -eq 2 ]] || { usage; exit 2; }
ACTION="$1"
SEED="$2"

case "$SEED" in
  2022) MASK_SEED=5042 ;;
  2023) MASK_SEED=6042 ;;
  *) echo "Unsupported seed: $SEED" >&2; exit 2 ;;
esac

LOG="$ROOT/results/stage4c_seed${SEED}.log"
RESULT="$OUT_ROOT/seed_${SEED}/scenario_metrics.csv"

case "$ACTION" in
  start)
    if pgrep -af "run_stage4c_missing_gate_mmsp.py --scope full" | grep -v "pgrep" >/dev/null; then
      echo "已有 Stage 4C full 进程，拒绝重复启动："
      pgrep -af "run_stage4c_missing_gate_mmsp.py --scope full" || true
      exit 1
    fi
    mkdir -p "$ROOT/results"
    cd "$ROOT"
    echo "Starting seed=$SEED on physical GPU=$GPU (process device cuda:0)"
    nohup env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u \
      run_stage4c_missing_gate_mmsp.py \
      --scope full \
      --seed "$SEED" \
      --mask-seed "$MASK_SEED" \
      --device cuda:0 \
      --epochs 3 \
      --patience 1 \
      > "$LOG" 2>&1 < /dev/null &
    echo "PID=$!"
    echo "LOG=$LOG"
    ;;
  status)
    if [[ -f "$RESULT" ]]; then
      echo "seed $SEED: complete"
      tail -n +1 "$RESULT"
    else
      echo "seed $SEED: incomplete"
      pgrep -af "run_stage4c_missing_gate_mmsp.py --scope full --seed $SEED" || true
      exit 1
    fi
    ;;
  monitor)
    echo "Tailing $LOG (Ctrl-C stops monitoring only)"
    tail -f "$LOG"
    ;;
  *) usage; exit 2 ;;
esac
