#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mode="${1:-preflight}"
run_tag="${2:-${mode}_$(date -u +%Y%m%dT%H%M%SZ)}"
python_bin="${SOLAR_PYTHON:-/home/zhaopp/miniconda3/envs/torch/bin/python}"

if [[ "$mode" == "preflight" ]]; then
  export SOLAR_RUN_TAG="$run_tag"
  export SOLAR_RUN_STATEGRID=0
  export SOLAR_RUN_SKIPPD=0
  export SOLAR_RUN_GEFCOM=1
  export SOLAR_GEFCOM_ZONES=1
  export SOLAR_SEEDS=2026
  export SOLAR_EPOCHS=1
  export SOLAR_USE_CACHE=0
elif [[ "$mode" == "full" ]]; then
  if [[ "${SOLAR_CONFIRM_FULL_RUN:-NO}" != "YES" ]]; then
    echo "Refusing full run: set SOLAR_CONFIRM_FULL_RUN=YES after acceptance approval." >&2
    exit 2
  fi
  export SOLAR_RUN_TAG="$run_tag"
  export SOLAR_RUN_STATEGRID=1
  export SOLAR_RUN_SKIPPD=1
  export SOLAR_RUN_GEFCOM=1
  export SOLAR_SITES=1,2,3,4,5,6,7,8
  export SOLAR_GEFCOM_ZONES=1,2,3
  export SOLAR_SEEDS="${SOLAR_SEEDS:-42,123}"
  export SOLAR_EPOCHS="${SOLAR_EPOCHS:-100}"
  export SOLAR_USE_CACHE="${SOLAR_USE_CACHE:-0}"
else
  echo "Usage: $0 [preflight|full] [unique_run_tag]" >&2
  exit 2
fi

exec "$python_bin" -u run_solarv4_full_verified.py
