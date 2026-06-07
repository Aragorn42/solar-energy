#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

# ===== Data =====
CSV=./dataset/GEFCom/gefcom15_by_zone/zone1.csv
DATE_COL=TIMESTAMP
TARGET_COL=POWER
DATASET_TAG=GEFCOM_zone1

# ===== Chronos-2 local model dir =====
CHRONOS_MODEL_DIR=/home/zhaopp/workspace/solar-energy/chronos-2
DEVICE_MAP=cuda

# ===== Rolling-window setup =====
TEST_RATIO=0.3
STRICT_TEST_ONLY=1

# dataset resolution: set "" to auto-infer mode delta
FREQ="1h"
SKIP_IRREGULAR_WINDOWS=1

# ===== Covariates (default OFF as agreed) =====
PAST_COVARIATE_COLS=""   # e.g. "temp,ghi"
FUTURE_COVARIATE_COLS="VAR78,VAR79,VAR157,VAR164,VAR169,VAR178" # e.g. "VAR78,VAR79,WS10"

# ===== Inference knobs =====
CONTEXT_LENGTH="" # empty -> do not override; otherwise integer
# CROSS_LEARNING=1		  # 0/1
MODEL_BATCH_SIZE=64
WINDOW_BATCH_SIZE=256

# QUANTILE_LEVELS="0.5"   # comma-separated, e.g. "0.1,0.5,0.9"
SAVE_ALL_REQUESTED_QUANTILES=0

# ===== Results =====
RESULTS_ROOT=./results/solar_chronos2_gefcom
EXP_ID=0

# Helper: summarize covariate settings into a short tag
cov_tag() {
	local pc="$1"
	local fc="$2"
	if [[ -z "$pc" && -z "$fc" ]]; then
		echo "covOff"
	elif [[ -n "$pc" && -z "$fc" ]]; then
		echo "covPast"
	elif [[ -z "$pc" && -n "$fc" ]]; then
		echo "covFut"
	else
		echo "covPastFut"
	fi
}

for QUANTILE_LEVELS in "0.4"; do
	for CROSS_LEARNING in 0; do
		for seq_len in 2048; do
			for pred_len in 1 4 72; do

				CTX_ARGS=()
				CTX_TAG="ctx${seq_len}"
				if [[ -n "${CONTEXT_LENGTH}" ]]; then
					CTX_ARGS+=(--context_length "${CONTEXT_LENGTH}")
					CTX_TAG="ctx${CONTEXT_LENGTH}"
				fi

				COV_TAG=$(cov_tag "${PAST_COVARIATE_COLS}" "${FUTURE_COVARIATE_COLS}")

				# Run-name: ONLY include knobs that actually exist in our Chronos experiments
				RUN_NAME="${DATASET_TAG}_${seq_len}_${pred_len}_Chronos2_${CTX_TAG}_cl${CROSS_LEARNING}_${COV_TAG}_freq${FREQ:-infer}_strict${STRICT_TEST_ONLY}_tr${TEST_RATIO}_skipIr${SKIP_IRREGULAR_WINDOWS}_mb${MODEL_BATCH_SIZE}_wb${WINDOW_BATCH_SIZE}_ql${QUANTILE_LEVELS//,/}-saveQ${SAVE_ALL_REQUESTED_QUANTILES}_Exp_${EXP_ID}"

				# Path-safe cleanup
				RUN_NAME=${RUN_NAME// /}
				RUN_NAME=${RUN_NAME//./p}  # 0.3 -> 0p3
				RUN_NAME=${RUN_NAME//:/}   # just in case
				RUN_NAME=${RUN_NAME//\//_} # avoid slashes

				python run_chronos2_zero_shot_solar_v4.py \
					--csv "${CSV}" \
					--date_col "${DATE_COL}" \
					--target_col "${TARGET_COL}" \
					--dataset_tag "${DATASET_TAG}" \
					--seq_len "${seq_len}" \
					--pred_len "${pred_len}" \
					--test_ratio "${TEST_RATIO}" \
					--strict_test_only "${STRICT_TEST_ONLY}" \
					--chronos_model_dir "${CHRONOS_MODEL_DIR}" \
					--device_map "${DEVICE_MAP}" \
					--past_covariate_cols "${PAST_COVARIATE_COLS}" \
					--future_covariate_cols "${FUTURE_COVARIATE_COLS}" \
					--freq "${FREQ}" \
					--skip_irregular_windows "${SKIP_IRREGULAR_WINDOWS}" \
					"${CTX_ARGS[@]}" \
					--cross_learning "${CROSS_LEARNING}" \
					--model_batch_size "${MODEL_BATCH_SIZE}" \
					--window_batch_size "${WINDOW_BATCH_SIZE}" \
					--quantile_levels "${QUANTILE_LEVELS}" \
					--save_all_requested_quantiles "${SAVE_ALL_REQUESTED_QUANTILES}" \
					--results_root "${RESULTS_ROOT}" \
					--run_name "${RUN_NAME}"

			done
		done
	done
done




# ===== Data =====
CSV=./dataset/GEFCom/gefcom15_by_zone/zone2.csv
DATE_COL=TIMESTAMP
TARGET_COL=POWER
DATASET_TAG=GEFCOM_zone2

# ===== Chronos-2 local model dir =====
CHRONOS_MODEL_DIR=/home/zhaopp/workspace/solar-energy/chronos-2
DEVICE_MAP=cuda

# ===== Rolling-window setup =====
TEST_RATIO=0.3
STRICT_TEST_ONLY=1

# dataset resolution: set "" to auto-infer mode delta
FREQ="1h"
SKIP_IRREGULAR_WINDOWS=1

# ===== Covariates (default OFF as agreed) =====
PAST_COVARIATE_COLS=""   # e.g. "temp,ghi"
FUTURE_COVARIATE_COLS="VAR78,VAR79,VAR157,VAR164,VAR169,VAR178" # e.g. "VAR78,VAR79,WS10"

# ===== Inference knobs =====
CONTEXT_LENGTH="" # empty -> do not override; otherwise integer
# CROSS_LEARNING=1		  # 0/1
MODEL_BATCH_SIZE=64
WINDOW_BATCH_SIZE=256

# QUANTILE_LEVELS="0.5"   # comma-separated, e.g. "0.1,0.5,0.9"
SAVE_ALL_REQUESTED_QUANTILES=0

# ===== Results =====
RESULTS_ROOT=./results/solar_chronos2_gefcom
EXP_ID=0

# Helper: summarize covariate settings into a short tag
cov_tag() {
	local pc="$1"
	local fc="$2"
	if [[ -z "$pc" && -z "$fc" ]]; then
		echo "covOff"
	elif [[ -n "$pc" && -z "$fc" ]]; then
		echo "covPast"
	elif [[ -z "$pc" && -n "$fc" ]]; then
		echo "covFut"
	else
		echo "covPastFut"
	fi
}

for QUANTILE_LEVELS in "0.4"; do
	for CROSS_LEARNING in 0; do
		for seq_len in 2048; do
			for pred_len in 1 4 72; do

				CTX_ARGS=()
				CTX_TAG="ctx${seq_len}"
				if [[ -n "${CONTEXT_LENGTH}" ]]; then
					CTX_ARGS+=(--context_length "${CONTEXT_LENGTH}")
					CTX_TAG="ctx${CONTEXT_LENGTH}"
				fi

				COV_TAG=$(cov_tag "${PAST_COVARIATE_COLS}" "${FUTURE_COVARIATE_COLS}")

				# Run-name: ONLY include knobs that actually exist in our Chronos experiments
				RUN_NAME="${DATASET_TAG}_${seq_len}_${pred_len}_Chronos2_${CTX_TAG}_cl${CROSS_LEARNING}_${COV_TAG}_freq${FREQ:-infer}_strict${STRICT_TEST_ONLY}_tr${TEST_RATIO}_skipIr${SKIP_IRREGULAR_WINDOWS}_mb${MODEL_BATCH_SIZE}_wb${WINDOW_BATCH_SIZE}_ql${QUANTILE_LEVELS//,/}-saveQ${SAVE_ALL_REQUESTED_QUANTILES}_Exp_${EXP_ID}"

				# Path-safe cleanup
				RUN_NAME=${RUN_NAME// /}
				RUN_NAME=${RUN_NAME//./p}  # 0.3 -> 0p3
				RUN_NAME=${RUN_NAME//:/}   # just in case
				RUN_NAME=${RUN_NAME//\//_} # avoid slashes

				python run_chronos2_zero_shot_solar_v4.py \
					--csv "${CSV}" \
					--date_col "${DATE_COL}" \
					--target_col "${TARGET_COL}" \
					--dataset_tag "${DATASET_TAG}" \
					--seq_len "${seq_len}" \
					--pred_len "${pred_len}" \
					--test_ratio "${TEST_RATIO}" \
					--strict_test_only "${STRICT_TEST_ONLY}" \
					--chronos_model_dir "${CHRONOS_MODEL_DIR}" \
					--device_map "${DEVICE_MAP}" \
					--past_covariate_cols "${PAST_COVARIATE_COLS}" \
					--future_covariate_cols "${FUTURE_COVARIATE_COLS}" \
					--freq "${FREQ}" \
					--skip_irregular_windows "${SKIP_IRREGULAR_WINDOWS}" \
					"${CTX_ARGS[@]}" \
					--cross_learning "${CROSS_LEARNING}" \
					--model_batch_size "${MODEL_BATCH_SIZE}" \
					--window_batch_size "${WINDOW_BATCH_SIZE}" \
					--quantile_levels "${QUANTILE_LEVELS}" \
					--save_all_requested_quantiles "${SAVE_ALL_REQUESTED_QUANTILES}" \
					--results_root "${RESULTS_ROOT}" \
					--run_name "${RUN_NAME}"

			done
		done
	done
done




# ===== Data =====
CSV=./dataset/GEFCom/gefcom15_by_zone/zone3.csv
DATE_COL=TIMESTAMP
TARGET_COL=POWER
DATASET_TAG=GEFCOM_zone3

# ===== Chronos-2 local model dir =====
CHRONOS_MODEL_DIR=/home/zhaopp/workspace/solar-energy/chronos-2
DEVICE_MAP=cuda

# ===== Rolling-window setup =====
TEST_RATIO=0.3
STRICT_TEST_ONLY=1

# dataset resolution: set "" to auto-infer mode delta
FREQ="1h"
SKIP_IRREGULAR_WINDOWS=1

# ===== Covariates (default OFF as agreed) =====
PAST_COVARIATE_COLS=""   # e.g. "temp,ghi"
FUTURE_COVARIATE_COLS="VAR78,VAR79,VAR157,VAR164,VAR169,VAR178" # e.g. "VAR78,VAR79,WS10"

# ===== Inference knobs =====
CONTEXT_LENGTH="" # empty -> do not override; otherwise integer
# CROSS_LEARNING=1		  # 0/1
MODEL_BATCH_SIZE=64
WINDOW_BATCH_SIZE=256

# QUANTILE_LEVELS="0.5"   # comma-separated, e.g. "0.1,0.5,0.9"
SAVE_ALL_REQUESTED_QUANTILES=0

# ===== Results =====
RESULTS_ROOT=./results/solar_chronos2_gefcom
EXP_ID=0

# Helper: summarize covariate settings into a short tag
cov_tag() {
	local pc="$1"
	local fc="$2"
	if [[ -z "$pc" && -z "$fc" ]]; then
		echo "covOff"
	elif [[ -n "$pc" && -z "$fc" ]]; then
		echo "covPast"
	elif [[ -z "$pc" && -n "$fc" ]]; then
		echo "covFut"
	else
		echo "covPastFut"
	fi
}

for QUANTILE_LEVELS in "0.4"; do
	for CROSS_LEARNING in 0; do
		for seq_len in 2048; do
			for pred_len in 1 4 72; do

				CTX_ARGS=()
				CTX_TAG="ctx${seq_len}"
				if [[ -n "${CONTEXT_LENGTH}" ]]; then
					CTX_ARGS+=(--context_length "${CONTEXT_LENGTH}")
					CTX_TAG="ctx${CONTEXT_LENGTH}"
				fi

				COV_TAG=$(cov_tag "${PAST_COVARIATE_COLS}" "${FUTURE_COVARIATE_COLS}")

				# Run-name: ONLY include knobs that actually exist in our Chronos experiments
				RUN_NAME="${DATASET_TAG}_${seq_len}_${pred_len}_Chronos2_${CTX_TAG}_cl${CROSS_LEARNING}_${COV_TAG}_freq${FREQ:-infer}_strict${STRICT_TEST_ONLY}_tr${TEST_RATIO}_skipIr${SKIP_IRREGULAR_WINDOWS}_mb${MODEL_BATCH_SIZE}_wb${WINDOW_BATCH_SIZE}_ql${QUANTILE_LEVELS//,/}-saveQ${SAVE_ALL_REQUESTED_QUANTILES}_Exp_${EXP_ID}"

				# Path-safe cleanup
				RUN_NAME=${RUN_NAME// /}
				RUN_NAME=${RUN_NAME//./p}  # 0.3 -> 0p3
				RUN_NAME=${RUN_NAME//:/}   # just in case
				RUN_NAME=${RUN_NAME//\//_} # avoid slashes

				python run_chronos2_zero_shot_solar_v4.py \
					--csv "${CSV}" \
					--date_col "${DATE_COL}" \
					--target_col "${TARGET_COL}" \
					--dataset_tag "${DATASET_TAG}" \
					--seq_len "${seq_len}" \
					--pred_len "${pred_len}" \
					--test_ratio "${TEST_RATIO}" \
					--strict_test_only "${STRICT_TEST_ONLY}" \
					--chronos_model_dir "${CHRONOS_MODEL_DIR}" \
					--device_map "${DEVICE_MAP}" \
					--past_covariate_cols "${PAST_COVARIATE_COLS}" \
					--future_covariate_cols "${FUTURE_COVARIATE_COLS}" \
					--freq "${FREQ}" \
					--skip_irregular_windows "${SKIP_IRREGULAR_WINDOWS}" \
					"${CTX_ARGS[@]}" \
					--cross_learning "${CROSS_LEARNING}" \
					--model_batch_size "${MODEL_BATCH_SIZE}" \
					--window_batch_size "${WINDOW_BATCH_SIZE}" \
					--quantile_levels "${QUANTILE_LEVELS}" \
					--save_all_requested_quantiles "${SAVE_ALL_REQUESTED_QUANTILES}" \
					--results_root "${RESULTS_ROOT}" \
					--run_name "${RUN_NAME}"

			done
		done
	done
done
