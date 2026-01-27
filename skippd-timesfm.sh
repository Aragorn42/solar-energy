export CUDA_VISIBLE_DEVICES=1
for pred_len in 16 288
do
  python run_timesfm_zero_shot_solar.py \
    --csv ./dataset/skippd.csv \
    --date_col date \
    --target_col OT \
    --dataset_tag SKIPPD \
    --seq_len 1024 \
    --pred_len $pred_len \
    --timesfm_dir /home/zhaopp/timesfm/src/timesfm-2.5-200m-pytorch \
    --strict_test_only 1 \
    --batch_size 1024 \
    --normalize_inputs 1 \
    --use_continuous_quantile_head 1 \
    --force_flip_invariance 1 \
    --infer_is_positive 1 \
    --fix_quantile_crossing 1
done
