export CUDA_VISIBLE_DEVICES=5

for pred_len in 1 4 72
do
    python run_timesfm_zero_shot_solar.py \
      --csv ./dataset/GEFCom/task15.csv \
      --date_col date \
      --target_col zone3 \
      --dataset_tag GEFCOM_TASK153 \
      --seq_len 336 \
      --pred_len $pred_len \
      --timesfm_dir /home/huangyx/workspace/hf_models/timesfm-2.5-200m-pytorch \
      --strict_test_only 1 \
      --batch_size 64
done