export CUDA_VISIBLE_DEVICES=7

for pred_len in 1 16 288
do
    python run_timesfm_zero_shot_solar.py \
      --csv ./dataset/csg_solar/Solar_station_site_7_Nominal_capacity-30MW.csv \
      --date_col date \
      --target_col "Power (MW)" \
      --dataset_tag CSGS7 \
      --seq_len 336 \
      --pred_len $pred_len \
      --timesfm_dir /home/huangyx/workspace/hf_models/timesfm-2.5-200m-pytorch \
      --strict_test_only 1 \
      --batch_size 64
done