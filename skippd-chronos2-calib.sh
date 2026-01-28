python run_chronos2_calib_df.py \
  --csv ./dataset/skippd.csv --date_col date --target_col skippd \
  --target_col OT \
  --dataset_tag SKIPPD \
  --seq_len 1024 --pred_len 1 \
  --strict_test_only 0 \
  --model_name /home/zhaopp/chronos-forecasting-main/chronos-2 \
  --device_map cuda \
  --quantiles 0.1,0.4,0.45,0.5,0.9 \
  --freq 15min --fill_method interpolate \
  --do_calib 0 --cap 30.1 \
  --log_csv ./results/chronos2/metrics_log.csv

python run_chronos2_calib_df.py \
  --csv ./dataset/skippd.csv --date_col date --target_col skippd \
  --target_col OT \
  --dataset_tag SKIPPD \
  --seq_len 1024 --pred_len 1 \
  --strict_test_only 0 \
  --model_name /home/zhaopp/chronos-forecasting-main/chronos-2 \
  --device_map cuda \
  --quantiles 0.1,0.4,0.45,0.5,0.9 \
  --freq 15min --fill_method interpolate \
  --do_calib 1 --calib_ratio 0.5 --cap 30.1 \
  --log_csv ./results/chronos2/metrics_log.csv
