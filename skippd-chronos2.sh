for seq_len in 352 704 1024
do
  python run_chronos-2.py \
    --csv ./dataset/skippd.csv \
    --date_col date \
    --target_col OT \
    --dataset_tag SKIPPD \
    --seq_len $seq_len \
    --pred_len 1 \
    --test_ratio 0.3 \
    --strict_test_only 1 \
    --device_map cuda \
    --quantiles 0.35,0.4,0.45,0.5,0.55,0.6
done

