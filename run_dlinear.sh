export CUDA_VISIBLE_DEVICES=4

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear

root_path_name=./dataset/
data_path_name=skippd.csv
model_id_name=SKIPPD
data_name=custom_solar

random_seed=2021

if [ ! -d "./logs/LongForecasting/${model_name}" ]; then
    mkdir ./logs/LongForecasting/${model_name}
fi

for pred_len in 1 16 288
do
    python -u run_longExp_solarv2.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}'_'${pred_len} \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --target 'OT' \
      --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/${model_name}/${model_name}'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

export CUDA_VISIBLE_DEVICES=4

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear

root_path_name=./dataset/GEFCom/
data_path_name=task15.csv
model_id_name=GEFCOM_TASK15
data_name=custom_solar

random_seed=2021

if [ ! -d "./logs/LongForecasting/${model_name}" ]; then
    mkdir ./logs/LongForecasting/${model_name}
fi

for pred_len in 1 4 72
do
    python -u run_longExp_solarv2.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}'_'${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 3 \
      --des 'Exp' \
      --train_epochs 100 \
      --patience 10 \
      --target 'zone3' \
      --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/${model_name}/${model_name}'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

#!/usr/bin/env sh
# 顺序运行当前目录下的 csgs3.sh ~ csgs8.sh

set -e  # 任一脚本报错则停止整个流程

for x in 1 2 3 4 5 6 7 8; do
    script="./script/huawei_solarv2/DLinear/csg_solar/csgs${x}.sh"
    if [ -x "$script" ]; then
        echo ">>> Running: $script"
        "$script"
    elif [ -f "$script" ]; then
        echo ">>> Running (via sh): $script"
        sh "$script"
    else
        echo "!!! Warning: $script not found, skip."
    fi
done

echo "All done."