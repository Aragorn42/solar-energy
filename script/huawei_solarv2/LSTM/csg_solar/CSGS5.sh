#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi



# 基础参数
seq_len=336
model_name=LSTM_baseline    # ✅ 改成你的LSTM模型类名
root_path_name=./dataset/csg_solar/
data_path_name=Solar_station_site_5_Nominal_capacity-110MW.csv
model_id_name=CSGS5
data_name=custom_solar
random_seed=2021

if [ ! -d "./logs/LongForecasting/${model_name}" ]; then
    mkdir ./logs/LongForecasting/${model_name}
fi

# 循环不同预测步长
for pred_len in 1 16 288
do
    python -u run_longExp_solarv2.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 1 \
      --c_out 1 \
      --d_model 64 \
      --d_ff 128 \
      --dropout 0.2 \
      --train_epochs 50 \
      --patience 10 \
      --lradj 'type3' \
      --pct_start 0.3 \
      --target 'Power (MW)' \
      --itr 1 \
      --batch_size 64 \
      --learning_rate 0.001 > logs/LongForecasting/${model_name}/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log 
done