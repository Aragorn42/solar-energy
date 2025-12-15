export CUDA_VISIBLE_DEVICES=3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=GBDT

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
      --des 'Exp' \
      --train_epochs 1000 \
      --patience 50 \
      --target 'OT' \
      --use_gpu False \
      --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/${model_name}/${model_name}'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done