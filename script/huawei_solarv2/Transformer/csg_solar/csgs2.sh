export CUDA_VISIBLE_DEVICES=2

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=Transformer

root_path_name=./dataset/csg_solar/
data_path_name=Solar_station_site_2_Nominal_capacity-130MW.csv
model_id_name=CSGS2_MS
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
      --features MS \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 6 \
      --dec_in 6 \
      --c_out 6 \
      --e_layers 3 \
      --d_layers 2 \
      --d_model 512 \
      --d_ff 1024 \
      --n_heads 4 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --target 'Power (MW)' \
      --itr 1 --batch_size 64 --learning_rate 0.0005 >logs/LongForecasting/${model_name}/${model_name}'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done