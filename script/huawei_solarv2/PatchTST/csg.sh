export CUDA_VISIBLE_DEVICES=4

if [ ! -d "./logs" ]; then
	mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
	mkdir ./logs/LongForecasting
fi

model_name=PatchTST
cap=(50 130 30 130 110 35 30 30)

root_path_name=./dataset/csg_solar/
data_name=custom_solar

random_seed=2021

for i in {1..8}; do
	data_path_name=Solar_station_site_${i}_Nominal_capacity-${cap[i - 1]}MW.csv
	model_id_name=CSGS${i}
	for seq_len in 96; do
		for pred_len in 1 16 288; do
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
				--e_layers 3 \
				--n_heads 16 \
				--d_model 128 \
				--d_ff 256 \
				--dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 --des 'Exp' \
				--train_epochs 100 --patience 10 --lradj 'TST' \
				--pct_start 0.2 --target 'Power (MW)' \
				--itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/${model_name}/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
		done
	done
done
