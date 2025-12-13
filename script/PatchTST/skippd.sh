export CUDA_VISIBLE_DEVICES=3
current_time=$(date +'%H:%M:%S')
echo $current_time

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi



seq_len=336
model_name=PatchTST
dataset=("skippd.csv")
enc_in=(1)
batch_size=(64)
learning_rate=(0.0001)
scale=10

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

for i in 0;do
    for pred_len in 1 16 288;do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path ${dataset[$i]} \
        --model_id ${dataset[$i]%%.*}_$pred_len \
        --model $model_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in ${enc_in[$i]} \
        --e_layers 3 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --lradj 'TST'\
        --pct_start 0.2\
        --itr 1 \
        --patience 10 \
        --batch_size ${batch_size[$i]} \
        --learning_rate ${learning_rate[$i]}  \
        --train_epochs 100 > ./logs/$model_name/$model_name'_'${dataset[$i]%%.*}'_'$seq_len'_'$pred_len'_'$current_time.txt
    done
done

current_time=$(date +'%H:%M:%S')
echo $current_time