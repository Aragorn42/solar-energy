current_time=$(date +'%H:%M:%S')
echo $current_time

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/DLinear" ]; then
    mkdir ./logs/DLinear
fi

seq_len=336
model_name=DLinear
dataset=("skippd.csv")
enc_in=(1)
batch_size=(64)
learning_rate=(0.001)
scale=10
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
        --des 'Exp' \
        --itr 1 \
        --patience 10 \
        --batch_size ${batch_size[$i]} \
        --learning_rate ${learning_rate[$i]}  \
        --train_epochs 100 > ./logs/DLinear/$model_name'_'${dataset[$i]%%.*}'_'$seq_len'_'$pred_len'_'$current_time.txt
    done
done

current_time=$(date +'%H:%M:%S')
echo $current_time