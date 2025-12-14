
HORIZONS=(1 4 72)
# 固定参数
LAG=48
PERIOD=24
for i in 1 2 3
do
    # 循环遍历每一个 horizon
    for HORIZON in "${HORIZONS[@]}"
    do
        # 调用 python 脚本并传入参数
        python GBDT.py \
            --lag_count $LAG \
            --forecast_horizon $HORIZON \
            --period_length $PERIOD \
            --data_path "./zone_$i.csv" \
            --cap 1
        # 检查上一个命令是否执行成功
        if [ $? -eq 0 ]; then
            echo "Success for Horizon $HORIZON"
        else
            echo "Error occurred at Horizon $HORIZON"
        fi
        echo "" # 打印空行
    done
done

# HORIZONS=(1 16)
# LAG=96
# PERIOD=96
# cap=(50 130 30 130 110 35 30 30)
# for i in {1..8}
# do
#     # 循环遍历每一个 horizon
#     for HORIZON in "${HORIZONS[@]}"
#     do
#         python GBDT.py \
#             --lag_count $LAG \
#             --forecast_horizon $HORIZON \
#             --period_length $PERIOD \
#             --data_path "./dataset/solar_stations/Solar station site $i (Nominal capacity-${cap[i-1]}MW).xlsx" \
#             --cap ${cap[$i-1]}
#         if [ $? -eq 0 ]; then
#             echo "Success for Horizon $HORIZON"
#         else
#             echo "Error occurred at Horizon $HORIZON"
#         fi
#         echo ""
#     done
# done
# HORIZONS=(1 16)
# # 固定参数
# LAG=96
# PERIOD=96
# for i in 1
# do
#     # 循环遍历每一个 horizon
#     for HORIZON in "${HORIZONS[@]}"
#     do
#         # 调用 python 脚本并传入参数
#         python GBDT.py \
#             --lag_count $LAG \
#             --forecast_horizon $HORIZON \
#             --period_length $PERIOD \
#             --data_path "./dataset/skippd.csv" \
#             --cap 30
#         # 检查上一个命令是否执行成功
#         if [ $? -eq 0 ]; then
#             echo "Success for Horizon $HORIZON"
#         else
#             echo "Error occurred at Horizon $HORIZON"
#         fi
#         echo "" # 打印空行
#     done
#done