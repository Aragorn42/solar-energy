#!/usr/bin/env sh
# 顺序运行当前目录下的 csgs3.sh ~ csgs8.sh

set -e  # 任一脚本报错则停止整个流程

for x in 1 2 3 4 5 6 7 8; do
    script="./script/huawei_solarv2/tcn/csg_solar/csgs${x}.sh"
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