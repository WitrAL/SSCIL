#!/bin/bash

# 指定使用 GPU 6 和 7
export CUDA_VISIBLE_DEVICES=6

# echo "开始运行 Icarl CUB 10%..."



nohup python main.py -i 21 -d imageneta > drawLog/INA_25%_03-08_beta08.log 2>&1 &
wait

nohup python main.py -i 62 -d cifar224 > drawLog/CIFAR_25%_03-08_beta08.log 2>&1 &
wait

nohup python main.py -i 55 -d cub > drawLog/CUB_25%_03-09_beta08.log 2>&1 &
wait



# nohup python main.py -i 120 -d imageneta > compareLog/INA/quantile/10%_RANPAC_06.log 2>&1 &
# wait

# nohup python main.py -i 109 -d imagenetr > compareLog/INR/quantile/25%_RANPAC_06.log 2>&1 &
# wait


# nohup python main.py -i 120 -d imagenetr > compareLog/INR/quantile/5%_RANPAC_06.log 2>&1 &
# wait

# nohup python main.py -i 109 -d cifar224 > compareLog/CIFAR/quantile/25%_RANPAC_06.log 2>&1 &
# wait


# nohup python main.py -i 120 -d cifar224 > compareLog/CIFAR/quantile/5%_RANPAC_06.log 2>&1 &
# wait




echo "所有任务已完成 ✅"
