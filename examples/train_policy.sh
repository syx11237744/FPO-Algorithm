#!/bin/bash

# wandb 配置
source /home/sunyuanxu/miniconda3/etc/profile.d/conda.sh
conda activate omnisafe
export WANDB_API_KEY="07a341b17ceb53dc437556bdf4f18df47002595d"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0 

# 默认参数
ALGO="PPOLag"
ENV_ID="SafetyPointGoal1-v0"
PARALLEL=1
TOTAL_STEPS=10000000
DEVICE="cuda:0"
VECTOR_ENV_NUMS=20
TORCH_THREADS=16
# !
# cost_limit: 0

# 运行训练脚本
python train_policy.py \
    --algo ${ALGO} \
    --env-id ${ENV_ID} \
    --parallel ${PARALLEL} \
    --total-steps ${TOTAL_STEPS} \
    --device ${DEVICE} \
    --vector-env-nums ${VECTOR_ENV_NUMS} \
    --torch-threads ${TORCH_THREADS} \
    --use_wandb \
    "$@"