#!/bin/bash

# wandb 配置
source /home/sunyuanxu/miniconda3/etc/profile.d/conda.sh
conda activate new_omnisafe
export WANDB_API_KEY="07a341b17ceb53dc437556bdf4f18df47002595d"

# 设置环境变量
export PYTHONPATH="/home/sunyuanxu/FPO-Algorithm:$PYTHONPATH"
# 默认参数
# ALGO="PPOLag"
# ALGO="RCPO"
# ALGO="P3O"
# ALGO="FOCOPS"
# ALGO="CPO"
# ALGO="TRPOPID"
# ALGO="PCPO"
ALGO="FPO"

# ENV_ID="SafetyPointCircle1-v0"
# ENV_ID="SafetyPointGoal1-v0"
# ENV_ID="SafetyPointPush1-v0"
# ENV_ID="SafetyPointButton1-v0"
ENV_ID="SafetyCarCircle1-v0" ## low_r
# ENV_ID="SafetyCarPush1-v0" ## low_r
# ENV_ID="SafetyCarGoal1-v0"
# ENV_ID="SafetyCarButton1-v0" ## 
# ENV_ID="SafetyHalfCheetahVelocity-v1" ## low_r
# ENV_ID="SafetySwimmerVelocity-v1" 
# ENV_ID="SafetyWalker2dVelocity-v1" ## low_r
# ENV_ID="SafetyAntVelocity-v1" ##low_r ！
# ENV_ID="SafetyHumanoidVelocity-v1" ##low_r！
# ENV_ID="SafetyHopperVelocity-v1" #low_r
export CUDA_VISIBLE_DEVICES=0
PARALLEL=1
TOTAL_STEPS=5000000
DEVICE="cuda:0"
VECTOR_ENV_NUMS=20
TORCH_THREADS=16
SEED=0
TASK_DESCRIPTION="{bs20000}_{negative_slope=0.01}_{recover}"
# TASK_DESCRIPTION="{divide_stand}_{out_in_pos_f)}_{in_r-f}_{bs20000}_{negative_slope=0.001}"
# TASK_DESCRIPTION="{out|in_pos_f}_{in_(r-f*lag)/(1+lag)}_{adv_f>0}_{negative_slope=0.001}_{250ep}"
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
    --task_description "${TASK_DESCRIPTION}" \
    --batch-size ${BATCH_SIZE} \
    --seed ${SEED} \
    "$@"