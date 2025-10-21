#!/bin/bash

project_name=${PROJECT_NAME}
experiment_name=${EXPERIMENT_NAME}
dataset_name=${DATASET_NAME}

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <step> <device>"
    exit 1
fi

# 接收两个参数：step 和 device
step=$1
device=$2


echo "Merging model at global_step_$step..."

local_dir="/mnt/blob-hptrainingwesteurope-pretraining/qingguo/${project_name}/${experiment_name}/global_step_${step}/actor"
target_dir="${local_dir}/huggingface"

# 判断local_dir是否存在
if [ ! -d "$local_dir" ]; then
    echo "Local directory $local_dir does not exist, skipping..."
    continue
fi


python evaluation/model_merger.py --local_dir "$local_dir"

python evaluation/eval_qwen.py \
    --model_path $target_dir \
    --dataset $dataset_name \
    --cuda $device \
    --output_dir $target_dir/eval_results
