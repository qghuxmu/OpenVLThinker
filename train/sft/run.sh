#!/bin/bash

# This script checks for GPUs with >=90% free memory and
# launches the training command when at least two are found.

while true; do
    free_gpus=()
    i=0
    # Use nvidia-smi to list each GPU's total and used memory.
    # The output is in the format: "total, used"
    while IFS=',' read -r total used; do
        # Remove any leading/trailing whitespace.
        total=$(echo "$total" | xargs)
        used=$(echo "$used" | xargs)
        # Calculate the fraction of free memory.
        free_fraction=$(echo "scale=2; ($total - $used) / $total" | bc -l)
        # Compare if free_fraction >= 0.90 (i.e., 90% free).
        cmp=$(echo "$free_fraction >= 0.90" | bc -l)
        if [ "$cmp" -eq 1 ]; then
            free_gpus+=("$i")
        fi
        i=$((i+1))
    done < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits)

    if [ "${#free_gpus[@]}" -ge 4 ]; then
        echo "Detected at least 4 free GPUs: ${free_gpus[0]}, ${free_gpus[1]}, ${free_gpus[2]}, ${free_gpus[3]}"
        # Construct and run the command using the first two free GPUs.
        CUDA_VISIBLE_DEVICES="${free_gpus[0]},${free_gpus[1]},${free_gpus[2]},${free_gpus[3]}" llamafactory-cli train examples/train_full/qwen2vl_full_sft_qwq.yaml
        break
    else
        echo "Not enough free GPUs detected. Checking again in 30 seconds..."
        sleep 30
    fi
done
