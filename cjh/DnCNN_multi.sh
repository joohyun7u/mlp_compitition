#!/bin/bash

#SBATCH --job-name dncnn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=24G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w ariel-v2
#SBATCH --array 0-4%1
#SBATCH -o logs/slurm-%A_%a-%x.out

source /data/joohyun7u/cjh/sh/setup.sh

lrs=(4e-2 8e-2 4e-3 8e-3)
batchs=(32 64 128 256)
# combines=()
# for lr_ in "${lrs[@]}"; do
#     for batch_ in "${batchs[@]}"; do
#         combines+=$lr_
#         combines+=$batch_
#     done
# done

echo $combines
current_time=$(date "+%Y%m%d-%H:%M:%S")
echo $current_time
py_dir=./models/DnCNN.py
dataset_dir=/home/joohyun7u/dataset/ff
# i=${SLURM_ARRAY_TASK_ID*2}
# j=${i+1}
lr="${lrs[SLURM_ARRAY_TASK_ID]}"
batch="${batchs[SLURM_ARRAY_TASK_ID]}"

#tar -xcvf /data/datasets/ImageNet.tar -C /local_datasets/

python -u $py_dir \
        --epoch=150 \
        --batch_size=64 \
        --lr=$lr \
        # --csv=$save_dir \
        # --model=$model \


echo 'done'
exit 0