#!/bin/bash

#SBATCH --job-name Make_CSV
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=24G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w ariel-v2
#SBATCH -o logs/slurm-%A-%x.out

source /data/joohyun7u/cjh/sh/setup.sh

current_time=$(date "+%Y%m%d-%H:%M:%S")
echo $current_time
py_dir=./make_csv.py
dataset_dir=/home/joohyun7u/dataset/ff

#lrs=(4e-2, 8e-2, 4e-3, 8e-3, 4e-4, 4e-5)
#lr="{lrs[SLURM_ARRAY_TASK_ID]}"

#tar -xcvf /data/datasets/ImageNet.tar -C /local_datasets/

python -u $py_dir

# python train.py \
#     --cfg-options \
#         optimizer.lr=$lr

echo 'done'
exit 0