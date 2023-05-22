#!/bin/bash

#SBATCH --job-name rfdn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=24G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w ariel-v8
#SBATCH -o logs/slurm-%A-%x.out

source /data/joohyun7u/cjh/sh/setup.sh

current_time=$(date "+%Y%m%d-%H:%M:%S")
echo $current_time
py_dir=./models/DnXXX.py
dataset_dir=/home/joohyun7u/dataset/ff
# DnCNN, ResNet18 34 50 101 152, RFDN
model=RFDN

#tar -xcvf /data/datasets/ImageNet.tar -C /local_datasets/

python -u $py_dir \
        --epoch=300 \
        --batch_size=32 \
        --lr=0.001 \
        --model=$model \
        --summary=False \
        --val=0.1 \
        --loss=1 \


echo 'done'
exit 0