#!/bin/bash

#SBATCH --job-name resnet
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
# DnCNN, ResNet18 34 50 101 152
model=ResNet18

#tar -xcvf /data/datasets/ImageNet.tar -C /local_datasets/

python -u $py_dir \
        --epoch=150 \
        --batch_size=14 \
        --lr=0.01 \
        --model=$model \


echo 'done'
exit 0