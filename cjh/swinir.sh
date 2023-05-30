#!/bin/bash

#SBATCH --job-name sir
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w ariel-v2
#SBATCH -o logs/slurm-%A-%x.out

source /data/joohyun7u/cjh/sh/setup.sh

current_time=$(date "+%Y%m%d-%H:%M:%S")
echo $current_time
py_dir=./models/swinir.py
dataset_dir=/home/joohyun7u/dataset/ff
# DnCNN, ResNet18 34 50 101 152, RFDN, DRLN, pix2pix, pix2pix2, swinir, swinirv2
model=swinirv2

#tar -xcvf /data/datasets/ImageNet.tar -C /local_datasets/

python -u $py_dir \
        --epoch=800 \
        --batch_size=32 \
        --lr=2e-4 \
        --model=$model \
        --summary=False \
        --val=0.001 \
        --loss=2 \
        --noise_train=False \
        --val_best_save=False \
        --train_img_size=32 \


echo 'done'
exit 0