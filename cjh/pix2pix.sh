#!/bin/bash

#SBATCH --job-name pp2-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w ariel-v8
#SBATCH -o logs/slurm-%A-%x.out

source /data/joohyun7u/cjh/sh/setup.sh

current_time=$(date "+%Y%m%d-%H:%M:%S")
echo $current_time
py_dir=./models/pix2pix2.py
dataset_dir=/home/joohyun7u/dataset/ff
# DnCNN, ResNet18 34 50 101 152, RFDN, DRLN, pix2pix, pix2pix2
model=pix2pix2

#tar -xcvf /data/datasets/ImageNet.tar -C /local_datasets/

python -u $py_dir \
        --epoch=1600 \
        --batch_size=1 \
        --lr=0.01 \
        --model=$model \
        --summary=False \
        --val=0.001 \
        --loss=2 \
        --noise_train=False \
        --val_best_save=False \
        --train_img_size=256 \


echo 'done'
exit 0