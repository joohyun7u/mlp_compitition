#!/bin/bash

#SBATCH --job-name UNet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y1
#SBATCH -o /data/worrospeed/logs/%x-%2t.out

version=alpha

epoch=200
batch_size=64
lr=0.01
val_rate=0.1
isCV=0
datasets_dir=/local_datasets/MLinP/
model_name=UNet
model_save_dir=./model_save/    
loss_save_dir=./loss_save/    

python -u trainer.py \
        --version=$version \
        --epoch=$epoch \
        --batch_size=$batch_size \
        --lr=$lr \
        --val_rate=$val_rate \
        --isCV=$isCV \
        --datasets_dir=$datasets_dir \
        --model=$model_name \
        --model_save_dir=$model_save_dir \
        --loss_save_dir=$loss_save_dir  
        
exit 0