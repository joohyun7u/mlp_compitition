#!/bin/bash

#SBATCH --job-name model_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w ariel-v8
#SBATCH -o ./logs/slurm-%A-%x.out

epoch=10
batch_size=32
lr=0.001
val_rate=0.1
isCV=0
datasets_dir=C:/local_datasets/MLinP/
model_name=DnCNN
model_save_dir=./model_save/    
loss_save_dir=./loss_save/    

python -u trainer.py \
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