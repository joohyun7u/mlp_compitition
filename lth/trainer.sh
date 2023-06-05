#!/bin/bash

#SBATCH --job-name SwinIR.init-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y3
#SBATCH -o /data/worrospeed/logs/%x.out

version=.init

total_iteration=1800000
batch_size=16
learning_rate=2e-4
val_rate=0.1
datasets_dir=/local_datasets/MLinP/
model_name=SwinIR
model_save_dir=./model_save/    
validation_output_dir=./valid_output/    

current_step=610000

python -u trainer.py \
        --version=$version \
        --total_iteration=$total_iteration \
        --batch_size=$batch_size \
        --learning_rate=$learning_rate \
        --val_rate=$val_rate \
        --datasets_dir=$datasets_dir \
        --model_name=$model_name \
        --model_save_dir=$model_save_dir \
        --validation_output_dir=$validation_output_dir \
        --current_step=$current_step 
        
exit 0
