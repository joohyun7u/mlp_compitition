#!/bin/bash

#SBATCH --job-name SwinIR.32Win
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -o /data/hch2454/logs/test-%x.out

version=.32Win

datasets_dir=/local_datasets/MLinP/train/clean/
model_name=SwinIR
model_save_dir=./model_save    
model_pth_name=SwinIR.32Win
output_dir=/local_datasets/MLinP/train/MSN/

python -u tester.py \
        --datasets_dir=$datasets_dir \
        --model=$model_name \
        --model_version=$version \
        --model_save_dir=$model_save_dir \
        --model_pth_name=$model_pth_name \
        --output_dir=$output_dir \
        
exit 0
