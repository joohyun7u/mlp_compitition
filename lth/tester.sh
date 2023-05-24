#!/bin/bash

#SBATCH --job-name UNetV2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y1
#SBATCH -o /data/worrospeed/logs/test-%x-%2t.out

version=alpha

datasets_dir=/local_datasets/MLinP/train/scan/
model_name=UNetV2
model_save_dir=./model_save    
model_pth_name=UNetV2alpha
output_dir=/data/worrospeed/outputs/
display_num=5

python -u tester.py \
        --datasets_dir=$datasets_dir \
        --model=$model_name \
        --model_version=$version \
        --model_save_dir=$model_save_dir \
        --model_pth_name=$model_pth_name \
        --output_dir=$output_dir \
        --display_num=$display_num
        
exit 0
