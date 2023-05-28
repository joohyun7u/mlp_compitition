#!/bin/bash

#SBATCH --job-name SwinIR.init
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y1
#SBATCH -o /data/worrospeed/logs/test-%x.out

version=.init

datasets_dir=/local_datasets/MLinP/test/scan/
model_name=SwinIR
model_save_dir=./model_save    
model_pth_name=SwinIR.init
output_dir=./test_output
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
