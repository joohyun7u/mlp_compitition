#!/bin/bash
datasets_dir=/local_datasets/MLinP/train/scan/
model_name=UNetV2
model_save_dir=./model_save    
model_pth_name=UNetV2alpha
output_dir=/data/worrospeed/outputs/
display_num=5

version=alpha

python -u tester.py \
        --datasets_dir=$datasets_dir \
        --model=$model_name \
        --model-version=$version \
        --model_save_dir=$model_save_dir \
        --model_pth_name=$model_pth_name \
        --output_dir=$output_dir \
        --display_num=$dis
        
exit 0
