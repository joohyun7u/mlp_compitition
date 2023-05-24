#!/bin/bash
datasets_dir=/local_datasets/MLinP/train/scan
model_name=UNet
model_save_dir=./model_save    
model_pth_name=UNET
output_dir=/local_datasets/MLinP/train/image_output/

version=alpha

python -u tester.py \
        --datasets_dir=$datasets_dir \
        --model=$model_name \
        --model-version=$version \
        --model_save_dir=$model_save_dir \
        --model_pth_name=$model_pth_name \
        --output_dir=$output_dir  
        
exit 0
