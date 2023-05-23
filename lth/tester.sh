#!/bin/bash
datasets_dir=C:/local_datasets/MLinP/train/scan
model_name=UNet
model_save_dir=./model_save    
output_dir=./outputs 

python -u tester.py \
        --datasets_dir=$datasets_dir \
        --model=$model_name \
        --model_save_dir=$model_save_dir \
        --output_dir=$output_dir  
        
exit 0
