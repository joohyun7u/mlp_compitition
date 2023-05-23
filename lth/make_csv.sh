#!/bin/bash

#SBATCH --job-name makeCSV
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y1
#SBATCH -o /data/worrospeed/logs/%x-%2t.out

datasets_dir=/local_datasets/MLinP/
csv_dir=./csv_save/
model_name=UNet
model_pth_dir=./model_save/
model_pth_name=UNet

python -u make_csv.py \
        --datasets_dir=$datasets_dir \
        --csv_dir=$csv_dir \
        --model_name=$model_name \
        --model_pth_dir=$model_pth_dir \
        --model_pth_name=$model_pth_name \

exit 0
