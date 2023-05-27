#!/bin/bash

#SBATCH --job-name kair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w ariel-v8
#SBATCH -o logs/slurm-%A-%x.out

source /data/joohyun7u/cjh/sh/setup.sh

current_time=$(date "+%Y%m%d-%H:%M:%S")
echo $current_time
py_dir=./models/swinir.py
dataset_dir=/home/joohyun7u/dataset/ff
# DnCNN, ResNet18 34 50 101 152, RFDN, DRLN, pix2pix, pix2pix2
model=swinir

#tar -xcvf /data/datasets/ImageNet.tar -C /local_datasets/

python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 -u KAIR/main_train_psnr.py --opt KAIR/options/swinir/train_swinir_denoising_color.json  --dist True


echo 'done'
exit 0