#!/bin/bash

#SBATCH --job-name erase
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=24G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w ariel-v9
#SBATCH -o logs/slurm-%A-%x.out
base_dir=/local_datasets/MLinP

fold=$base_dir'/test/scan/'
if [ -d $fold ]; then
    echo $fold' exist'
    rm -rf $fold
fi

fold=$base_dir'/train/scan/'
if [ -d $fold ]; then
    echo $fold' exist'
    rm -rf $fold
fi

fold=$base_dir'/train/clean/'
if [ -d $fold ]; then
    echo $fold' exist'
    rm -rf $fold
fi

stat $fold
du $base_dir