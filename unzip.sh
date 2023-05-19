#!/bin/bash

#SBATCH --job-name unzip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=24G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w ariel-v2
#SBATCH -o logs/slurm-%A-%x.out


fold='/local_datasets/MLinP/test/scan'
if [ -d $fold ]; then
    echo $fold' exist'
else
    echo $fold' not exist'
    mkdir -p $fold
fi

file=$fold'/BookOfMaking.00016.01.tif'
target_zip='test_scan.zip'
if [ -f $file ]; then
    echo $target_zip' 이미 압축 해제 되었음'
else
    rm -rf $fold
    mkdir -p $fold
    echo $target_zip' 압축 해제중'
    unzip $target_zip -d $fold
fi



fold='/local_datasets/MLinP/train/scan'
if [ -d $fold ]; then
    echo $fold' exist'
else
    echo $fold' not exist'
    mkdir -p $fold
fi

file=$fold'BookOfMaking.00002.02.tif'
target_zip='train_scan.zip'
if [ -f $file ]; then
    echo $target_zip' 이미 압축 해제 되었음'
else
    rm -rf $fold
    mkdir -p $fold
    echo $target_zip' 압축 해제중'
    unzip $target_zip -d $fold
fi


fold='/local_datasets/MLinP/train/clean'
if [ -d $fold ]; then
    echo $fold' exist'
else
    echo $fold' not exist'
    mkdir -p $fold
fi

file=$fold'BookOfMaking.00002.02.tif'
target_zip='train_clean.zip'
if [ -f $file ]; then
    echo $target_zip' 이미 압축 해제 되었음'
else
    rm -rf $fold
    mkdir -p $fold
    echo $target_zip' 압축 해제중'
    unzip $target_zip -d $fold
fi