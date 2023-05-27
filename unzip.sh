#!/bin/bash

#SBATCH --job-name unzip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=24G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -w ariel-v8
#SBATCH -o logs/slurm-%A-%x.out
base_dir=/local_datasets/MLinP

fold=$base_dir'/test/scan/'
if [ -d $fold ]; then
    echo $fold' exist'
else
    echo $fold' not exist'
    mkdir -p $fold
fi


file=$fold'/Wireframe57.00108.02.tif'
target_zip='test_scan.zip'
echo $file
if [ -f $file ]; then
    echo $target_zip' 이미 압축 해제 되었음'
else
    rm -rf $fold
    mkdir -p $fold
    echo $target_zip' 압축 해제중'
    unzip $target_zip -d $fold
fi
stat $fold


fold=$base_dir'/train/scan/'
if [ -d $fold ]; then
    echo $fold' exist'
else
    echo $fold' not exist'
    mkdir -p $fold
fi

file=$fold'Wireframe57.00116.01.tif'
target_zip='train_scan.zip'
echo $file
if [ -f $file ]; then
    echo $target_zip' 이미 압축 해제 되었음'
else
    echo heheh
    rm -rf $fold
    mkdir -p $fold
    echo $target_zip' 압축 해제중'
    unzip $target_zip -d $fold
fi
stat $fold

fold=$base_dir'/train/clean/'
if [ -d $fold ]; then
    echo $fold' exist'
else
    echo $fold' not exist'
    mkdir -p $fold
fi

file=$fold'Wireframe57.00116.01.tif'
target_zip='train_clean.zip'
if [ -f $file ]; then
    echo $target_zip' 이미 압축 해제 되었음'
else
    rm -rf $fold
    mkdir -p $fold
    echo $target_zip' 압축 해제중'
    unzip $target_zip -d $fold
fi
stat $fold
chmod 755 $base_dir -R
du $base_dir