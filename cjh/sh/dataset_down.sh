#!/bin/bash

file='./test_scan.zip'
if [ ! -e $file]
then
    echo $file+' not exist'
    gdown https://drive.google.com/uc?id=16zAeGDmqbAvn7Iy8V-mBylKx6rG-wgLD
else
    echo $file+' exist'
fi

file='./train_clean.zip'
if [ ! -e $file]
then
    echo $file+' not exist'
    gdown https://drive.google.com/uc?id=1cqxSVFxfonx5qKVIdfByOq7uYdNZY8ea
else
    echo $file+' exist'
fi


file='./train_scan.zip'
if [ ! -e $file]
then
    echo $file+' not exist'
    gdown https://drive.google.com/uc?id=1zmfrXzT9lnLg7NlQ-hXekZlyX9aGNNqj
else
    echo $file+' exist'
fi


echo 'done!'
exit 0
