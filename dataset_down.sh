#!/bin/bash

file='./test_scan.zip'
if [ -f $file ]; then
    echo $file+' exist'
else
    echo $file+' not exist'
    gdown https://drive.google.com/uc?id=1uuuvlmPtrjGuG83tAzxMbHBqHSlFpGjr
    # gdown https://drive.google.com/uc?id=1zmfrXzT9lnLg7NlQ-hXekZlyX9aGNNqj
fi

file='./train_clean.zip'
if [ -f $file ]; then
    echo $file+' exist'
else
    echo $file+' not exist'
    gdown https://drive.google.com/uc?id=18larBxdbutp0u1GCHo7QJ2qo4f48SU0C #cjh 드라이브
    # gdown https://drive.google.com/uc?id=1cqxSVFxfonx5qKVIdfByOq7uYdNZY8ea
fi

file='./train_scan.zip'
if [ -f $file ]; then
    echo $file+' exist'
else
    echo $file+' not exist'
    gdown https://drive.google.com/uc?id=17YqE_aLUEwwVgTSzoZ2Awy5A3A8BmssV
    # gdown https://drive.google.com/uc?id=16zAeGDmqbAvn7Iy8V-mBylKx6rG-wgLD
fi

echo 'done!'
exit 0
