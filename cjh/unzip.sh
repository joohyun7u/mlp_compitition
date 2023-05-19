#!/bin/bash

file='../test_scan.zip'
if [ ! -e $file]; 
then
    echo $file+' not exist'
    gdown https://drive.google.com/uc?id=16zAeGDmqbAvn7Iy8V-mBylKx6rG-wgLD
else
    echo $file+' exist'
fi

echo 'done'
exit 0
