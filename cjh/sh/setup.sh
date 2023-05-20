#!/bin/bash
env=mlp

echo "activate torch environment"
{
    source /data/joohyun7u/anaconda3/etc/profile.d/conda.sh &&
    source activate $env &&
    echo "$env Activated Successfully"
} ||
{
    echo "$env Activate Failed"
}