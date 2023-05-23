#!/bin/bash

#SBATCH --job-name make_some_noise
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time 1-0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y1
#SBATCH -o /data/worrospeed/logs/slurm-%A.out

echo "start"

python -u /data/worrospeed/mlp_compitition/lth/.seraph/make_residual.py

echo "end"

exit 0