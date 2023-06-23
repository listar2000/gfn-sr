#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes

#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=15:20:00

#SBATCH --job-name=train_lstm
#SBATCH --output slurm/train_lstm.out

python benchmark.py