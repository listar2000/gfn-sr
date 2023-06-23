#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes

#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=15:20:00

#SBATCH --job-name=train_transformer
#SBATCH --output slurm/train_transformer.out

python benchmark.py