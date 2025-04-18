#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 5G
#SBATCH --cpus-per-task 4
#SBATCH --time 5:30:00
#SBATCH --job-name pruning_thesis
#SBATCH --partition rivulet
#SBATCH --output run_out
source ~/.bashrc

echo "Running"
python model_run.py --model cnv_1w1a