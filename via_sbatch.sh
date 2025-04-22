#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 5G
#SBATCH --cpus-per-task 4
#SBATCH --time 10:30:00
#SBATCH --job-name pruning_thesis
#SBATCH --partition rivulet
#SBATCH --output runs_slurm_log/run_out_%x_%A
source ~/.bashrc

echo "Running"
python wrapper.py --model cnv_1w1a --iterative