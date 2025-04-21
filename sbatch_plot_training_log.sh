#!/bin/bash

datetime=$(date +%Y%m%d_%H%M%S)

#SBATCH --job-name=LLaVa-FT-datikz_plot_training_log
#SBATCH --gres=gpu:0
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=./log/sbatch_plot_result.txt
#SBATCH --error=./log/sbatch_plot_error.txt

singularity exec --nv ../singularity-sif/llava-ft-datikz_latest.sif python3 src/plot_training_log.py