#!/bin/bash
#SBATCH --job-name=ELYZA-8b-LLama3-train-explanation
#SBATCH --partition=a6000
#SBATCH --gres=gpu:4
#SBATCH --time=72:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=./log/sbatch_train_result.txt
#SBATCH --error=./log/sbatch_train_error.txt

singularity exec --nv ../sif/test2.sif bash scripts/train_explanation.sh
