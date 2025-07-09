#!/bin/bash
#SBATCH --job-name=Evalution-Captioning-Task
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:4
#SBATCH --time=72:30:00
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./log/sbatch_eval_result.txt
#SBATCH --error=./log/sbatch_eval_error.txt


singularity exec --nv ../sif/test2.sif bash scripts/evaluate_captioning.sh