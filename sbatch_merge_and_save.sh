#!/bin/bash
#SBATCH --job-name=LLaVa-FT-datikz_merge_and_save
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:a6000_ada:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./log/sbatch_merge_and_save_result.txt
#SBATCH --error=./log/sbatch_merge_and_save_error.txt


singularity exec --nv ../singularity-sif/llava-ft-datikz_latest.sif bash scripts/merge_and_save.sh