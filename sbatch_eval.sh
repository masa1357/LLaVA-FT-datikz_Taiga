#!/bin/bash
#SBATCH --job-name=LLaVa-FT-datikz_eval
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:a6000_ada:1
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./log/sbatch_eval_after_result.txt
#SBATCH --error=./log/sbatch_eval_after_error.txt

singularity exec --nv ../singularity-sif/llava-ft-datikz_latest.sif bash scripts/evaluate_captioning.sh