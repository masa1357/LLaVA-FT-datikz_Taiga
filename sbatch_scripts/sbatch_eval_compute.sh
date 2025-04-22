#!/bin/bash
#SBATCH --job-name=LLaVa-FT-datikz_eval
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:0
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./log/sbatch_eval_compute_base_result.txt
#SBATCH --error=./log/sbatch_eval_compute_base_error.txt


singularity exec --nv ../singularity-sif/llava-ft-datikz_latest.sif bash scripts/evaluate_captioning.sh