#!/bin/bash

datetime=$(date +%Y%m%d_%H%M%S)

#SBATCH --job-name=LLaVa-FT-datikz_eval
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:a6000_ada:1
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./log/sbatch_eval_result_%j_${datetime}.txt
#SBATCH --error=./log/sbatch_eval_error_%j_${datetime}.txt

singularity exec --nv ../singularity-sif/llava-ft-datikz_latest.sif bash scripts/evaluate_captioning.sh