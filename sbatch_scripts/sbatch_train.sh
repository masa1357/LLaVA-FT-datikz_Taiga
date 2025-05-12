#!/bin/bash
#SBATCH --job-name=ELYZA-8b-LLama3-train
#SBATCH --partition=a6000
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./log/sbatch_train_result.txt
#SBATCH --error=./log/sbatch_train_error.txt


#--clearenvでCUDAドライバの整合性をとってる -> あとで修正
singularity exec --nv ../sif/test.sif bash scripts/train.sh
