#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=qwen-baseline
#SBATCH --output=logs/qwen_baseline_%j.out
#SBATCH --error=logs/qwen_baseline_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40s:1

cd /data/users/tongf/master_thesis_tang

source /data/users/tongf/miniconda3/etc/profile.d/conda.sh
conda activate mt

python scripts/profile_qwen3_flashhead.py