#!/bin/bash
#SBATCH --job-name=generate_dpo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=dpo_generation.log

echo "Starting DPO Data Generation..."
python train_star_warm.py
echo "Finished!"
