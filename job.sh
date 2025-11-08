#!/bin/bash
# Simple SBATCH job script for running the baseline training on the cluster
# Requests 1 GPU, a few CPUs, and an appropriate memory allocation.
# Adjust paths (repo location, venv) if you keep the repo in a different place.

# SBATCH directives
# Use a short, descriptive job name
# (Max runtime limited by cluster policy; set <= 24:00:00)
# Request one GPU (important: cluster allows max 1 GPU per user/job)
# Request up to 40 CPUs if you need; here we request 4 as a reasonable default
# Memory should follow the cluster rule: up to 4GB per CPU thread requested
# (e.g., 4 CPUs => 16G). Adjust as needed.
# The output and error files are placed into the `logs/` directory.

#SBATCH --job-name=baseline_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "Job starting on $(hostname) at $(date)"

# Optional: load modules if your cluster uses Environment Modules
# Uncomment or adjust if needed, e.g.:
# module purge
# module load python/3.10 cuda/12.1

# Activate a Python virtual environment. Update the path if your venv is elsewhere.
if [ -d "$HOME/venv" ]; then
    source "$HOME/venv/bin/activate"
else
    echo "Warning: virtualenv not found at $HOME/venv. Continuing without venv."
fi

# Move to repository root (assumes repository is cloned to $HOME/autonomous-latent-reasoning)
cd "$HOME/autonomous-latent-reasoning" || { echo "Repo directory not found"; exit 1; }

# Ensure logs and outputs directories exist
mkdir -p logs saved_models

echo "Using Python: $(which python) -- $(python --version)"

# Sanity check: show GPUs visible to the job
echo "NVIDIA SMI output:" && nvidia-smi || true

# Run the training script (unbuffered output)
python -u train_baseline.py

echo "Job finished at $(date)"
