# Autonomous Latent Reasoning

A small research / training codebase for baseline experiments in autonomous latent reasoning. This repository contains a lightweight training script, dataset JSONL files, and supporting data/model utilities under `src/`.

This README explains the repository layout and gives step-by-step setup instructions for both a local Windows (PowerShell) environment and a remote Linux server, including virtual environment creation, dependency installation, and how to run the baseline training script.

## Repository layout

- `train_baseline.py` — main training entrypoint used for baseline experiments.
- `data/` — dataset files (example: `train.jsonl`, `validation.jsonl`).
- `src/` — helper modules used by the training pipeline:
  - `dataset.py`
  - `model.py`
  - `prosqa_generator.py`

## Quick notes / contract

- Inputs: JSONL dataset files inside `data/` (e.g. `data/train.jsonl`, `data/validation.jsonl`).
- Outputs: model checkpoints and logs (wherever `train_baseline.py` writes them — check the script for exact args/path overrides).
- Errors: missing dependencies or wrong Python version; activation/permission issues on PowerShell.

## Prerequisites

- Python 3.8+ (3.10/3.11 recommended). Ensure `python` or `python3` points to that interpreter.
- Git (to clone the repo remotely)
- For GPU training: CUDA-compatible drivers + matching CUDA toolkit (follow PyTorch or other framework docs).

If you don't have a `requirements.txt` in the repo, create one from your working environment with `pip freeze > requirements.txt` (instructions below).

## 1) Local setup (Linux / macOS)

These instructions focus on Linux / macOS shells (bash, zsh). They work for remote servers, cloud VMs, and local Linux development machines.

Recommended: create an isolated virtual environment using Python's builtin `venv` or use `conda` if you prefer that workflow.

Open a terminal and run:

```bash
# 1. Clone the repo (if you haven't already):
git clone <repo-url>
cd autonomous-latent-reasoning

# 2. Create a virtual environment (named "venv"):
python3 -m venv venv

# 3. Activate the virtual environment:
source venv/bin/activate

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
# If the repository includes a requirements.txt (recommended):
pip install -r requirements.txt
# Example (if no requirements.txt):
pip install torch transformers datasets tqdm

# 5. Run the training script (example):
python train_baseline.py
```

Notes:
- Use `python3` if `python` on your system points to Python 2.x. On many systems `python3` is the proper command.
- If you need GPU support for PyTorch, install the correct build for your CUDA version following the PyTorch install guide (for example, use the `--index-url` provided by PyTorch wheel host).
- To exit the venv run `deactivate`.

## 2) Remote setup (Linux server / cloud VM)

This section assumes you have SSH access to a Linux server (Ubuntu/Debian recommended). For GPU machines, use a CUDA-enabled image or install drivers following your cloud provider's docs.

```bash
# 1. Clone the repo on the remote machine:
git clone <repo-url>
cd autonomous-latent-reasoning

# 2. Create and activate a venv:
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install dependencies:
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run training (background example):
python train_baseline.py &> train.log &
```

Tips:
- If you will run long jobs, consider using `tmux` or `screen` or a job manager (SLURM) on HPC.
- For reproducibility and dependency management on remote machines, consider using `conda` for easier CUDA-enabled package installs.

## Creating / managing the Python dependencies

- If the repository does not ship a `requirements.txt`, create one from an environment where everything is installed and tested:

```bash
# after pip installing packages you need locally
pip freeze > requirements.txt
```

- Example minimal `requirements.txt` (adjust to your project):

```
torch
transformers
datasets
tqdm
sentencepiece
```

Install with:

```bash
pip install -r requirements.txt
```

## How to run

- By default run the main training script: `python train_baseline.py`.
- Check `train_baseline.py` for available CLI flags (data paths, logging directory, batch sizes, learning rate, number of epochs). Example pattern:

```powershell
python train_baseline.py --train data/train.jsonl --validation data/validation.jsonl --output-dir outputs/run1
```

Adjust the exact argument names to match the script — open `train_baseline.py` to see its argparse or config usage.

## Troubleshooting

- Activation errors on PowerShell: use `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
- Missing C/C++ build tools while installing some libraries (rare): install `Build Tools for Visual Studio` on Windows or `build-essential` on Debian/Ubuntu.
- CUDA / GPU issues: ensure GPU drivers and CUDA toolkit match the installed PyTorch build. If unsure, use CPU-only PyTorch build.

## Development notes

- The `src/` folder contains the modules used by `train_baseline.py`. Edit them when adding new features or implementing changes to the model or dataset.
- Add unit tests and CI if you add public APIs. Keep `requirements.txt` up to date.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Run and verify code in an isolated venv.
4. Open a pull request with clear description and motivation.

## Contact / License

If this repo doesn't contain a `LICENSE` file, check with the owner for reuse permissions. For questions, open an issue on the repository.

