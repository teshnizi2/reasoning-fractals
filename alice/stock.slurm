#!/bin/bash
#SBATCH --job-name=rf-stock
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%j.out
# The authors' own code path (loopscape stock solver) on the frozen asset pixels, a few numeric configs.
set -euo pipefail
cd "$HOME/reasoning-fractals"
export PYTHONPATH="$PWD/code" PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
env/bin/python alice/stock_check.py --out out/stock
