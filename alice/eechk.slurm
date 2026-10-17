#!/bin/bash
#SBATCH --job-name=rf-eechk
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$HOME/reasoning-fractals"
export PYTHONPATH="$PWD/code" RF_WEIGHTS_NPZ="$PWD/data/eqr_sudoku_weights.npz" PYTHONUNBUFFERED=1
env/bin/python alice/earlyexit_check.py
