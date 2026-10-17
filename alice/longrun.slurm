#!/bin/bash
#SBATCH --job-name=rf-long
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$HOME/reasoning-fractals"
export PYTHONPATH="$PWD/code" RF_WEIGHTS_NPZ="$PWD/data/eqr_sudoku_weights.npz" PYTHONUNBUFFERED=1
env/bin/python alice/longrun_check.py --loops "${LOOPS:-96}" --n "${N:-400}"
