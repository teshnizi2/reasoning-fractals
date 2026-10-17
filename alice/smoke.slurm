#!/bin/bash
#SBATCH --job-name=rf-smoke
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%j.out
# Smoke + benchmark job: ~15-30 min of one A100.  Submit from ~/reasoning-fractals:  sbatch alice/smoke.slurm
set -euo pipefail
cd "$HOME/reasoning-fractals"
export PYTHONPATH="$PWD/code" RF_WEIGHTS_NPZ="$PWD/data/eqr_sudoku_weights.npz" PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
env/bin/python alice/bench_cuda.py --out out/smoke
