#!/bin/bash
#SBATCH --job-name=rf-chase
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# usage: sbatch --export=ALL,C0=0.31,0.42,HW0=0.125 alice/chase.slurm
set -euo pipefail
cd "$HOME/reasoning-fractals"
export PYTHONPATH="$PWD/code" RF_WEIGHTS_NPZ="$PWD/data/eqr_sudoku_weights.npz" PYTHONUNBUFFERED=1
: "${C0:?set C0=x,y}"
env/bin/python alice/chase.py --out "${OUT:-out/chase}" --c0 "$C0" --hw0 "${HW0:-0.125}" --levels "${LEVELS:-4}"
