#!/bin/bash
#SBATCH --job-name=rf-pscan
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# Multistability scan over ~2800 stratified Sudoku-Extreme puzzles x 16 native random starts.
set -euo pipefail
cd "$HOME/reasoning-fractals"
export PYTHONPATH="$PWD/code" RF_WEIGHTS_NPZ="$PWD/data/eqr_sudoku_weights.npz" PYTHONUNBUFFERED=1
env/bin/python alice/puzzle_scan.py --out out/pscan
