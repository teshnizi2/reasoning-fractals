#!/bin/bash
#SBATCH --job-name=rf-oscan
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# usage: sbatch [--gres=gpu:4g.40gb:1] --export=ALL,ARGS="--puzzles asset,notebook --pscan out/pscan/pscan.json --pscan-top 4" alice/oscan.slurm
set -euo pipefail
cd "$HOME/reasoning-fractals"
export PYTHONPATH="$PWD/code" RF_WEIGHTS_NPZ="$PWD/data/eqr_sudoku_weights.npz" PYTHONUNBUFFERED=1
env/bin/python alice/outcome_scan.py --out "${OUT:-out/oscan}" ${ARGS:-}
