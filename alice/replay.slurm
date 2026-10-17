#!/bin/bash
#SBATCH --job-name=rf-replay
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# usage (after alice/sync.sh push and rsync of data/replay_in.json): sbatch alice/replay.slurm
set -euo pipefail
cd "$HOME/reasoning-fractals"
export PYTHONPATH="$PWD/code" RF_WEIGHTS_NPZ="$PWD/data/eqr_sudoku_weights.npz" PYTHONUNBUFFERED=1
env/bin/python alice/replay.py --inp data/replay_in.json --out out/replay
