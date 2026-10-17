#!/bin/bash
#SBATCH --job-name=rf-chase
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# usage: sbatch --export=ALL,CX=0.887,CY=0.870,HW0=0.1,LEVELS=4 alice/chase.slurm   (no commas inside a value)
set -euo pipefail
cd "$HOME/reasoning-fractals"
export PYTHONPATH="$PWD/code" RF_WEIGHTS_NPZ="$PWD/data/eqr_sudoku_weights.npz" PYTHONUNBUFFERED=1
: "${CX:?set CX}" "${CY:?set CY}"
env/bin/python alice/chase.py --out "${OUT:-out/chase}" --c0 "$CX,$CY" --hw0 "${HW0:-0.125}" --levels "${LEVELS:-4}"
