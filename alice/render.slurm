#!/bin/bash
#SBATCH --job-name=rf-render
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%A_%a.out
# Tiled, resumable render.  One array task = one worker = every WORKERS-th tile.  Re-submitting the same
# command continues where it stopped (finished tiles are skipped).  Example:
#   sbatch --array=0-7 --export=ALL,SPEC=alice/specs/hero.json,OUT=out/hero,WORKERS=8 alice/render.slurm
# MAX_MIN (default 215) stops a worker before the 4 h limit; every tile must finish well inside the margin.
set -euo pipefail
cd "$HOME/reasoning-fractals"
export PYTHONPATH="$PWD/code" RF_WEIGHTS_NPZ="$PWD/data/eqr_sudoku_weights.npz" PYTHONUNBUFFERED=1
: "${SPEC:?set SPEC}" "${OUT:?set OUT}" "${WORKERS:?set WORKERS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
env/bin/python -m render.tilerun run --spec "$SPEC" --out "$OUT" \
    --worker "${SLURM_ARRAY_TASK_ID:-0}" --workers "$WORKERS" --max-minutes "${MAX_MIN:-215}"
