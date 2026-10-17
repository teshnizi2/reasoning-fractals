#!/bin/bash
#SBATCH --job-name=rf-extra
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# Two short follow-ups to the puzzle scan, one allocation (the queue is the bottleneck, not the GPU):
#   1. longrun_check.py: are the unsolved native starts stuck or just slow?  (96 loops instead of 24)
#   2. outcome_scan.py : outcome maps of 2-D slices around native-scale base points of the asset puzzle and the
#                        three most multistable puzzles of the scan (h = half-width of the window in the plane)
set -euo pipefail
cd "$HOME/reasoning-fractals"
export PYTHONPATH="$PWD/code" RF_WEIGHTS_NPZ="$PWD/data/eqr_sudoku_weights.npz" PYTHONUNBUFFERED=1
echo "== longrun $(date +%H:%M:%S)"
env/bin/python alice/longrun_check.py --loops 96 --n 400
echo "== oscan $(date +%H:%M:%S)"
env/bin/python alice/outcome_scan.py --out out/oscan --puzzles asset --pscan out/pscan/pscan.json --pscan-top 3 \
    --seeds 0,1 --zl base --hs 10,30,100,300
echo "== done $(date +%H:%M:%S)"
