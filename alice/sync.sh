#!/bin/bash
# Local helper.  push: code + weights + specs -> ~/reasoning-fractals on ALICE (and nothing else there).
#                pull NAME: results of out/NAME back to data/cluster/NAME.
set -euo pipefail
HOST=alice2
REMOTE=reasoning-fractals
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-}" in
  push)
    ssh "$HOST" "mkdir -p ~/$REMOTE/code ~/$REMOTE/data ~/$REMOTE/logs ~/$REMOTE/out ~/$REMOTE/alice"
    rsync -a --delete --exclude='__pycache__' "$LOCAL/render/" "$HOST:$REMOTE/code/render/"
    rsync -a --delete "$LOCAL/alice/" "$HOST:$REMOTE/alice/"
    rsync -a "$LOCAL/data/raw/eqr_sudoku_weights.npz" "$LOCAL/data/plane_seed0.npz" "$HOST:$REMOTE/data/"
    [ -f "$LOCAL/data/asset_samples.npz" ] && rsync -a "$LOCAL/data/asset_samples.npz" "$HOST:$REMOTE/data/" || true
    ;;
  pull)
    NAME="${2:?usage: sync.sh pull NAME}"
    mkdir -p "$LOCAL/data/cluster/$NAME"
    rsync -a "$HOST:$REMOTE/out/$NAME/" "$LOCAL/data/cluster/$NAME/"
    ;;
  *) echo "usage: $0 push | pull NAME"; exit 1;;
esac
