#!/bin/bash
# Local helper: how many tiles of out/NAME exist on the cluster, and what is in the queue.  usage: alice/status.sh NAME [TOTAL]
set -euo pipefail
NAME="${1:?usage: status.sh NAME [TOTAL]}"; TOTAL="${2:-?}"
ssh -o BatchMode=yes alice2 "n=\$(ls ~/reasoning-fractals/out/$NAME/tiles 2>/dev/null | grep -c '^t_[0-9]*_[0-9]*\.npz\$' || true); echo \"$NAME: \$n / $TOTAL tiles\"; squeue -h -u \$USER -o '%i %j %T %M %b' | grep rf- || true"
