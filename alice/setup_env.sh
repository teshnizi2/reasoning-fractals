#!/bin/bash
# Run ON THE ALICE LOGIN NODE (it has internet; compute nodes may not).  Creates ~/reasoning-fractals/env.
# Only touches ~/reasoning-fractals.
set -euo pipefail
mkdir -p "$HOME/reasoning-fractals" && cd "$HOME/reasoning-fractals"
module load Miniconda3/24.7.1-0
[ -x env/bin/python ] || conda create -y -p ./env python=3.12 -c conda-forge --override-channels
env/bin/python -m pip install --upgrade pip
env/bin/python -m pip install torch numpy scipy pillow
env/bin/python - <<'PY'
import torch, numpy, scipy, PIL
print("torch", torch.__version__, "cuda build", torch.version.cuda, "numpy", numpy.__version__)
PY
