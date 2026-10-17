#!/usr/bin/env python
"""The loopscape README quickstart, verbatim, on the CPU: 64 random initial latents, EqR, 24 loops, noise off.

Writes results/quickstart.json (device, wall time, how many of the 64 solved, steps).  The puzzle is the README's own
(Sudoku-Extreme test puzzle, rating 162).
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOOPSCAPE_CACHE_DIR", str(ROOT / "data/raw/loopscape_cache"))
os.environ.setdefault("LOOPSCAPE_EQR_SRC", str(ROOT / "vendor/EqR"))

import torch  # noqa: E402
from loopscape import get_solver  # noqa: E402

puzzle = ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2...."
solver = get_solver("eqr")                         # CPU, float32
torch.manual_seed(0)
z = torch.randn(64, 97, 512)                       # 64 initial conditions
t0 = time.time()
r = solver.solve_batch(puzzle, z_H=z, z_L=z * 0, max_steps=24, noise_scale=0.0, return_intermediates=True)
dt = time.time() - t0
out = dict(script="scripts/quickstart.py", device="cpu (Apple M4)", torch=torch.__version__, n_initial_conditions=64,
           n_solved=int(sum(r.solved)), steps=r.steps, halted=bool(r.halted), seconds=dt, rollouts_per_s=64 / dt,
           puzzle=puzzle, note="loopscape README quickstart; z_H = randn(64,97,512), z_L = 0, max_steps=24, noise_scale=0")
(ROOT / "results/quickstart.json").write_text(json.dumps(out, indent=1))
print(out)
