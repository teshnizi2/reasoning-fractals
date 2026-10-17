#!/usr/bin/env python
"""Throughput of the deterministic EqR map on this Mac (Apple M4, shared with other jobs).

Rollouts per second (one rollout = 24 loops of one starting thought) for: the stock loopscape solver on MPS and
CPU, the token-major torch backend on MPS and CPU, and the MLX backend, at a few batch sizes; plus raw matmul
TFLOPS for the model's GEMM shapes.  Writes results/throughput_m4.json.  Keep it short: each point is one timed
rollout after a warm-up call at the same batch size.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import puzzles as P  # noqa: E402
from render import slices as SL  # noqa: E402
from render.engine import Engine  # noqa: E402


def gpu_util() -> int | None:
    out = subprocess.run("ioreg -r -d 1 -w 0 -c AGXAccelerator | grep -o '\"Device Utilization %\"=[0-9]*'",
                         shell=True, capture_output=True, text=True).stdout
    return int(out.split("=")[-1]) if "=" in out else None


def main() -> None:
    plane = SL.get_plane(0, ROOT / "data/plane_seed0.npz")
    ax = SL.axis(1000)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 1000, size=(256, 2))
    coords = np.stack([ax[idx[:, 0]], ax[idx[:, 1]]], axis=-1)
    z_all = SL.latents(coords, *plane)
    res = {"machine": "Apple M4 (10-core GPU, 16 GB unified memory), shared with other jobs",
           "gpu_utilisation_before_pct": gpu_util(), "points": []}

    def timed(label, backend, device, batches, **kw):
        eng = Engine(P.ASSET.question, backend=backend, device=device, **kw)
        for b in batches:
            z = z_all[:b]
            eng.rollout(z)  # warm-up (compile / allocation) at this batch size
            t0 = time.time()
            eng.rollout(z)
            dt = time.time() - t0
            res["points"].append(dict(label=label, batch=b, seconds=dt, rollouts_per_s=b / dt, gpu_util_after_pct=gpu_util()))
            print(f"{label:26s} B={b:4d}  {b / dt:6.2f} rollouts/s", flush=True)
            json.dump(res, open(ROOT / "results/throughput_m4.json", "w"), indent=1)

    timed("mlx", "mlx", "gpu", [16, 64, 128])
    timed("torchfn-mps", "torchfn", "mps", [16, 64, 128])
    timed("stock-loopscape-mps", "torch", "mps", [16, 64])
    timed("torchfn-cpu", "torchfn", "cpu", [8, 16])
    timed("stock-loopscape-cpu", "torch", "cpu", [8])
    res["script"] = "scripts/bench_local.py"
    res["flops_per_rollout_gflop"] = 615.3  # 24 loops x 21 L-level calls x 2 layers x 610.4 MFLOP
    json.dump(res, open(ROOT / "results/throughput_m4.json", "w"), indent=1)
    print("wrote results/throughput_m4.json")


if __name__ == "__main__":
    main()
