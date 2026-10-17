#!/usr/bin/env python
"""Does the authors' own code path reproduce the asset better than our fast engine?

Runs loopscape's stock EqR solver (notebook recipe: z_H = a*u + b*v, z_L = 0, 24 loops, noise 0, batches of
128 through ``batched_apply``) on the frozen asset pixels, in a few numeric configurations, and compares the
settle times with the asset's levels and with our fast engine's fp32 result.  Run from ~/reasoning-fractals.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
os.environ["LOOPSCAPE_CACHE_DIR"] = str(ROOT / "data/loopscape_cache")
os.environ["LOOPSCAPE_EQR_SRC"] = str(ROOT / "vendor/EqR")
sys.path[:0] = [str(ROOT / "vendor/loopscape"), str(ROOT / "code")]

import torch  # noqa: E402
from loopscape import get_solver  # noqa: E402
from loopscape.utils import batched_apply  # noqa: E402

from render import repro as R  # noqa: E402
from render import slices as SL  # noqa: E402

PUZZLE = "2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3..."


def to_traces(result):  # notebook cell 1, verbatim
    return torch.swapaxes(torch.tensor(np.asarray(result.intermediates), dtype=torch.int8), 0, 1)


def convergence_times(traces):  # notebook cell 1, verbatim
    conv = (traces == traces[:, -1:]).all(-1).all(-1)
    return conv.shape[1] - conv.sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/stock")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--configs", default="fp32:128,tf32:128,fp32:32")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load("data/asset_samples.npz")
    xi, yi, levels, nbr = z["xi"][: a.n], z["yi"][: a.n], z["levels"][: a.n], z["nbr_diff"][: a.n]
    ax = SL.axis(1000)
    pts = torch.tensor(np.stack([ax[xi], ax[yi]], axis=-1))
    u, v = (torch.tensor(t) for t in SL.get_plane(0, "data/plane_seed0.npz"))
    ours = np.load("out/smoke/sample_fp32.npz")["settle_nb"][: a.n] if Path("out/smoke/sample_fp32.npz").exists() else None
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n": int(len(pts))}
    for cfg in a.configs.split(","):
        prec, b = cfg.split(":")
        b = int(b)
        torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
        solver = get_solver("eqr", device="cuda")

        def f(p):
            zz = p[:, 0, None, None] * u + p[:, 1, None, None] * v
            r = solver.solve_batch(PUZZLE, z_H=zz, z_L=zz * 0, max_steps=24, noise_scale=0.0, return_intermediates=True)
            return to_traces(r)

        t0 = time.time()
        traces = batched_apply(f, pts, batch_size=b, pad_dim=1)
        T = convergence_times(traces).numpy().astype(np.int64)
        dt = time.time() - t0
        shift, frac = R.best_shift_agreement(levels, T, clip_top=10)
        clipped = np.clip(T - shift, 0, 10)
        row = dict(seconds=dt, rollouts_per_s=len(pts) / dt, asset_best_shift=shift, asset_agreement=frac,
                   asset_spearman=R.spearman(levels, T),
                   asset_agreement_interior=float((levels == clipped)[nbr == 0].mean()),
                   asset_agreement_boundary=float((levels == clipped)[nbr > 0].mean()),
                   settle_hist=np.bincount(T, minlength=25).tolist())
        if ours is not None:
            row["vs_our_fast_fp32_settle_equal"] = float((T == ours).mean())
        res[cfg] = row
        np.savez_compressed(out / f"stock_{prec}_B{b}.npz", settle_nb=T)
        print(cfg, row, flush=True)
        (out / "stock.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
