#!/usr/bin/env python
"""Freeze N random pixels of the upstream asset (plane-axis indices + ordinal level) for the cluster check.

The cluster never sees the upstream image itself, only these sampled levels.  The orientation hypothesis
(`--variant`, found by scripts/repro_probe.py) decides which plane coordinates each image pixel corresponds to.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import repro as R  # noqa: E402

ASSET = ROOT / "vendor/loopscape/assets/basins_1000_0_eqr_sudoku_plane_rating47_7d68927f389fbdaa_rating47.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=R.VARIANTS)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    levels, _ = R.load_asset_levels(ASSET)
    n = levels.shape[0]
    rng = np.random.default_rng(a.seed)
    flat = rng.choice(n * n, size=a.n, replace=False)
    r, c = np.divmod(flat, n)
    xi, yi = R.variant_axes(a.variant, r, c, n)
    nbr = R.neighbour_disagreement(levels, r, c)
    np.savez(ROOT / "data/asset_samples.npz", xi=xi.astype(np.int64), yi=yi.astype(np.int64),
             levels=levels[r, c], rows=r, cols=c, nbr_diff=nbr, variant=a.variant, n_asset=n)
    print("saved", a.n, "samples; level histogram", np.bincount(levels[r, c], minlength=11).tolist(),
          "| interior (no differing neighbour):", float((nbr == 0).mean()))


if __name__ == "__main__":
    main()
