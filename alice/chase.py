#!/usr/bin/env python
"""Zoom chase: from an initial centre, render a small window, pick the richest sub-window, zoom in, repeat.

Each level renders an N x N window (fp32, early exit) whose half-width is ``zoom`` times smaller than the
previous one and centred on the previous level's pick.  Everything is saved, plus the choices and the
per-level boundary statistics, so the final keyframe centre is documented.  Run from ~/reasoning-fractals.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "code"))
from render import chase as CH  # noqa: E402
from render import puzzles as P  # noqa: E402
from render import runner as RU  # noqa: E402
from render import slices as SL  # noqa: E402
from render import sudoku as S  # noqa: E402
from render import tilerun as TR  # noqa: E402
from render.engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/chase")
    ap.add_argument("--c0", required=True, help="x,y of the first zoom centre (plane units)")
    ap.add_argument("--hw0", type=float, default=0.125, help="half-width of the first window")
    ap.add_argument("--zoom", type=float, default=8.0)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--puzzle", default=P.ASSET.question)
    ap.add_argument("--early-m", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sol = S.pack([S.solve(a.puzzle)])[0].astype(np.int8)
    plane = SL.get_plane(0, "data/plane_seed0.npz")
    eng = Engine(a.puzzle, backend="torchfn", device="cuda", precision="fp32")
    cx, cy = (float(v) for v in a.c0.split(","))
    hw = a.hw0
    log = []
    t0 = time.time()
    for lvl in range(1, a.levels + 1):
        spec = dict(mode="window", center=[cx, cy], half_width=hw, n=a.n, tile=a.n, batch=2048)
        coords = TR.pixel_coords(spec, TR.plan_tiles(a.n, a.n)[0])
        res = RU.run_pixels(eng, coords, plane, a.puzzle, sol, batch=2048, precision="float64", early_m=a.early_m)
        settle = res["settle_nb"].reshape(a.n, a.n)
        flags = (res["solved"].astype(np.uint8) | (res["valid"].astype(np.uint8) << 1) | (res["censored"].astype(np.uint8) << 2)).reshape(a.n, a.n)
        win = max(8, int(round(a.n / a.zoom)))
        row, col, score = CH.pick_centre(settle, win=win, step=max(2, win // 6))
        nx_, ny_ = CH.pixel_to_plane(row, col, (cx, cy), hw, a.n)
        sc = CH.window_scores(settle, win=win, step=max(2, win // 6))
        entry = dict(level=lvl, center=[cx, cy], half_width=hw, zoom=1.0 / hw, n=a.n, seconds=res["seconds"],
                     mean_loops=float(res["retired_at"].mean()), frac_solved=float(res["solved"].mean()),
                     frac_censored=float(res["censored"].mean()), n_levels=int(len(np.unique(settle))),
                     boundary_fraction=float(CH.window_scores(settle, win=a.n, step=a.n)["boundary"].max()),
                     settle_hist=np.bincount(settle.ravel(), minlength=25).tolist(),
                     pick=dict(row=row, col=col, score=score, x=nx_, y=ny_), best_window_boundary=float(sc["boundary"].ravel()[np.argmax(sc["score"])]))
        np.savez_compressed(out / f"level{lvl}.npz", settle_nb=settle, flags=flags, n_wrong=res["n_wrong"].reshape(a.n, a.n))
        log.append(entry)
        (out / "chase.json").write_text(json.dumps(dict(args=vars(a), levels=log), indent=1))
        print(f"level {lvl}: zoom {1 / hw:.0f}x  boundary {entry['boundary_fraction']:.2f}  levels {entry['n_levels']}  "
              f"pick ({nx_:+.8f},{ny_:+.8f}) score {score:.2f}  {res['seconds']:.0f}s  loops {entry['mean_loops']:.1f}", flush=True)
        cx, cy, hw = nx_, ny_, hw / a.zoom
    print("final centre", cx, cy, "next half-width", hw, "total s", time.time() - t0)


if __name__ == "__main__":
    main()
