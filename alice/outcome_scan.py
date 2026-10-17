#!/usr/bin/env python
"""Does a slice around a random base point show more than one outcome?

For each puzzle, base-point seed, z_L mode and half-width h, render a small window
z(a, b) = z0 + a*u + b*v, (a, b) in [-h, h]^2 (z0 = EqR's own truncated-normal sample), and record how many pixels
end on the true solution, how many on other grids, and how many are still changing at loop 24.  Run from
~/reasoning-fractals.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "code"))
from render import init as I  # noqa: E402
from render import puzzles as P  # noqa: E402
from render import runner as RU  # noqa: E402
from render import slices as SL  # noqa: E402
from render import sudoku as S  # noqa: E402
from render.engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/oscan")
    ap.add_argument("--puzzles", default="asset", help="comma list of: asset, notebook, idx:N (index into data/puzzles_scan.json)")
    ap.add_argument("--pscan", default=None, help="out/pscan/pscan.json: also take the --pscan-top most multistable puzzles")
    ap.add_argument("--pscan-top", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--hs", default="10,30,100,300")
    ap.add_argument("--zl", default="zero,base")
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pz = json.loads(Path("data/puzzles_scan.json").read_text())
    todo = []
    for tok in a.puzzles.split(","):
        if tok == "asset":
            todo.append(("asset", P.ASSET.question))
        elif tok == "notebook":
            todo.append(("notebook", P.NOTEBOOK.question))
        elif tok.startswith("idx:"):
            i = int(tok[4:])
            todo.append((f"idx{i}", pz[i]["q"]))
    if a.pscan and a.pscan_top:
        ps = json.loads(Path(a.pscan).read_text())["puzzles"]
        ps = [p for p in ps if p["n_distinct_final"] >= 2 and 0.1 <= p["frac_solved"] <= 0.9]
        ps.sort(key=lambda p: abs(p["frac_solved"] - 0.5))
        for p in ps[: a.pscan_top]:
            todo.append((f"idx{p['i']}", p["q"]))
    plane = SL.get_plane(0, "data/plane_seed0.npz")
    rows = []
    t0 = time.time()
    for name, q in todo:
        sol = S.pack([S.solve(q)])[0].astype(np.int8)
        eng = Engine(q, backend="torchfn", device="cuda", precision="fp32")
        for seed in (int(s) for s in a.seeds.split(",")):
            rng = np.random.default_rng([777, seed])
            z0h, z0l = I.trunc_normal((97, 512), 1.0, rng), I.trunc_normal((97, 512), 1.0, rng)
            for zl in a.zl.split(","):
                base = (z0h, z0l if zl == "base" else np.zeros_like(z0l))
                for h in (float(x) for x in a.hs.split(",")):
                    coords = SL.Window(center=(0.0, 0.0), half_width=h, n=a.n).coords()
                    r = RU.run_pixels(eng, coords, plane, q, sol, batch=a.n * a.n, precision="float64", early_m=3, base=base)
                    uniq, cnt = np.unique(r["final"], axis=0, return_counts=True)
                    row = dict(puzzle=name, seed=seed, zl=zl, h=h, seconds=r["seconds"], frac_solved=float(r["solved"].mean()),
                               frac_censored=float(r["censored"].mean()), n_distinct_final=int(len(uniq)),
                               top_wrong_counts=sorted([int(c) for u, c in zip(uniq, cnt) if not (u == sol).all()], reverse=True)[:5],
                               mean_settle=float(r["settle_nb"].mean()), mean_loops=float(r["retired_at"].mean()))
                    rows.append(row)
                    np.savez_compressed(out / f"{name}_s{seed}_{zl}_h{h:g}.npz", settle_nb=r["settle_nb"].reshape(a.n, a.n),
                                        solved=r["solved"].reshape(a.n, a.n), censored=r["censored"].reshape(a.n, a.n),
                                        n_wrong=r["n_wrong"].reshape(a.n, a.n), final=r["final"])
                    print(row, flush=True)
                    (out / "oscan.json").write_text(json.dumps(rows, indent=1))
    print("done", time.time() - t0, "s")


if __name__ == "__main__":
    main()
