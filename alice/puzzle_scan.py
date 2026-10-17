#!/usr/bin/env python
"""Multistability scan: for many puzzles, run EqR (24 loops, noise 0) from K of its OWN random initial latents
(z_H and z_L both truncated-normal, std 1, exactly what EqR resets to) and record how often it lands on the true
solution, how many different grids it ends on, and how long it takes.  Puzzles where the model is genuinely torn
between answers are the ones whose basin map can show more than one outcome.  Run from ~/reasoning-fractals.
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
from render import weights as W  # noqa: E402
from render.engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", default="data/puzzles_scan.json")
    ap.add_argument("--out", default="out/pscan")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64, help="puzzles per batch (rows = chunk * k)")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pz = json.loads(Path(a.puzzles).read_text())
    eng = Engine(P.ASSET.question, backend="torchfn", device="cuda", precision="fp32")
    eng.steps = a.steps
    tokens = np.stack([W.puzzle_tokens(p["q"]) for p in pz])
    sols = np.array([[int(c) for c in p["a"]] for p in pz], dtype=np.int8)
    N, K = len(pz), a.k
    solved = np.zeros((N, K), dtype=bool)
    settle = np.zeros((N, K), dtype=np.uint8)
    nwrong = np.zeros((N, K), dtype=np.uint8)
    finals = {}
    t0 = time.time()
    for c0 in range(0, N, a.chunk):
        c1 = min(N, c0 + a.chunk)
        n = c1 - c0
        rng = np.random.default_rng([a.seed, c0])
        zh = I.trunc_normal((n * K, 97, 512), 1.0, rng)
        zl = I.trunc_normal((n * K, 97, 512), 1.0, rng)
        tok = np.repeat(tokens[c0:c1], K, axis=0)
        sol = np.repeat(sols[c0:c1], K, axis=0)
        tr, ret = eng.rollout_early(zh, zl, solution=sol, m=3, tokens=tok)
        fin = tr[:, -1]
        eq = (tr == fin[:, None, :]).all(-1)
        s = (a.steps - eq.sum(1)).astype(np.uint8)
        ok = (fin == sol).all(-1)
        nw = (fin != sol).sum(-1).astype(np.uint8)
        solved[c0:c1] = ok.reshape(n, K)
        settle[c0:c1] = s.reshape(n, K)
        nwrong[c0:c1] = nw.reshape(n, K)
        for i in range(n):
            f = fin[i * K:(i + 1) * K]
            finals[c0 + i] = [("".join(str(int(v)) if 0 <= v <= 9 else "?" for v in g)) for g in np.unique(f, axis=0)]
        print(f"{c1}/{N} puzzles  {time.time() - t0:.0f}s  frac solved so far {solved[:c1].mean():.4f}", flush=True)
    per = []
    for i, p in enumerate(pz):
        per.append(dict(i=i, rating=p["rating"], q=p["q"], frac_solved=float(solved[i].mean()),
                        n_distinct_final=len(finals[i]), mean_settle=float(settle[i].mean()),
                        max_settle=int(settle[i].max()), median_nwrong_unsolved=(float(np.median(nwrong[i][~solved[i]])) if (~solved[i]).any() else None)))
    (out / "pscan.json").write_text(json.dumps(dict(k=K, steps=a.steps, seconds=time.time() - t0, puzzles=per)))
    np.savez_compressed(out / "pscan.npz", solved=solved, settle=settle, nwrong=nwrong)
    multi = [p for p in per if 0.15 <= p["frac_solved"] <= 0.85]
    print("overall frac solved", float(solved.mean()), "| puzzles with 0.15<=solved<=0.85:", len(multi), "| fully solved:", sum(p["frac_solved"] == 1 for p in per), "of", N)
    for p in sorted(multi, key=lambda p: abs(p["frac_solved"] - 0.5))[:15]:
        print(p["i"], "rating", p["rating"], "solved", round(p["frac_solved"], 2), "distinct", p["n_distinct_final"], "mean settle", round(p["mean_settle"], 1))


if __name__ == "__main__":
    main()
