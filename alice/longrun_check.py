#!/usr/bin/env python
"""Are 'wrong' endings really stuck, or just slow?  Re-run unsolved native starts of the puzzle scan for many more loops.

Takes the (puzzle, start) pairs that were NOT on the solution after 24 loops in out/pscan (same RNG, so the same
latents), runs them for LOOPS loops (default 96) and reports how many reach the solution and when, and how many
stay on a wrong grid at the end.  Run from ~/reasoning-fractals.
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
    ap.add_argument("--pscan", default="out/pscan")
    ap.add_argument("--out", default="out/longrun")
    ap.add_argument("--loops", type=int, default=96)
    ap.add_argument("--n", type=int, default=400, help="how many unsolved starts to re-run")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(a.pscan) / "pscan.json").read_text())
    K = meta["k"]
    z = np.load(Path(a.pscan) / "pscan.npz")
    solved = z["solved"]
    pz = json.loads(Path("data/puzzles_scan.json").read_text())
    ps, ks = np.nonzero(~solved)
    rng = np.random.default_rng(1)
    pick = rng.choice(len(ps), size=min(a.n, len(ps)), replace=False)
    ps, ks = ps[pick], ks[pick]
    eng = Engine(P.ASSET.question, backend="torchfn", device="cuda", precision="fp32")
    eng.steps = a.loops
    res = []
    t0 = time.time()
    # regenerate the exact latents: chunk c0 used default_rng([seed, c0]) -> zh then zl of shape (n*K, 97, 512)
    by_chunk = {}
    for p, k in zip(ps, ks):
        by_chunk.setdefault((int(p) // a.chunk) * a.chunk, []).append((int(p), int(k)))
    for c0, items in sorted(by_chunk.items()):
        n = min(len(pz) - c0, a.chunk)
        r = np.random.default_rng([a.seed, c0])
        zh = I.trunc_normal((n * K, 97, 512), 1.0, r)
        zl = I.trunc_normal((n * K, 97, 512), 1.0, r)
        rows = np.array([(p - c0) * K + k for p, k in items])
        tokens = np.stack([W.puzzle_tokens(pz[p]["q"]) for p, _ in items])
        sols = np.array([[int(c) for c in pz[p]["a"]] for p, _ in items], dtype=np.int8)
        tr, ret = eng.rollout_early(zh[rows], zl[rows], solution=sols, m=3, tokens=tokens)
        for (p, k), t, rt, s in zip(items, tr, ret, sols):
            fin = t[-1]
            ok = bool((fin == s).all())
            first = int(np.argmax((t == s).all(-1))) + 1 if ok else None
            changed_late = int((t[-12:] != t[-1]).any(-1).sum()) if not ok else 0
            res.append(dict(puzzle=p, rating=pz[p]["rating"], k=k, solved_by_end=ok, first_solved_loop=first,
                            n_wrong_end=int((fin != s).sum()), changing_in_last_12_loops=changed_late))
        print(f"chunk {c0}: {len(items)} starts  {time.time() - t0:.0f}s", flush=True)
    sb = np.array([r["solved_by_end"] for r in res])
    first = np.array([r["first_solved_loop"] for r in res if r["first_solved_loop"]])
    summary = dict(loops=a.loops, n=len(res), frac_solved_by_end=float(sb.mean()),
                   first_solved_percentiles={str(q): float(np.percentile(first, q)) for q in (10, 50, 90)} if len(first) else None,
                   still_wrong_and_changing=int(sum(1 for r in res if not r["solved_by_end"] and r["changing_in_last_12_loops"] > 0)),
                   still_wrong_and_frozen=int(sum(1 for r in res if not r["solved_by_end"] and r["changing_in_last_12_loops"] == 0)))
    (out / "longrun.json").write_text(json.dumps(dict(summary=summary, rows=res), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
