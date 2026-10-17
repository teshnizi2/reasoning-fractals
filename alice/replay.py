#!/usr/bin/env python
"""Replay chosen pixels on the render engine (torch fp32 on CUDA): plain run vs stored values, and nudged runs.

Input  data/replay_in.json  (written by scripts/plan_replay.py): puzzle, eps, nudged, seed, items[{id, x, y, prec, stored}]
Output out/replay/replay_out.json: per item settle/flags/trace/nudged/equal, plus the engine description.
Run from ~/reasoning-fractals.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "code"))
from render import replay as RP  # noqa: E402
from render import slices as SL  # noqa: E402
from render import sudoku as S  # noqa: E402
from render.engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/replay_in.json")
    ap.add_argument("--out", default="out/replay")
    ap.add_argument("--batch", type=int, default=1024)
    a = ap.parse_args()
    spec = json.loads(Path(a.inp).read_text())
    sol = S.pack([S.solve(spec["puzzle"])])[0].astype(np.int8)
    plane = SL.get_plane(0, "data/plane_seed0.npz")
    eng = Engine(spec["puzzle"], backend="torchfn", device="cuda", precision="fp32")
    t0 = time.time()
    res = RP.run_items(eng.rollout, plane, spec["puzzle"], sol, spec["items"], nudged=spec["nudged"], eps=spec["eps"],
                       seed=spec["seed"], batch=a.batch)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(engine=eng._b.name, items=len(res), nudged=spec["nudged"], eps=spec["eps"], seconds=time.time() - t0,
                equal=int(sum(r["equal"] for r in res)))
    (out / "replay_out.json").write_text(json.dumps(dict(meta=meta, items=res)))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
