#!/usr/bin/env python
"""Does early exit change the picture beyond what a different batch size already does?

Same 4096 frozen asset pixels, fp32, several ways: full 24 loops at batch 1024 (twice, to test determinism),
full at batch 1000 (different GEMM shapes), early exit (3 loops on the solution) at batch 1024 and 2048.
Reports pairwise agreement of settle time and final grid, and the loops actually computed.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "code"))
from render import puzzles as P  # noqa: E402
from render import repro as R  # noqa: E402
from render import runner as RU  # noqa: E402
from render import slices as SL  # noqa: E402
from render import sudoku as S  # noqa: E402
from render.engine import Engine  # noqa: E402

out = Path("out/eechk")
out.mkdir(parents=True, exist_ok=True)
z = np.load("data/asset_samples.npz")
ax = SL.axis(1000)
coords = np.stack([ax[z["xi"]], ax[z["yi"]]], axis=-1)
levels, nbr = z["levels"], z["nbr_diff"]
plane = SL.get_plane(0, "data/plane_seed0.npz")
sol = S.pack([P.ASSET.answer])[0].astype(np.int8)
eng = Engine(P.ASSET.question, backend="torchfn", device="cuda", precision="fp32")
runs, res = {}, {}
for name, kw in [("full_B1024_a", dict(batch=1024)), ("full_B1024_b", dict(batch=1024)), ("full_B1000", dict(batch=1000)),
                 ("early_B1024", dict(batch=1024, early_m=3)), ("early_B2048", dict(batch=2048, early_m=3))]:
    t0 = time.time()
    r = RU.run_pixels(eng, coords, plane, P.ASSET.question, sol, **kw)
    runs[name] = r
    sh, ag = R.best_shift_agreement(levels, r["settle_nb"], clip_top=10)
    res[name] = dict(seconds=time.time() - t0, rollouts_per_s=len(coords) / (time.time() - t0),
                     mean_loops=float(r["retired_at"].mean()) if "retired_at" in r else 24.0,
                     asset_agreement=ag, frac_solved=float(r["solved"].mean()))
    print(name, res[name], flush=True)
    np.savez_compressed(out / f"{name}.npz", settle_nb=r["settle_nb"], final=r["final"])

def cmp(a, b):
    A, B = runs[a], runs[b]
    same_t = A["settle_nb"] == B["settle_nb"]
    return dict(settle_equal=float(same_t.mean()), final_equal=float((A["final"] == B["final"]).all(-1).mean()),
                settle_equal_interior=float(same_t[nbr == 0].mean()), settle_equal_boundary=float(same_t[nbr > 0].mean()))

pairs = [("full_B1024_a", "full_B1024_b"), ("full_B1024_a", "full_B1000"), ("full_B1024_a", "early_B1024"),
         ("full_B1024_a", "early_B2048"), ("full_B1000", "early_B1024")]
res["pairs"] = {f"{a}__{b}": cmp(a, b) for a, b in pairs}
for k, v in res["pairs"].items():
    print(k, v, flush=True)
(out / "eechk.json").write_text(json.dumps(res, indent=1))
