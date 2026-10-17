#!/usr/bin/env python
"""How well does a second float32 stack reproduce the A100 renders, pixel by pixel?  (MLX on this Mac vs torch/CUDA.)

    .venv/bin/python scripts/replay_agreement.py --chase data/cluster/chase2 --run data/cluster/hero --n 160 \
        --out results/replay_agreement.json

Random pixels of each field are re-run locally (full 24 loops, no early exit) and compared with the stored settle
time, split by how rough the stored field is around the pixel (largest settle-time difference to its 4 neighbours).
Flat pixels reproduce; pixels in the speckled "dust" mostly do not, and slow dust pixels (stored settle time >= 8)
essentially never do.  That is the numerical-sensitivity caveat behind every pixel-exact claim in the README.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import puzzles as P  # noqa: E402
from render import runner as RU  # noqa: E402
from render import slices as SL  # noqa: E402
from render import sudoku as S  # noqa: E402
from render import tilerun as TR  # noqa: E402
from render.engine import Engine  # noqa: E402

BINS = ((0, 0), (1, 1), (2, 4), (5, 99))


def roughness(s: np.ndarray) -> np.ndarray:
    s = s.astype(int)
    pad = np.pad(s, 1, mode="edge")
    ny, nx = s.shape
    return np.max([np.abs(s - pad[1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx]) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))], axis=0)


def fields(a) -> list[tuple[str, dict, np.ndarray]]:
    out = []
    if a.chase:
        d = ROOT / a.chase
        c = json.loads((d / "chase.json").read_text())
        for lv in c["levels"]:
            z = np.load(d / f"level{lv['level']}.npz")
            n = z["settle_nb"].shape[0]
            spec = dict(mode="window", center=lv["center"], half_width=lv["half_width"], nx=n, ny=n, n=n, latent_precision="float64")
            out.append((f"chase window x{lv['zoom']:.0f}", spec, z["settle_nb"]))
    for r in a.run:
        d = ROOT / r
        spec = json.loads((d / "spec.json").read_text())
        out.append((f"{d.name}", spec, np.load(d / "field.npz")["settle_nb"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chase", default=None)
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=160, help="random pixels per field")
    ap.add_argument("--out", default="results/replay_agreement.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sol = S.pack([P.ASSET.answer])[0].astype(np.int8)
    plane = SL.get_plane(0, ROOT / "data/plane_seed0.npz")
    eng = Engine(P.ASSET.question, backend="mlx")
    rng = np.random.default_rng(a.seed)
    rep = dict(script="scripts/replay_agreement.py", engine="MLX float32, Apple M4 (local) vs torch fp32 on A100 (stored)",
               n_per_field=a.n, seed=a.seed, bins="roughness = max |settle difference| to the 4 neighbours in the stored field",
               fields=[])
    for name, spec, settle in fields(a):
        ny, nx = settle.shape
        idx = rng.choice(ny * nx, size=min(a.n, ny * nx), replace=False)
        r, c = np.divmod(idx, nx)
        coords = np.concatenate([TR.pixel_coords(spec, TR.Tile(0, 0, 0, int(i), int(j), 1, 1)) for i, j in zip(r, c)])
        t0 = time.time()
        res = RU.run_pixels(eng, coords, plane, P.ASSET.question, sol, batch=80, precision=TR._latent_precision(spec))
        stored = settle[r, c].astype(int)
        got = res["settle_nb"].astype(int)
        rough = roughness(settle)[r, c]
        rows = []
        for lo, hi in BINS:
            m = (rough >= lo) & (rough <= hi)
            slow = m & (stored >= 8)
            rows.append(dict(roughness=f"{lo}" if lo == hi else f"{lo}-{hi if hi < 99 else '+'}", n=int(m.sum()),
                             equal=int((stored[m] == got[m]).sum()), n_slow=int(slow.sum()), slow_equal=int((stored[slow] == got[slow]).sum())))
        rep["fields"].append(dict(name=name, shape=[ny, nx], n=int(len(idx)), seconds=time.time() - t0,
                                  equal=int((stored == got).sum()), frac_equal=float((stored == got).mean()), by_roughness=rows))
        print(name, f"{time.time() - t0:.0f}s", "equal", rep["fields"][-1]["frac_equal"], flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
