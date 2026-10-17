#!/usr/bin/env python
"""Reproduction probe: render a K x K grid at pixels of the upstream 1000x1000 asset and compare.

Every sampled pixel sits exactly on the asset's own grid (axis = float32 linspace(-1, 1, 1000)), so the
comparison is pixel against pixel.  Because the asset only stores colours, the comparison is ordinal:
level (dark->light rank of its colour) versus our notebook-definition settle time, under each of the eight
axis-orientation hypotheses.  Writes results/repro/<tag>.json and <tag>.npz.

    .venv/bin/python scripts/repro_probe.py --k 24 --tag probe_k24
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

from render import repro as R  # noqa: E402
from render import runner as RU  # noqa: E402
from render import slices as SL  # noqa: E402
from render import sudoku as S  # noqa: E402
from render.engine import Engine  # noqa: E402

NOTEBOOK_PUZZLE = ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2...."
ASSET = ROOT / "vendor/loopscape/assets/basins_1000_0_eqr_sudoku_plane_rating47_7d68927f389fbdaa_rating47.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=24, help="K x K aligned grid")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--puzzle", default=NOTEBOOK_PUZZLE)
    ap.add_argument("--backend", default="mlx")
    args = ap.parse_args()
    t0 = time.time()

    levels, colours = R.load_asset_levels(ASSET)
    n = levels.shape[0]
    idx = R.aligned_indices(args.k, n)
    rows, cols = np.meshgrid(idx, idx, indexing="ij")
    coords = SL.coords_from_indices(rows.ravel(), cols.ravel(), n=n)
    solution = S.solve(args.puzzle)
    sol = S.pack([solution])[0].astype(np.int8)

    # Raw rollouts are saved chunk by chunk BEFORE any analysis (a crash in the analysis once cost 11 min).
    raw_path = ROOT / "results/repro" / f"{args.tag}.raw.npz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ("final", "settle_nb", "settle_last", "solved", "valid", "n_wrong", "n_changes", "n_distinct", "censored")
    acc = {k: [] for k in keys}
    done, secs = 0, 0.0
    if raw_path.exists():
        z = np.load(raw_path)
        done, secs = int(z["done"]), float(z["seconds"])
        acc = {k: [z[k]] for k in keys}
        print(f"resuming from {done}/{len(coords)} pixels", flush=True)
    if done < len(coords):
        engine = Engine(args.puzzle, backend=args.backend)
        plane = SL.get_plane(seed=0, path=ROOT / "data/plane_seed0.npz")
        chunk = args.batch
        while done < len(coords):
            r = RU.run_pixels(engine, coords[done : done + chunk], plane, args.puzzle, sol, batch=chunk)
            for k in keys:
                acc[k].append(r[k])
            done += len(coords[done : done + chunk])
            secs += r["seconds"]
            np.savez(raw_path, done=done, seconds=secs, **{k: np.concatenate(acc[k]) for k in keys})
            print(f"  {done}/{len(coords)} px  {secs:.0f}s  {done / secs:.2f} rollouts/s", flush=True)
    res = {k: np.concatenate(acc[k]) for k in keys}
    res["seconds"] = secs
    t = res["settle_nb"].reshape(args.k, args.k)
    a = levels[np.ix_(idx, idx)]

    variants = {}
    for name in R.VARIANTS:
        sv = R.variant_sample(a, name)
        shift, frac = R.best_shift_agreement(sv.ravel(), t.ravel())
        variants[name] = dict(
            best_shift=shift, agreement=frac, spearman=R.spearman(sv.ravel(), t.ravel()),
            contingency=R.contingency(sv.ravel(), np.clip(t.ravel() - shift, 0, 10), 11, 11).tolist(),
        )
    best = max(variants, key=lambda k: variants[k]["agreement"])
    out = dict(
        script="scripts/repro_probe.py", puzzle=args.puzzle, solution=solution, plane_seed=0,
        backend=args.backend, k=args.k, n_asset=n, indices=idx.tolist(), n_pixels=int(len(coords)),
        seconds=res["seconds"], rollouts_per_s=len(coords) / res["seconds"],
        settle_nb_hist=np.bincount(res["settle_nb"], minlength=25).tolist(),
        asset_level_hist_at_samples=np.bincount(a.ravel(), minlength=11).tolist(),
        frac_solved=float(res["solved"].mean()), frac_censored=float(res["censored"].mean()),
        n_distinct_final=int(len(np.unique(res["final"], axis=0))),
        variants=variants, best_variant=best,
    )
    p = ROOT / "results/repro"
    p.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p / f"{args.tag}.json", "w"), indent=1)
    np.savez_compressed(p / f"{args.tag}.npz", settle_nb=t, settle_last=res["settle_last"].reshape(args.k, args.k),
                        solved=res["solved"].reshape(args.k, args.k), final=res["final"], idx=idx)
    print("settle_nb histogram:", out["settle_nb_hist"][:20])
    print("asset level histogram at samples:", out["asset_level_hist_at_samples"])
    for name, v in variants.items():
        print(f"{name:8s} agreement={v['agreement']:.3f} shift={v['best_shift']:+d} spearman={v['spearman']:.3f}")
    print("best:", best, f"| solved {out['frac_solved']:.3f} | {out['rollouts_per_s']:.2f} rollouts/s | total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
