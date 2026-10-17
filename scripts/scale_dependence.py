#!/usr/bin/env python
"""How complex is the settle-time field at different pixel pitches?  (the paper's App. E question)

Basin entropy and boundary dimension (loopscape.metrics, 5x5 boxes, eps = 1..5 px) of: the upstream image at several
strides, our 250x250 reproduction, the hero at several strides (if rendered), and the small zoom windows of the chase
jobs.  Writes results/scale_dependence.json and results/figures/scale_dependence.png.  A boundary dimension that falls
as the pitch gets finer is the signature of a *slim* fractal; one that stays constant is a true fractal.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import metrics as M  # noqa: E402
from render import repro as R  # noqa: E402

ASSET = ROOT / "vendor/loopscape/assets/basins_1000_0_eqr_sudoku_plane_rating47_7d68927f389fbdaa_rating47.png"


def entry(label, field, pitch, source):
    m = M.label_metrics(field)
    return dict(label=label, source=source, pitch=pitch, shape=list(field.shape), basin_entropy=m["basin_entropy"],
                boundary_basin_entropy=m["boundary_basin_entropy"], boundary_box_fraction=m["boundary_box_fraction"],
                uncertainty_exponent=m["uncertainty_exponent"], boundary_dimension=m["boundary_dimension"],
                uncertainty_r2=m["uncertainty_r2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", default=None, help="merged hero run dir (field.npz)")
    ap.add_argument("--chase", nargs="*", default=[], help="chase run dirs (chase.json + level*.npz)")
    ap.add_argument("--out", default="results/scale_dependence.json")
    a = ap.parse_args()
    rows = []
    levels, _ = R.load_asset_levels(ASSET)
    for s in (1, 2, 4, 8):
        rows.append(entry(f"upstream image, every {s}th pixel", levels[::s, ::s], 2.0 / 999 * s, "upstream asset (levels)"))
    rp = np.load(ROOT / "results/repro/repro250_field.npz")["settle_nb"]
    rows.append(entry("ours, 250x250 (asset grid, fp32)", np.clip(rp.astype(int) - 2, 0, 10), 2.0 / 999 * 4, "results/repro/repro250_field.npz"))
    if a.hero:
        h = np.load(ROOT / a.hero / "field.npz")["settle_nb"]
        for s in (1, 2, 4, 8):
            rows.append(entry(f"hero 2048x2048, every {s}th pixel", h[::s, ::s], 2.0 / 2047 * s, "hero"))
    for d in a.chase:
        d = Path(ROOT / d)
        meta = json.loads((d / "chase.json").read_text())
        for lv in meta["levels"]:
            z = np.load(d / f"level{lv['level']}.npz")
            rows.append(entry(f"window x{lv['zoom']:g} (160x160, {d.name})", z["settle_nb"], 2 * lv["half_width"] / (lv["n"] - 1), f"{d.name}/level{lv['level']}"))
    rows.sort(key=lambda r: -r["pitch"])
    out = ROOT / a.out
    out.write_text(json.dumps(dict(script="scripts/scale_dependence.py", rows=rows), indent=1))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), facecolor="#0e1117")
    for axi, key, lab in zip(ax, ("boundary_dimension", "basin_entropy", "boundary_box_fraction"),
                             ("boundary dimension 2 - alpha", "basin entropy S_b", "fraction of boxes touching a boundary")):
        axi.set_facecolor("#151a23")
        for kind, col in (("upstream", "#8893ab"), ("ours", "#f59e0b"), ("hero", "#f59e0b"), ("window", "#7fdc9b")):
            pts = [(r["pitch"], r[key]) for r in rows if r["label"].startswith(kind)]
            if pts:
                axi.plot(*zip(*pts), "o-", color=col, label=kind, ms=6)
        axi.set_xscale("log")
        axi.invert_xaxis()
        axi.set_xlabel("pixel pitch in the (a, b) plane (finer to the right)", color="#e8ecf5")
        axi.set_ylabel(lab, color="#e8ecf5")
        axi.tick_params(colors="#8893ab")
        for sp in axi.spines.values():
            sp.set_color("#2e3a52")
        axi.legend(facecolor="#151a23", edgecolor="#2e3a52", labelcolor="#e8ecf5")
    fig.tight_layout()
    (ROOT / "results/figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "results/figures/scale_dependence.png", dpi=110, facecolor="#0e1117")
    for r in rows:
        print(f"{r['label']:44s} pitch {r['pitch']:.2e}  D={r['boundary_dimension']:.3f}  S_b={r['basin_entropy']:.3f}  boxes {r['boundary_box_fraction']:.2f}")


if __name__ == "__main__":
    main()
