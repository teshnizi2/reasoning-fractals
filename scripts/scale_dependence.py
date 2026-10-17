#!/usr/bin/env python
"""How complex is the settle-time field at different pixel pitches?  (the paper's App. E question)

Settle times are clipped to 2..12 (the upstream image's 11 levels) so every row uses the same labels.
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


def R_ord(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


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
    ap.add_argument("--keyframes", nargs="*", default=[], help="merged keyframe run dirs (field.npz + spec.json)")
    ap.add_argument("--out", default="results/scale_dependence.json")
    a = ap.parse_args()
    rows = []
    levels, _ = R.load_asset_levels(ASSET)
    for s in (1, 2, 4, 8):
        rows.append(entry(f"upstream image, every {R_ord(s)} pixel" if s > 1 else "upstream image, every pixel", levels[::s, ::s], 2.0 / 999 * s, "upstream asset (levels)"))
    rp = np.load(ROOT / "results/repro/repro250_field.npz")["settle_nb"]
    rows.append(entry("ours, 250x250 (asset grid, fp32)", np.clip(rp.astype(int) - 2, 0, 10), 2.0 / 999 * 4, "results/repro/repro250_field.npz"))
    if a.hero:
        h = np.clip(np.load(ROOT / a.hero / "field.npz")["settle_nb"], 2, 12)  # the upstream image's 11 levels (top one saturated)
        for s in (1, 2, 4, 8):
            rows.append(entry(f"hero 2048x2048, every {R_ord(s)} pixel" if s > 1 else "hero 2048x2048, every pixel", h[::s, ::s], 2.0 / 2047 * s, "hero (settle time clipped to 2..12)"))
    for d in a.keyframes:
        d = Path(ROOT / d)
        spec = json.loads((d / "spec.json").read_text())
        kf = np.clip(np.load(d / "field.npz")["settle_nb"], 2, 12)
        rows.append(entry(f"keyframe x{spec['zoom']:g} ({kf.shape[1]}x{kf.shape[0]}, centred on the zoom target)", kf,
                          2 * spec["half_width"] / (spec["nx"] - 1), f"{d.name} (settle time clipped to 2..12)"))
    for d in a.chase:
        d = Path(ROOT / d)
        meta = json.loads((d / "chase.json").read_text())
        for lv in meta["levels"]:
            z = np.load(d / f"level{lv['level']}.npz")
            rows.append(entry(f"window x{lv['zoom']:g} (160x160, {d.name})", np.clip(z["settle_nb"], 2, 12), 2 * lv["half_width"] / (lv["n"] - 1), f"{d.name}/level{lv['level']} (settle time clipped to 2..12)"))
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
        for kind, col in (("upstream", "#8893ab"), ("ours", "#f59e0b"), ("hero", "#f59e0b"), ("keyframe", "#fcc95a"), ("window", "#7fdc9b")):
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
