#!/usr/bin/env python
"""Compare a cluster render on the upstream asset's own pixel grid with the asset, pixel by pixel.

    .venv/bin/python scripts/repro_report.py --run data/cluster/repro250 --tag repro250

Reads <run>/field.npz (mode "asset" spec: every STRIDE-th pixel of the 1000-px notebook axis) and the upstream
PNG; writes results/repro/<tag>_summary.json and results/repro/<tag>_side_by_side.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import metrics as M  # noqa: E402
from render import repro as R  # noqa: E402

ASSET = ROOT / "vendor/loopscape/assets/basins_1000_0_eqr_sudoku_plane_rating47_7d68927f389fbdaa_rating47.png"
VARIANT = "xy_fy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run = ROOT / a.run
    spec = json.loads((run / "spec.json").read_text())
    z = np.load(run / "field.npz")
    t = z["settle_nb"].astype(np.int64)
    k, stride = t.shape[0], spec["stride"]
    levels, colours = R.load_asset_levels(ASSET)
    n = levels.shape[0]
    ours_levels = np.clip(t - 2, 0, 10)
    asset_at = R.align_asset(levels, stride, k, VARIANT)
    shift, agree = R.best_shift_agreement(asset_at.ravel(), t.ravel(), clip_top=10)
    ours_levels = np.clip(t - shift, 0, 10)
    match = ours_levels == asset_at

    j, i = np.meshgrid(np.arange(k), np.arange(k))
    r, c = R.asset_pixel_for(VARIANT, stride * j, stride * i, n)
    nbr = R.neighbour_disagreement(levels, r.ravel(), c.ravel()).reshape(k, k)
    interior = nbr == 0

    summary = dict(
        script="scripts/repro_report.py", run=a.run, spec=spec, orientation=VARIANT, level_offset=int(shift),
        grid=[k, k], stride=stride, n_pixels=int(k * k),
        agreement=float(agree), spearman=R.spearman(asset_at.ravel(), t.ravel()),
        agreement_interior=float(match[interior].mean()), frac_interior=float(interior.mean()),
        agreement_boundary=float(match[~interior].mean()),
        agreement_by_asset_level=[float(match[asset_at == q].mean()) if (asset_at == q).any() else None for q in range(11)],
        within_one_level=float((np.abs(ours_levels - asset_at) <= 1).mean()),
        contingency=R.contingency(asset_at.ravel(), ours_levels.ravel(), 11, 11).tolist(),
        hist_ours=np.bincount(ours_levels.ravel(), minlength=11).tolist(),
        hist_asset_same_pixels=np.bincount(asset_at.ravel(), minlength=11).tolist(),
        hist_asset_full=np.bincount(levels.ravel(), minlength=11).tolist(),
        frac_solved=float(((z["flags"] & 1) > 0).mean()), frac_censored=float(((z["flags"] & 4) > 0).mean()),
        metrics_ours=M.label_metrics(ours_levels), metrics_asset_same_pixels=M.label_metrics(asset_at),
        metrics_asset_full={s: v for s, v in M.multi_scale(levels, strides=(1, 2, 4)).items()},
    )
    out = ROOT / "results/repro"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}_summary.json").write_text(json.dumps(summary, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from render import colour as C

    rgb = np.asarray(Image.open(ASSET).convert("RGB"))
    pal = colours.astype(np.uint8)
    flip = lambda a: a[::-1]  # our arrays have row 0 = y_min; show everything y-up like the asset itself  # noqa: E731
    panels = [
        (f"ours: {k}x{k} pixels of the asset's grid\nEqR, fp32, A100", flip(pal[ours_levels])),
        ("upstream asset, the same pixels", flip(pal[asset_at])),
        (f"disagree (amber): {100 * (1 - agree):.1f}% of pixels\nflat regions {100 * (1 - match[interior].mean()):.2f}%, filaments {100 * (1 - match[~interior].mean()):.1f}%",
         flip(np.where(match[..., None], np.array([21, 26, 35], np.uint8), np.array([245, 158, 11], np.uint8)))),
        (f"upstream asset, area-averaged to {k}x{k}", C.downsample(rgb, stride)),
        ("upstream asset, full 1000x1000", rgb),
    ]
    fig, ax = plt.subplots(1, 5, figsize=(25, 5.6), facecolor="#0e1117")
    for axi, (title, img) in zip(ax, panels):
        axi.imshow(img, interpolation="nearest")
        axi.set_title(title, color="#e8ecf5", fontsize=11, loc="left")
        axi.axis("off")
    fig.suptitle(f"Reproduction on the asset's own grid: {100 * agree:.1f}% of {k * k:,} pixels identical, "
                 f"basin entropy {summary['metrics_ours']['basin_entropy']:.3f} vs {summary['metrics_asset_same_pixels']['basin_entropy']:.3f}, "
                 f"boundary dimension {summary['metrics_ours']['boundary_dimension']:.3f} vs {summary['metrics_asset_same_pixels']['boundary_dimension']:.3f}",
                 color="#e8ecf5", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / f"{a.tag}_side_by_side.png", dpi=100, facecolor="#0e1117")
    print(json.dumps({k_: summary[k_] for k_ in ("agreement", "spearman", "agreement_interior", "agreement_boundary", "within_one_level")}, indent=1))


if __name__ == "__main__":
    main()
