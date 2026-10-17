#!/usr/bin/env python
"""Write the explorer's copy (subtitle, limits, how-it-was-made) into a pack manifest, with every number read from results/.

    .venv/bin/python scripts/site_text.py --pack site/data

Numbers come from results/metrics.json (hero), results/repro/repro250_summary.json (reproduction),
results/alice/gpu_hours.json and the pack itself; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def build(pack: dict, metrics: dict, repro: dict, gpu: dict | None, scale: dict | None = None, nums: dict | None = None) -> dict:
    nums = nums or {}
    hero = metrics["hero"]
    thin_sentence = "(scale data not available.)"
    deepest = None
    if scale:
        rows = sorted(scale["rows"], key=lambda r: -r["pitch"])
        coarse = next((r for r in rows if r["label"].startswith("hero 2048x2048, every pixel")), None) or \
            next((r for r in rows if r["label"].startswith("upstream image, every pixel")), rows[0])
        wins = [r for r in rows if r["label"].startswith("window")]
        kfs = [r for r in rows if r["label"].startswith("keyframe")]
        deepest = max((1 / (r["pitch"] * (r["shape"][0] - 1) / 2) for r in wins), default=None)
        fine = min(kfs, key=lambda r: r["pitch"]) if kfs else (wins[0] if wins else None)  # unbiased keyframes first, else the first chase window
        if fine:
            zoom_fine = 1 / (fine["pitch"] * (fine["shape"][1] - 1) / 2)
            thin_sentence = (f"In 5×5-pixel boxes, {pct(coarse['boundary_box_fraction'], 0)} touch a boundary at the full map's pixel pitch "
                             f"({coarse['pitch']:.0e}), but only {pct(fine['boundary_box_fraction'], 0)} in the layer magnified ×{zoom_fine:.0f} (pitch {fine['pitch']:.0e}).")
    layers = pack["layers"]
    n_px = sum(l["nx"] * l["ny"] for l in layers)
    zoom_max = max(l["zoom"] for l in layers)
    n_unsolved = int(round((1 - hero["frac_solved"]) * hero["pixels"]))
    n_cens = min(n_unsolved, int(round(hero["frac_censored"] * hero["pixels"])))
    limits = [
        "<b>One puzzle, one model.</b> EqR's released Sudoku checkpoint on a rating-47 puzzle, the one behind loopscape's own figure. Other puzzles give other maps.",
        "<b>One slice.</b> A random 2-D plane (seed 0) through the all-zero starting state of a 49,664-dimensional space. The starting thoughts here are tiny "
        + (f"(distance from zero at most {nums['pscan.plane_norm']:.2f}) next to EqR's usual random start (about {nums['pscan.native_norm']:.0f})." if "pscan.plane_norm" in nums and nums.get("pscan.native_norm")
           else "next to EqR's usual random start."),
        f"<b>Almost every start ends right.</b> {pct(hero['frac_solved'])} of the {hero['pixels']:,} starts of the full map reach the true solution. What is fractal is <i>how long</i> it takes. The {n_unsolved:,} pixels that do not end on the solution are drawn in amber: {n_cens:,} are starts still changing at loop 24 (UNSETTLED), {n_unsolved - n_cens:,} settled on a wrong grid (STUCK).",
        "<b>float32 is touchy.</b> "
        + (f"Re-run on a second float32 implementation, {pct(nums['replay.flat'], 0)} of the pixels with flat surroundings keep their settle time but only {pct(nums['replay.dust'], 0)} of the speckle pixels do. "
           if "replay.flat" in nums and "replay.dust" in nums else "Two different float32 implementations agree where the map is locally flat but far less in the speckle. ")
        + (f"On the same GPU, a nudge of 1e-6 to a start leaves {pct(nums['nudge.flat'], 0)} of flat-region pixels unchanged and {pct(nums['nudge.dust'], 0)} of speckle pixels. " if "nudge.flat" in nums and "nudge.dust" in nums else "")
        + f"(Ours vs loopscape's published image: {pct(repro['agreement'])} of {repro['n_pixels']:,} pixels identical.) TF32 or bfloat16 give a different picture. Read single speckle pixels as one realisation; the structure is the result.",
        f"<b>The complexity thins out when you zoom.</b> {thin_sentence} Zoom beyond ×{int(zoom_max):,} was not rendered: " + (f"a trial window at ×{deepest:,.0f} held only speckle next to a flat domain." if deepest else "the map is speckle there."),
        "<b>Settle time</b> is measured over 24 loops with the model's noise switched off, as in loopscape: the loop after which the decoded grid never changes again.",
    ]
    how = [
        "EqR checkpoint (locuslab) run as in loopscape: z_H = a·u + b·v, z_L = 0, 24 loops, noise off, float32.",
        f"{n_px:,} starting thoughts computed on A100 GPUs of the ALICE cluster (Leiden): {hero['shape'][0]}² for the full plane and {len(layers) - 1} zoom layers around the marked point."
        + (f" About {gpu['total']['a100_hours']:.0f} A100-hours." if gpu else ""),
        "A start stops being computed once its decoded grid has equalled the true solution for 3 loops in a row (checked against full 24-loop runs).",
        f"Reproduction check: loopscape's published 1000×1000 image, every 4th pixel: basin entropy {repro['metrics_ours']['basin_entropy']:.3f} vs {repro['metrics_asset_same_pixels']['basin_entropy']:.3f}, boundary dimension {repro['metrics_ours']['boundary_dimension']:.3f} vs {repro['metrics_asset_same_pixels']['boundary_dimension']:.3f}.",
    ]
    return dict(subtitle="Each pixel is one starting thought for the same Sudoku. Colour is how many loops the model took to settle on its answer.",
                limits=limits, how=how)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="site/data")
    a = ap.parse_args()
    mp = ROOT / a.pack / "manifest.json"
    man = json.loads(mp.read_text())
    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    repro = json.loads((ROOT / "results/repro/repro250_summary.json").read_text())
    gp = ROOT / "results/alice/gpu_hours.json"
    gpu = json.loads(gp.read_text()) if gp.exists() else None
    sp = ROOT / "results/scale_dependence.json"
    scale = json.loads(sp.read_text()) if sp.exists() else None
    sys.path.insert(0, str(ROOT))
    from render import readme as RM

    N = RM.load_numbers(ROOT)
    man["meta"].update(build(man, metrics, repro, gpu, scale, N))
    if all(k in N for k in ("replay.flat", "replay.edge", "replay.dust")):  # shown in the inspector: how often a second float32 stack agrees
        man["meta"]["reproduce"] = dict(flat=N["replay.flat"], edge=N["replay.edge"], dust=N["replay.dust"])
    mp.write_text(json.dumps(man))
    print("wrote text into", mp)


if __name__ == "__main__":
    main()
