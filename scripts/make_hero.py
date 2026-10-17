#!/usr/bin/env python
"""Hero images from a rendered field (settle time + flags per pixel).

    .venv/bin/python scripts/make_hero.py --run data/cluster/hero --way loopscape --out media

Writes (y up, like the upstream figure):
  hero.png            native resolution, one colour per pixel (the data as computed)
  hero-print.png      2x Lanczos upsample of the colour image in linear light (for print; no new information)
  hero-1080x1350.png  4:5 crop for the LinkedIn feed, area-averaged to 1080 wide
  hero-1080x1080.png  square crop
Crop windows are given in plane units (--crop cx,cy,half_width_x) or default to the centre of the plane.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import colour as C  # noqa: E402


def crop_box(n: int, cx: float, cy: float, hw: float, aspect: float) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of a window centred on plane (cx, cy), width 2*hw, height 2*hw*aspect, y up."""
    ppu = (n - 1) / 2.0  # pixels per plane unit on [-1, 1]
    w = int(round(2 * hw * ppu))
    h = int(round(w * aspect))
    x0 = int(round((cx + 1) * ppu - w / 2))
    y0 = int(round((1 - cy) * ppu - h / 2))  # row 0 = y max
    x0 = max(0, min(n - w, x0))
    y0 = max(0, min(n - h, y0))
    return x0, y0, x0 + w, y0 + h


def to_size(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resample an RGB uint8 image to (w, h): linear-light area average down, Lanczos up."""
    ih, iw, _ = img.shape
    if iw > w and iw % w == 0 and ih % h == 0 and iw // w == ih // h:
        return C.downsample(img, iw // w)
    lin = C.srgb_to_linear(img.astype(np.float64) / 255.0)
    chans = [np.asarray(Image.fromarray(lin[..., k].astype(np.float32), mode="F").resize((w, h), Image.LANCZOS)) for k in range(3)]
    return np.rint(C.linear_to_srgb(np.stack(chans, -1)) * 255.0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--way", default="loopscape")
    ap.add_argument("--out", default="media")
    ap.add_argument("--crop", default="0,0,0.8", help="cx,cy,half_width_x for the 4:5 crop (plane units)")
    ap.add_argument("--no-print", action="store_true")
    a = ap.parse_args()
    z = np.load(ROOT / a.run / "field.npz")
    cw = C.COLOURWAYS[a.way]
    img = C.colourize(z["settle_nb"], z["flags"], cw)[::-1]  # y up
    n = img.shape[0]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out / "hero.png", optimize=True)
    if not a.no_print:
        Image.fromarray(to_size(img, 2 * n, 2 * n)).save(out / "hero-print.png", optimize=True)
    cx, cy, hw = (float(v) for v in a.crop.split(","))
    x0, y0, x1, y1 = crop_box(n, cx, cy, hw, 1.25)
    Image.fromarray(to_size(np.ascontiguousarray(img[y0:y1, x0:x1]), 1080, 1350)).save(out / "hero-1080x1350.png", optimize=True)
    x0, y0, x1, y1 = crop_box(n, cx, cy, hw, 1.0)
    Image.fromarray(to_size(np.ascontiguousarray(img[y0:y1, x0:x1]), 1080, 1080)).save(out / "hero-1080x1080.png", optimize=True)
    print("wrote", sorted(p.name for p in out.glob("hero*.png")), "colourway", a.way, "n", n)


if __name__ == "__main__":
    main()
