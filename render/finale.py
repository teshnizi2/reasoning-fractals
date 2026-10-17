"""Pair finale of the video: two neighbouring starts ringed on the fractal, their Sudoku grids replaying beneath.

Everything is drawn in frame pixels; ``layout`` gives the card geometry for one frame format (``PORTRAIT`` is for a
1080x1350 frame, ``scaled`` adapts it to other sizes).  The spec comes from ``scripts/build_pair.py``: two adjacent
samples of the deepest keyframe (a to the left of b) whose loop-by-loop traces were re-run locally and reproduce the
stored settle times.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import inset as IN
from . import zoomvideo as ZV
from .overlay import load_font

CARD_FADE = 0.7      # seconds for rings and cards to fade in
REPLAY_DELAY = 0.5   # seconds between the cards being up and the first loop
AMBER = (245, 158, 11)
WHITE = (232, 236, 245)

# 1080x1350 frame: pair of blocks at pair_y (screen y of the pair's centre), two cards side by side below it
PORTRAIT = dict(pair_y=400, card_size=470, card_y=725, gap=30)
# 1080x1080 centre crop of the same frame (crop top = 135): the pair sits 135 px higher, cards are smaller
SQUARE = dict(pair_y=400 - 135, card_size=380, card_y=555, gap=30)


def scaled(layout: dict, k: float) -> dict:
    return {key: v * k for key, v in layout.items()}


def loop_at(sec: float, spec: dict) -> int:
    """Loop shown on the cards at time ``sec``: 0 (the puzzle) until the replay starts, then one loop per step."""
    t0 = spec["t_in"] + CARD_FADE + REPLAY_DELAY
    if sec < t0:
        return 0
    return int(min(spec["loops_shown"], 1 + (sec - t0) // spec["loop_seconds"]))


def block_px(view: tuple[float, float, float, float], size: tuple[int, int], spec: dict) -> float:
    """On-screen size of one sample of the finest layer."""
    scale = size[0] / (view[3] / view[2])
    return spec["pitch"] * scale


def draw_finale(frame: np.ndarray, spec: dict, sec: float, view: tuple[float, float, float, float], layout: dict) -> np.ndarray:
    """Copy of ``frame`` with rings, connectors, the two cards and a loop readout; ``view`` = (cx, cy, zoom, view_width)."""
    if sec <= spec["t_in"]:
        return frame.copy()
    h, w = frame.shape[:2]
    fade = min(1.0, (sec - spec["t_in"]) / CARD_FADE)
    cx, cy, zoom, vw = view
    size = int(layout["card_size"])
    gap = layout["gap"]
    x_cards = {"a": int(round(w / 2 - gap / 2 - size)), "b": int(round(w / 2 + gap / 2))}
    y_card = int(layout["card_y"])
    half = block_px(view, (w, h), spec) / 2
    ring_w = max(2, int(round(half * 0.12)))
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    al = int(round(255 * fade))
    c = size // 9
    for key, col in (("a", AMBER), ("b", WHITE)):
        s = spec[key]
        sx, sy = ZV.to_screen(s["x"], s["y"], cx, cy, zoom, (w, h), vw)
        d.rectangle([sx - half - ring_w - 2, sy - half - ring_w - 2, sx + half + ring_w + 2, sy + half + ring_w + 2], outline=(14, 17, 23, int(al * 0.8)), width=ring_w + 4)
        d.rectangle([sx - half - ring_w, sy - half - ring_w, sx + half + ring_w, sy + half + ring_w], outline=(*col, al), width=ring_w)
        card_top = (x_cards[key] + size / 2, y_card - 8)
        d.line([(sx, sy + half + ring_w), card_top], fill=(14, 17, 23, int(al * 0.85)), width=6)  # dark underlay: stays visible on busy colours
        d.line([(sx, sy + half + ring_w), card_top], fill=(*col, int(al * 0.9)), width=3)
        f = load_font("mono", max(14, int(c * 0.8)))
        bb = d.textbbox((0, 0), s["label"], font=f)
        d.text((sx - (bb[2] - bb[0]) / 2 - bb[0], sy - half - ring_w - (bb[3] - bb[1]) - 12 - bb[1]), s["label"], font=f, fill=(*col, al))
    out = np.asarray(Image.alpha_composite(Image.fromarray(frame).convert("RGBA"), lay).convert("RGB")).copy()
    loop = loop_at(sec, spec)
    for key in ("a", "b"):
        s = spec[key]
        grid = spec["puzzle"] if loop == 0 else s["trace"][loop - 1]
        title = f"{s['label']} · settles in {s['settle']} loops"
        out = IN.draw_sudoku(out, grid, spec["puzzle"], spec["solution"], (x_cards[key], y_card), size, loop, s["settle"], title=title, alpha=fade,
                             loop_label=f"loop {loop}")
    return out
