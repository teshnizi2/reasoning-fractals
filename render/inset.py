"""A small Sudoku card for the video: one start's decoded grid at a given loop, with a 24-tick loop bar."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .overlay import load_font

HEADER = 34  # px reserved above the grid for the title and the tick bar (at size 180; scales with size)
AMBER = (245, 158, 11)


def cell_box(xy: tuple[int, int], size: int, row: int, col: int) -> tuple[int, int, int, int]:
    c = size // 9
    x0, y0 = xy[0] + col * c, xy[1] + HEADER + row * c
    return x0, y0, x0 + c, y0 + c


def draw_sudoku(frame: np.ndarray, grid: str, puzzle: str, solution: str, xy: tuple[int, int], size: int, loop: int,
                settle: int, title: str = "", alpha: float = 1.0) -> np.ndarray:
    """Return a copy of ``frame`` with a Sudoku card at ``xy`` (top-left), ``size`` px wide.

    ``grid`` is the decoded 81-char grid after ``loop`` loops ('.' blank, '?' invalid); cells that are not given and
    differ from ``solution`` are amber; ``settle`` lights the first ``settle`` loop ticks.
    """
    base = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = lambda v: int(round(255 * alpha * v))  # noqa: E731
    c = size // 9
    gw = c * 9
    d.rounded_rectangle([xy[0] - 10, xy[1] - 8, xy[0] + gw + 10, xy[1] + HEADER + gw + 10], radius=12, fill=(14, 17, 23, a(0.82)),
                        outline=(46, 58, 82, a(1.0)), width=2)
    if title:
        d.text((xy[0], xy[1] - 2), title, font=load_font("sans-bold", max(12, c // 2)), fill=(232, 236, 245, a(1.0)))
    # loop ticks
    tw = gw / 24.0
    for t in range(24):
        x = xy[0] + t * tw
        on = t < settle
        col = AMBER if on else (35, 44, 61)
        d.rectangle([x + 1, xy[1] + HEADER - 12, x + tw - 2, xy[1] + HEADER - 5], fill=(*col, a(1.0)))
        if t == loop - 1:
            d.rectangle([x, xy[1] + HEADER - 14, x + tw - 1, xy[1] + HEADER - 3], outline=(232, 236, 245, a(1.0)), width=1)
    font_n = load_font("mono", max(10, int(c * 0.62)))
    font_b = load_font("sans-bold", max(10, int(c * 0.62)))
    for r in range(9):
        for k in range(9):
            i = r * 9 + k
            x0, y0, x1, y1 = cell_box(xy, size, r, k)
            given = puzzle[i] != "."
            ch = grid[i]
            wrong = (not given) and ch != solution[i]
            if given:
                d.rectangle([x0, y0, x1, y1], fill=(232, 236, 245, a(0.10)))
            elif wrong:
                d.rectangle([x0, y0, x1, y1], fill=(*AMBER, a(0.95)))
            d.rectangle([x0, y0, x1, y1], outline=(38, 49, 73, a(1.0)), width=1)
            shown = puzzle[i] if given else ("·" if ch in ".?" else ch)
            col = (232, 236, 245) if given else ((26, 18, 4) if wrong else (185, 196, 222))
            f = font_b if given else font_n
            bb = d.textbbox((0, 0), shown, font=f)
            d.text((x0 + (c - (bb[2] - bb[0])) / 2 - bb[0], y0 + (c - (bb[3] - bb[1])) / 2 - bb[1]), shown, font=f, fill=(*col, a(1.0)))
    for q in (0, 3, 6, 9):  # box lines
        d.line([xy[0] + q * c, xy[1] + HEADER, xy[0] + q * c, xy[1] + HEADER + gw], fill=(74, 90, 122, a(1.0)), width=2)
        d.line([xy[0], xy[1] + HEADER + q * c, xy[0] + gw, xy[1] + HEADER + q * c], fill=(74, 90, 122, a(1.0)), width=2)
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB")).copy()
