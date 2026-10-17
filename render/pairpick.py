"""Choose neighbouring pixel pairs for the video finale from a rendered keyframe.

A candidate is a pair of horizontally adjacent pixels, both solved and settled (not censored), whose settle times
differ by at least ``min_gap`` loops, away from the image border.  Ranking: bigger gap first, then closer to the
image centre.  Arrays use the render orientation (row 0 = smallest y); the explorer/video flip rows for display.
"""
from __future__ import annotations

import numpy as np


def candidates(settle: np.ndarray, flags: np.ndarray, min_gap: int = 6, margin: int = 20, k: int = 50,
               max_gap: int | None = None, clean: bool = False) -> list[tuple[int, int, int]]:
    """Up to ``k`` (row, col, gap) with the pair at (row, col) and (row, col + 1); gaps above ``max_gap`` are skipped.

    ``clean`` keeps only clean steps: each pixel equals its other three neighbours (left/up/down for the left one,
    right/up/down for the right one), i.e. a straight border between two flat domains rather than speckle."""
    s = np.asarray(settle).astype(np.int64)
    f = np.asarray(flags)
    ok = ((f & 1) > 0) & ((f & 4) == 0)
    ny, nx = s.shape
    gap = np.abs(s[:, 1:] - s[:, :-1])
    gap = np.where(ok[:, 1:] & ok[:, :-1], gap, -1)
    if clean:
        a, b = s[1:-1, 1:-2], s[1:-1, 2:-1]  # the pair at (r, c), (r, c + 1) for r in 1..ny-2, c in 1..nx-3
        flat = ((a == s[1:-1, :-3]) & (a == s[:-2, 1:-2]) & (a == s[2:, 1:-2])
                & (b == s[1:-1, 3:]) & (b == s[:-2, 2:-1]) & (b == s[2:, 2:-1]))
        keep = np.zeros_like(gap, dtype=bool)
        keep[1:-1, 1:-1] = flat
        gap = np.where(keep, gap, -1)
    rows, cols = np.mgrid[0:ny, 0 : nx - 1]
    inside = (rows >= margin) & (rows < ny - margin) & (cols >= margin) & (cols < nx - margin - 1)
    gap = np.where(inside, gap, -1)
    cy, cx = (ny - 1) / 2.0, (nx - 2) / 2.0
    dist = np.hypot((rows - cy) / ny, (cols - cx) / nx)
    sel = np.argwhere((gap >= min_gap) & (gap <= (max_gap if max_gap is not None else 10**6)))
    order = sorted(((int(-gap[r, c]), float(dist[r, c]), int(r), int(c)) for r, c in sel))
    return [(r, c, int(gap[r, c])) for _, _, r, c in order[:k]]
