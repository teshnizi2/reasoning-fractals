"""Choose neighbouring pixel pairs for the video finale from a rendered keyframe.

A candidate is a pair of horizontally adjacent pixels, both solved and settled (not censored), whose settle times
differ by at least ``min_gap`` loops, away from the image border.  Ranking: bigger gap first, then closer to the
image centre.  Arrays use the render orientation (row 0 = smallest y); the explorer/video flip rows for display.
"""
from __future__ import annotations

import numpy as np


def candidates(settle: np.ndarray, flags: np.ndarray, min_gap: int = 6, margin: int = 20, k: int = 50) -> list[tuple[int, int, int]]:
    """Up to ``k`` (row, col, gap) with the pair at (row, col) and (row, col + 1)."""
    s = np.asarray(settle).astype(np.int64)
    f = np.asarray(flags)
    ok = ((f & 1) > 0) & ((f & 4) == 0)
    ny, nx = s.shape
    gap = np.abs(s[:, 1:] - s[:, :-1])
    gap = np.where(ok[:, 1:] & ok[:, :-1], gap, -1)
    rows, cols = np.mgrid[0:ny, 0 : nx - 1]
    inside = (rows >= margin) & (rows < ny - margin) & (cols >= margin) & (cols < nx - margin - 1)
    gap = np.where(inside, gap, -1)
    cy, cx = (ny - 1) / 2.0, (nx - 2) / 2.0
    dist = np.hypot((rows - cy) / ny, (cols - cx) / nx)
    sel = np.argwhere(gap >= min_gap)
    order = sorted(((int(-gap[r, c]), float(dist[r, c]), int(r), int(c)) for r, c in sel))
    return [(r, c, int(gap[r, c])) for _, _, r, c in order[:k]]
