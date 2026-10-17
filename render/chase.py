"""Choosing where to zoom: windows with rich, structured, not-yet-saturated boundary detail.

The score of a window of a settle-time field combines
* a boundary term: fraction of pixels with a different-valued 4-neighbour -- zero in flat domains, ~1 in pure
  dust where every pixel differs from its neighbours; a smooth bump peaking at 40% (saturated dust looks like
  noise and carries no structure that can persist under zoom, flat domains carry none either);
* a level term: how many distinct settle values hold >= 2% of the window (up to 10).
The terms are smooth so scores do not saturate and ties are not broken by raster order; ``pick_centre`` can also
favour windows near an anchor (the previous centre) so a chase stays with the structure it was pointed at.
"""
from __future__ import annotations

import numpy as np


def _smooth(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _integral(a: np.ndarray) -> np.ndarray:
    return np.pad(a.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _box(ii: np.ndarray, rows: np.ndarray, cols: np.ndarray, win: int) -> np.ndarray:
    r, c = rows[:, None], cols[None, :]
    return ii[r + win, c + win] - ii[r, c + win] - ii[r + win, c] + ii[r, c]


def window_scores(settle: np.ndarray, win: int = 32, step: int = 8, nlevels: int = 26) -> dict:
    s = np.asarray(settle).astype(np.int64)
    h, w = s.shape
    dx, dy = np.diff(s, axis=1) != 0, np.diff(s, axis=0) != 0
    b = np.zeros((h, w), dtype=bool)
    b[:, :-1] |= dx
    b[:, 1:] |= dx
    b[:-1, :] |= dy
    b[1:, :] |= dy
    rows = np.arange(0, h - win + 1, step)
    cols = np.arange(0, w - win + 1, step)
    frac = _box(_integral(b), rows, cols, win) / (win * win)
    nlev = np.zeros_like(frac)
    for lv in np.unique(s):
        share = _box(_integral(s == lv), rows, cols, win) / (win * win)
        nlev += share >= 0.02
    boundary_term = np.where(frac < 0.03, 0.0, np.exp(-(((frac - 0.40) / 0.17) ** 2)))  # flat windows score 0
    level_term = np.minimum(nlev, 10.0) / 10.0
    return dict(rows=rows, cols=cols, boundary=frac, levels=nlev, score=boundary_term * level_term)


def pick_centre(settle: np.ndarray, win: int = 32, step: int = 8, anchor: tuple[float, float] | None = None,
                anchor_weight: float = 0.0) -> tuple[int, int, float]:
    """(row, col, score) of the centre pixel of the best window; windows far from ``anchor`` are down-weighted."""
    sc = window_scores(settle, win, step)
    score = sc["score"].copy()
    if anchor is not None and anchor_weight > 0:
        h, w = np.asarray(settle).shape
        rr = (sc["rows"] + win // 2)[:, None]
        cc = (sc["cols"] + win // 2)[None, :]
        dist = np.hypot(rr - anchor[0], cc - anchor[1]) / (0.5 * np.hypot(h, w))
        score = score * (1.0 - anchor_weight * np.minimum(1.0, dist))
    i, j = np.unravel_index(np.argmax(score), score.shape)
    return int(sc["rows"][i] + win // 2), int(sc["cols"][j] + win // 2), float(score[i, j])


def pixel_to_plane(row: int, col: int, center: tuple[float, float], half_width: float, n: int) -> tuple[float, float]:
    """Plane coordinates of pixel (row, col) of an n x n window; row 0 is y_min (field-array orientation)."""
    t = lambda k: -1.0 + 2.0 * k / (n - 1)  # noqa: E731
    return center[0] + half_width * t(col), center[1] + half_width * t(row)
