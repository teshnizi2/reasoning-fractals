"""Candidate "featured starts" for the explorer, chosen from a rendered field (settle time + flag word per pixel).

Arrays use the render orientation (row 0 = smallest y); ``to_display`` converts to the pack's y-up image rows.
Each function returns (row, col) candidates, best first; the explorer only keeps ones whose loop-by-loop replay
(re-run locally) reproduces the stored settle time and outcome.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def _differs_from_neighbour(s: np.ndarray) -> np.ndarray:
    d = np.zeros(s.shape, dtype=bool)
    dx, dy = s[:, 1:] != s[:, :-1], s[1:, :] != s[:-1, :]
    d[:, :-1] |= dx
    d[:, 1:] |= dx
    d[:-1, :] |= dy
    d[1:, :] |= dy
    return d


def fast(settle: np.ndarray, flags: np.ndarray, k: int = 5) -> list[tuple[int, int]]:
    """Solved pixels of the fastest *common* settle time (>= 1% of the pixels), deepest inside their flat domain first."""
    s = np.asarray(settle)
    ok = ((np.asarray(flags) & 1) > 0) & ((np.asarray(flags) & 4) == 0)
    counts = np.bincount(s[ok].astype(np.int64))
    tmin = int(np.flatnonzero(counts >= 0.01 * ok.sum())[0])
    dist = distance_transform_edt(~_differs_from_neighbour(s))
    score = np.where(ok & (s == tmin), dist, -1.0)
    idx = np.argsort(score.ravel())[::-1][:k]
    return [tuple(int(v) for v in np.unravel_index(i, s.shape)) for i in idx if score.ravel()[i] >= 0]


def slow(settle: np.ndarray, flags: np.ndarray, k: int = 5, max_settle: int = 20) -> list[tuple[int, int]]:
    """Solved, settled pixels with the longest settle times below ``max_settle`` (at the 24-loop cap a replay wobbles by a loop)."""
    s = np.asarray(settle).astype(np.int64)
    ok = ((np.asarray(flags) & 1) > 0) & ((np.asarray(flags) & 4) == 0) & (s <= max_settle)
    score = np.where(ok, s, -1)
    idx = np.argsort(score.ravel(), kind="stable")[::-1][:k]
    return [tuple(int(v) for v in np.unravel_index(i, s.shape)) for i in idx if score.ravel()[i] >= 0]


def pair(settle: np.ndarray, flags: np.ndarray, k: int = 5, min_gap: int = 6) -> list[tuple[int, int]]:
    """Left pixel of horizontally adjacent solved pixels whose settle times differ by at least ``min_gap`` loops."""
    s = np.asarray(settle).astype(np.int64)
    ok = ((np.asarray(flags) & 1) > 0) & ((np.asarray(flags) & 4) == 0)
    gap = np.abs(s[:, 1:] - s[:, :-1])
    gap = np.where(ok[:, 1:] & ok[:, :-1], gap, -1)
    idx = np.argsort(gap.ravel(), kind="stable")[::-1][:k]
    return [tuple(int(v) for v in np.unravel_index(i, gap.shape)) for i in idx if gap.ravel()[i] >= min_gap]


def wrong(settle: np.ndarray, flags: np.ndarray, k: int = 5) -> list[tuple[int, int]]:
    """Pixels that end on a wrong grid: settled ones first, then unsettled ones (still changing at the cap)."""
    f = np.asarray(flags)
    s = np.asarray(settle).astype(np.int64)
    settled = np.argwhere(((f & 1) == 0) & ((f & 4) == 0))
    unsettled = np.argwhere((f & 4) > 0)
    order = [tuple(int(v) for v in p) for p in settled[np.argsort(-s[settled[:, 0], settled[:, 1]], kind="stable")]]
    order += [tuple(int(v) for v in p) for p in unsettled[np.argsort(-s[unsettled[:, 0], unsettled[:, 1]], kind="stable")]]
    return order[:k]


def to_display(row: int, col: int, ny: int) -> tuple[int, int]:
    return ny - 1 - row, col
