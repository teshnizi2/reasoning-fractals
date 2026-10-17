"""Helpers for the reproduction check against loopscape's 1000x1000 EqR/Sudoku asset.

The asset is a PNG with exactly 11 colours: the settle-time field pushed through a sequential colormap.
Colours are ranked by luminance (dark -> light), which is the colormap's own order, so a pixel's *level*
is an ordinal copy of its settle time and can be compared with ours without knowing the colormap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata

# (swap_axes, flip_x, flip_y): how an image pixel (row r, col c) maps to plane axis indices.
_VARIANT_FLAGS = {
    "xy": (False, False, False),
    "xy_fx": (False, True, False),
    "xy_fy": (False, False, True),
    "xy_fxfy": (False, True, True),
    "yx": (True, False, False),
    "yx_fx": (True, True, False),
    "yx_fy": (True, False, True),
    "yx_fxfy": (True, True, True),
}
VARIANTS = tuple(_VARIANT_FLAGS)


def load_asset_levels(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """PNG -> (levels uint8 (H, W) ranked dark -> light, colours (K, 3))."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    colours, inv = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    lum = colours @ np.array([0.2126, 0.7152, 0.0722])
    order = np.argsort(lum)
    rank = np.empty(len(colours), dtype=np.uint8)
    rank[order] = np.arange(len(colours))
    return rank[inv.reshape(rgb.shape[:2])], colours[order]


def variant_axes(name: str, r: np.ndarray, c: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Plane-axis indices (x_index, y_index) of image pixels (r, c) under an orientation hypothesis."""
    swap, fx, fy = _VARIANT_FLAGS[name]
    r, c = np.asarray(r), np.asarray(c)
    xi, yi = (r, c) if swap else (c, r)
    if fx:
        xi = n - 1 - xi
    if fy:
        yi = n - 1 - yi
    return xi, yi


def aligned_indices(k: int, n: int) -> np.ndarray:
    """k asset-grid indices spanning [0, n-1], mirror-symmetric (idx + idx[::-1] == n - 1)."""
    x = np.arange(k) * (n - 1) / (k - 1)
    idx = np.floor(x + 0.5).astype(np.int64)
    half = k // 2
    idx[k - half :] = (n - 1) - idx[:half][::-1]
    return idx


def best_shift_agreement(levels: np.ndarray, t: np.ndarray, max_shift: int = 40) -> tuple[int, float]:
    """Best integer s so that level == t - s most often; returns (s, fraction of agreeing pixels)."""
    levels, t = np.asarray(levels).astype(np.int64), np.asarray(t).astype(np.int64)
    best = (0, -1.0)
    for s in range(-max_shift, max_shift + 1):
        f = float((levels == t - s).mean())
        if f > best[1]:
            best = (s, f)
    return best


def contingency(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> np.ndarray:
    m = np.zeros((na, nb), dtype=np.int64)
    np.add.at(m, (np.asarray(a), np.asarray(b)), 1)
    return m


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def variant_sample(a: np.ndarray, name: str) -> np.ndarray:
    """Re-express a K x K sample ``a = asset[idx, idx]`` in the plane frame of an orientation hypothesis.

    Result ``s[i, j]`` is the asset level at the pixel whose plane-axis indices are (x=idx[j], y=idx[i]).
    Valid because ``aligned_indices`` is mirror-symmetric, so flipping an axis just reverses the sample.
    """
    swap, fx, fy = _VARIANT_FLAGS[name]
    k = a.shape[0]
    i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    ii = k - 1 - i if fy else i
    jj = k - 1 - j if fx else j
    return a[jj, ii] if swap else a[ii, jj]
