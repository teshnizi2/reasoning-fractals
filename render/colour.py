"""Colourways: settle time -> luminance, outcome -> hue.

Every pixel of a render has a settle time (loops until the decoded answer stopped changing), a flag word
(bit0 solved, bit1 valid, bit2 censored = still changing at the loop cap) and an outcome.  A *colourway*
maps (settle, flags) to RGB with two ramps over the same settle range, both getting lighter as the model
takes longer:

* ``solved_lut`` -- where the model ends on the true solution;
* ``wrong_lut``  -- where it ends on some other grid (a hue clearly apart from the solved ramp).

Ramps are interpolated in OKLab so neighbouring settle levels blend smoothly after anti-aliased
down-sampling, which is done in linear light.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _to_oklab(rgb255: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(np.asarray(rgb255, dtype=np.float64) / 255.0)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def _from_oklab(lab: np.ndarray) -> np.ndarray:
    lin = (np.asarray(lab) @ _M2_INV.T) ** 3 @ _M1_INV.T
    return np.rint(linear_to_srgb(lin) * 255.0).astype(np.uint8)


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def ramp(stops: list[tuple[float, tuple[int, int, int]]], n: int = 256) -> np.ndarray:
    """(n, 3) uint8 LUT through colour stops [(position 0..1, (r, g, b))], interpolated in OKLab."""
    pos = np.array([p for p, _ in stops], dtype=np.float64)
    lab = _to_oklab(np.array([c for _, c in stops], dtype=np.float64))
    t = np.linspace(0.0, 1.0, n)
    out = np.stack([np.interp(t, pos, lab[:, k]) for k in range(3)], axis=-1)
    return _from_oklab(out)


def _table(stops, t_lo: int, t_hi: int) -> np.ndarray:
    """256-entry table indexed by settle time; times outside [t_lo, t_hi] repeat the end colours."""
    base = ramp(stops, t_hi - t_lo + 1)
    idx = np.clip(np.arange(256), t_lo, t_hi) - t_lo
    return base[idx]


@dataclass(frozen=True)
class Colourway:
    name: str
    t_lo: int
    t_hi: int
    solved_lut: np.ndarray
    wrong_lut: np.ndarray
    censored: tuple[int, int, int]
    solved_stops: tuple
    wrong_stops: tuple

    def with_range(self, t_lo: int, t_hi: int) -> "Colourway":
        return replace(self, t_lo=t_lo, t_hi=t_hi, solved_lut=_table(self.solved_stops, t_lo, t_hi),
                       wrong_lut=_table(self.wrong_stops, t_lo, t_hi))


def _make(name, solved, wrong, censored, t_lo=2, t_hi=12) -> Colourway:
    solved = tuple((p, hex_rgb(c)) for p, c in solved)
    wrong = tuple((p, hex_rgb(c)) for p, c in wrong)
    return Colourway(name, t_lo, t_hi, _table(solved, t_lo, t_hi), _table(wrong, t_lo, t_hi), censored, solved, wrong)


COLOURWAYS: dict[str, Colourway] = {
    # a refined take on loopscape's look: deep indigo -> cobalt -> teal -> mint; stuck answers burn warm
    "loopscape": _make(
        "loopscape",
        [(0.00, "#0b0620"), (0.10, "#1a0f4d"), (0.22, "#2a2a8f"), (0.38, "#1561a8"), (0.52, "#0f8ea6"),
         (0.66, "#2fb59a"), (0.80, "#7fdc9b"), (1.00, "#c9f7e0")],
        [(0.00, "#1c0a1f"), (0.25, "#5c1a3d"), (0.50, "#b3304f"), (0.75, "#f0722b"), (1.00, "#ffd27a")],
        (255, 255, 255),
    ),
    # the series palette: page background -> slate -> ember -> amber #f59e0b -> cream; stuck answers go ice-cyan
    "series": _make(
        "series",
        [(0.00, "#0e1117"), (0.12, "#151b2b"), (0.28, "#232a40"), (0.44, "#4a3a3c"), (0.58, "#8a5a22"),
         (0.72, "#d98a0f"), (0.84, "#f59e0b"), (0.93, "#fcc95a"), (1.00, "#fff1c9")],
        [(0.00, "#0d1a1f"), (0.30, "#0f4c5c"), (0.60, "#1fb8d1"), (1.00, "#d6f6ff")],
        (255, 244, 214),
    ),
}


def colourize(settle: np.ndarray, flags: np.ndarray, cw: Colourway) -> np.ndarray:
    """(H, W) settle times + flag words -> (H, W, 3) uint8."""
    t = np.asarray(settle).astype(np.int64)
    solved = (np.asarray(flags) & 1) > 0
    censored = (np.asarray(flags) & 4) > 0
    img = np.where(solved[..., None], cw.solved_lut[t], cw.wrong_lut[t])
    img[censored] = cw.censored
    return img.astype(np.uint8)


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Area-average ``factor`` x ``factor`` blocks in linear light (sRGB uint8 in and out)."""
    h, w, _ = img.shape
    if h % factor or w % factor:
        raise ValueError(f"image {h}x{w} is not divisible by {factor}")
    lin = srgb_to_linear(img.astype(np.float64) / 255.0)
    lin = lin.reshape(h // factor, factor, w // factor, factor, 3).mean(axis=(1, 3))
    return np.rint(linear_to_srgb(lin) * 255.0).astype(np.uint8)
