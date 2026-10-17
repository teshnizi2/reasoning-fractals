"""Deep-zoom compositing from nested keyframe images (no model, pure image work).

Each layer is a colour image of a window of the (a, b) plane (centre, half-width; y up, row 0 = top).  A frame is
a view (cx, cy, zoom): layers are drawn coarse to fine with a bicubic affine resample, and each finer layer is
faded in as the next-coarser one is magnified past ~1 screen pixel per sample -- the same rule the explorer uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Camera:
    """Fly-in path: log-zoom with a smoothstep ease (peak speed 1.5x the mean, about one octave per second in a
    17 s, 12-octave move); the centre drifts to the target early in the move."""

    c0: tuple[float, float]
    c1: tuple[float, float]
    zoom0: float
    zoom1: float

    def at(self, t: float) -> tuple[float, float, float]:
        t = min(1.0, max(0.0, t))
        e = t * t * (3 - 2 * t)
        zoom = float(np.exp(np.log(self.zoom0) + (np.log(self.zoom1) - np.log(self.zoom0)) * e))
        m = _smooth(0.0, 0.45, e)
        return (self.c0[0] + (self.c1[0] - self.c0[0]) * m, self.c0[1] + (self.c1[1] - self.c0[1]) * m, zoom)


def _smooth(a: float, b: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def layer_alpha(prev_pitch_px: float) -> float:
    """Opacity of a layer given the on-screen pitch (px per sample) of the next-coarser layer."""
    return _smooth(1.0, 2.3, prev_pitch_px)


@dataclass
class Layer:
    center: tuple[float, float]
    half_width: float
    img: np.ndarray  # (ny, nx, 3) uint8, y up (row 0 = top)

    @property
    def nx(self) -> int:
        return self.img.shape[1]

    @property
    def ny(self) -> int:
        return self.img.shape[0]

    @property
    def pitch(self) -> float:
        return 2 * self.half_width / (self.nx - 1)


def render_frame(layers: list[Layer], cx: float, cy: float, zoom: float, size: tuple[int, int],
                 view_width_units: float = 2.0, resample=Image.BICUBIC) -> np.ndarray:
    """One frame (height, width, 3) uint8. ``zoom`` 1 shows ``view_width_units`` of plane across the width."""
    w, h = size
    scale = w / (view_width_units / zoom)  # frame px per plane unit
    prev_pitch_px = 0.0
    yy, xx = np.mgrid[0:h, 0:w]
    px = (xx + 0.5 - w / 2) / scale + cx  # plane x of each frame pixel centre
    py = cy - (yy + 0.5 - h / 2) / scale  # plane y
    out = np.zeros((h, w, 3), dtype=np.float64)
    for i, L in enumerate(layers):
        pitch_px = L.pitch * scale
        alpha = 1.0 if i == 0 else layer_alpha(prev_pitch_px)
        prev_pitch_px = pitch_px
        if alpha <= 0.0:
            continue
        # source pixel coordinates: sample k sits at x = centre - hw + k * pitch (pixel centres)
        sx = (px - (L.center[0] - L.half_width)) / L.pitch + 0.5
        sy = ((L.center[1] + L.half_width * (L.ny - 1) / (L.nx - 1)) - py) / L.pitch + 0.5
        inside = (sx >= 0) & (sx <= L.nx) & (sy >= 0) & (sy <= L.ny)
        if not inside.any():
            continue
        src = Image.fromarray(L.img)
        # affine map output (x, y) -> input (a x + b y + c, d x + e y + f) covers pure scale + translate
        a = 1.0 / (L.pitch * scale)
        c0 = ((cx - (w / 2) / scale) - (L.center[0] - L.half_width)) / L.pitch + 0.5 + 0.5 * a
        top = L.center[1] + L.half_width * (L.ny - 1) / (L.nx - 1)
        f0 = (top - (cy + (h / 2) / scale)) / L.pitch + 0.5 + 0.5 * a
        warped = np.asarray(src.transform((w, h), Image.AFFINE, (a, 0, c0 - 0.5, 0, a, f0 - 0.5), resample=resample,
                                          fillcolor=(0, 0, 0)), dtype=np.float64)
        mask = (inside.astype(np.float64) * alpha)[..., None]
        out = out * (1 - mask) + warped * mask
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
