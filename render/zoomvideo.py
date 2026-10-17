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


def timeline(fps: int, hold0: float, move: float, tail: float) -> list[float]:
    """Camera progress u in [0, 1] for every frame: parked at 0 for ``hold0`` s, travelling for ``move`` s, parked at 1 for ``tail`` s."""
    n0, n1, n2 = (int(round(v * fps)) for v in (hold0, move, tail))
    return [0.0] * n0 + [float(v) for v in np.linspace(0.0, 1.0, n1)] + [1.0] * n2


def _smooth(a: float, b: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def _smooth_arr(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def to_screen(px: float, py: float, cx: float, cy: float, zoom: float, size: tuple[int, int],
              view_width_units: float = 2.0) -> tuple[float, float]:
    """Continuous frame coordinates (x right, y down, pixel i spans [i, i+1)) of the plane point (px, py)."""
    w, h = size
    scale = w / (view_width_units / zoom)
    return (w / 2 + (px - cx) * scale, h / 2 - (py - cy) * scale)


def shifted_center(px: float, py: float, zoom: float, size: tuple[int, int], view_width_units: float,
                   offset: tuple[float, float]) -> tuple[float, float]:
    """Camera centre that shows the plane point (px, py) at ``offset`` screen pixels from the frame centre (y down)."""
    scale = size[0] / (view_width_units / zoom)
    return (px - offset[0] / scale, py + offset[1] / scale)


FEATHER = 20.0  # samples over which a finer layer fades out towards the edge of its footprint
BLOCKS_FROM, BLOCKS_TO = 3.0, 8.0  # on-screen px per sample between which smooth resampling turns into hard blocks


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
                 view_width_units: float = 2.0, resample=Image.BICUBIC, background: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """One frame (height, width, 3) uint8. ``zoom`` 1 shows ``view_width_units`` of plane across the width."""
    w, h = size
    scale = w / (view_width_units / zoom)  # frame px per plane unit
    prev_pitch_px = 0.0
    yy, xx = np.mgrid[0:h, 0:w]
    px = (xx + 0.5 - w / 2) / scale + cx  # plane x of each frame pixel centre
    py = cy - (yy + 0.5 - h / 2) / scale  # plane y
    out = np.empty((h, w, 3), dtype=np.float64)
    out[:] = background
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
        # PIL maps output pixel (x, y) to input (a (x + .5) + c, a (y + .5) + f) in pixel-edge coordinates, where sample
        # k's centre is k + .5 (nearest takes the floor; bicubic subtracts .5 itself): c = c' + .5 with c' the sample
        # coordinate of the frame's left edge (pure scale + translate, so the 6 affine numbers are enough)
        a = 1.0 / (L.pitch * scale)
        c_edge = ((cx - (w / 2) / scale) - (L.center[0] - L.half_width)) / L.pitch
        top = L.center[1] + L.half_width * (L.ny - 1) / (L.nx - 1)
        f_edge = (top - (cy + (h / 2) / scale)) / L.pitch
        params = (a, 0, c_edge + 0.5, 0, a, f_edge + 0.5)
        warped = np.asarray(src.transform((w, h), Image.AFFINE, params, resample=resample, fillcolor=(0, 0, 0)), dtype=np.float64)
        hard = _smooth(BLOCKS_FROM, BLOCKS_TO, pitch_px)  # big pixels become hard-edged blocks ("one pixel = one thought")
        if hard > 0.0:
            blocks = np.asarray(src.transform((w, h), Image.AFFINE, params, resample=Image.NEAREST, fillcolor=(0, 0, 0)), dtype=np.float64)
            warped = warped * (1.0 - hard) + blocks * hard
        mask = inside.astype(np.float64) * alpha
        if i > 0:  # fade a finer layer out over its last FEATHER samples so its footprint never ends in a hard rectangle
            mask = mask * _smooth_arr(0.0, FEATHER, np.minimum.reduce([sx, L.nx - sx, sy, L.ny - sy]))
        mask = mask[..., None]
        out = out * (1 - mask) + warped * mask
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
