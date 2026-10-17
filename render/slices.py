"""2-D slices through the model's initial-latent space, exactly as the loopscape notebook builds them.

The notebook (demos/basins.ipynb, "EqR on Sudoku") takes a random orthonormal 2-plane (u, v) through
the flattened (97 x 512) latent space and starts the model at z_H = a*u + b*v, z_L = 0 for a grid of
(a, b) in [-1, 1]^2.  Everything here reproduces that arithmetic in float32 ("notebook mode"); a float64
path is provided for deep zooms where neighbouring pixels differ below float32 resolution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

SHAPE = (97, 512)  # 16 register tokens + 81 cells, hidden size 512


def plane(shape=SHAPE, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random orthonormal 2-plane through the full latent space (notebook ``plane``)."""
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(int(np.prod(shape)), 2))
    return q[:, 0].reshape(shape).numpy().copy(), q[:, 1].reshape(shape).numpy().copy()


def save_plane(path, seed: int = 0, shape=SHAPE) -> None:
    """Freeze a plane to a file so every machine (Mac, cluster) uses bit-identical (u, v)."""
    u, v = plane(shape, seed)
    np.savez(path, u=u, v=v, seed=np.int64(seed))


def load_plane(path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["u"].astype(np.float32), z["v"].astype(np.float32)


def get_plane(seed: int = 0, path=None, shape=SHAPE) -> tuple[np.ndarray, np.ndarray]:
    """The frozen plane file if it exists, else the notebook construction for ``seed``."""
    from pathlib import Path

    if path is not None and Path(path).exists():
        return load_plane(path)
    return plane(shape, seed)


def axis(n: int) -> np.ndarray:
    """The notebook's pixel axis: float32 ``linspace(-1, 1, n)``."""
    return torch.linspace(-1, 1, n).numpy()


def mesh(n: int) -> np.ndarray:
    """(n*n, 2) float32 plane coordinates, row-major with x varying fastest (notebook ``mesh``)."""
    g = axis(n)
    x, y = np.meshgrid(g, g, indexing="xy")
    return np.stack([x.ravel(), y.ravel()], axis=-1)


def coords_from_indices(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """Plane coordinates of pixels (row, col) of an n x n notebook-style render."""
    g = axis(n)
    return np.stack([g[np.asarray(cols)], g[np.asarray(rows)]], axis=-1)


def latents(coords: np.ndarray, u: np.ndarray, v: np.ndarray, precision: str = "float32") -> np.ndarray:
    """z_H = a*u + b*v for each coordinate pair; always returns float32 (B, 97, 512).

    ``precision="float32"`` replicates the notebook's float32 arithmetic bit for bit;
    ``"float64"`` forms the combination in float64 before rounding (needed for deep zoom).
    """
    if precision == "float32":
        c = coords.astype(np.float32, copy=False)
        return c[:, 0, None, None] * u + c[:, 1, None, None] * v
    if precision == "float64":
        c = coords.astype(np.float64, copy=False)
        z = c[:, 0, None, None] * u.astype(np.float64) + c[:, 1, None, None] * v.astype(np.float64)
        return z.astype(np.float32)
    raise ValueError(f"unknown precision {precision!r}")


def n_distinct_latents(z: np.ndarray) -> int:
    """How many different latents survive float32 rounding (detects zoom-depth collapse)."""
    flat = np.ascontiguousarray(z.reshape(z.shape[0], -1))
    return len({row.tobytes() for row in flat})


@dataclass(frozen=True)
class Window:
    """Square window of the (a, b) plane sampled on an n x n grid that includes its edges."""

    center: tuple[float, float]
    half_width: float
    n: int

    def axis(self) -> np.ndarray:
        # float64 unit axis; the float32 notebook axis is only used for the base [-1, 1] window
        t = np.linspace(-1.0, 1.0, self.n, dtype=np.float64)
        return t

    def coords(self) -> np.ndarray:
        t = self.axis()
        x = self.center[0] + self.half_width * t
        y = self.center[1] + self.half_width * t
        xx, yy = np.meshgrid(x, y, indexing="xy")
        return np.stack([xx.ravel(), yy.ravel()], axis=-1)

    def zoomed(self, factor: float) -> "Window":
        return Window(center=self.center, half_width=self.half_width / factor, n=self.n)

    def octave(self, k: int) -> "Window":
        """Zoom level k = magnification 2**k about the same centre."""
        return self.zoomed(2.0 ** k)
