"""Run an engine over many pixels in batches and return per-pixel outcome arrays."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from . import outcomes as O
from . import slices as SL

_FIELDS = (
    "final", "settle_nb", "settle_last", "solved", "valid_sudoku", "valid",
    "n_wrong", "n_changes", "n_distinct", "censored",
)


def run_pixels(
    engine,
    coords: np.ndarray,
    plane: tuple[np.ndarray, np.ndarray],
    puzzle: str,
    solution: np.ndarray,
    batch: int = 96,
    precision: str = "float32",
    progress: Callable[[int, int, float], None] | None = None,
    keep_traces: bool = False,
    early_m: int | None = None,
    base: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Roll out ``engine`` from z_H = a*u + b*v (z_L = 0) for every coordinate pair.

    Returns a dict of per-pixel arrays (see ``outcomes.Outcomes``), plus ``seconds`` (wall time of the
    rollouts) and, if ``keep_traces``, the decoded traces (N, T, 81) int8.  With ``early_m`` the engine's
    early-exit rollout retires pixels that sit on the true solution for that many loops; ``retired_at`` (N,)
    then reports how many loops each pixel was actually computed for.  ``base`` = (z0_H, z0_L) starts every pixel
    from that point plus the plane offset (z_H = z0_H + a*u + b*v, z_L = z0_L) instead of the notebook's origin.
    """
    u, v = plane
    n = len(coords)
    parts: dict[str, list] = {k: [] for k in _FIELDS}
    traces_all: list[np.ndarray] = []
    retired_all: list[np.ndarray] = []
    t0 = time.time()
    for s in range(0, n, batch):
        z = SL.latents(coords[s : s + batch], u, v, precision=precision)
        z_l = None
        if base is not None:
            z = (z + base[0][None]).astype(np.float32)
            z_l = np.broadcast_to(base[1][None], z.shape).astype(np.float32)
        if early_m:
            tr, retired = engine.rollout_early(z, z_l, solution=solution, m=early_m)
            retired_all.append(retired)
        else:
            tr = engine.rollout(z) if z_l is None else engine.rollout(z, z_l)
        res = O.analyze(tr, puzzle, solution)
        for k in _FIELDS:
            parts[k].append(getattr(res, k))
        if keep_traces:
            traces_all.append(tr)
        if progress is not None:
            progress(min(s + batch, n), n, time.time() - t0)
    out = {k: np.concatenate(parts[k]) for k in _FIELDS}
    out["seconds"] = time.time() - t0
    if early_m:
        out["retired_at"] = np.concatenate(retired_all)
    if keep_traces:
        out["traces"] = np.concatenate(traces_all)
    return out
