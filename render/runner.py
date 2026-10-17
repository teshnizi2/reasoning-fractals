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
) -> dict:
    """Roll out ``engine`` from z_H = a*u + b*v (z_L = 0) for every coordinate pair.

    Returns a dict of per-pixel arrays (see ``outcomes.Outcomes``), plus ``seconds`` (wall time of the
    rollouts) and, if ``keep_traces``, the decoded traces (N, T, 81) int8.
    """
    u, v = plane
    n = len(coords)
    parts: dict[str, list] = {k: [] for k in _FIELDS}
    traces_all: list[np.ndarray] = []
    t0 = time.time()
    for s in range(0, n, batch):
        z = SL.latents(coords[s : s + batch], u, v, precision=precision)
        tr = engine.rollout(z)
        res = O.analyze(tr, puzzle, solution)
        for k in _FIELDS:
            parts[k].append(getattr(res, k))
        if keep_traces:
            traces_all.append(tr)
        if progress is not None:
            progress(min(s + batch, n), n, time.time() - t0)
    out = {k: np.concatenate(parts[k]) for k in _FIELDS}
    out["seconds"] = time.time() - t0
    if keep_traces:
        out["traces"] = np.concatenate(traces_all)
    return out
