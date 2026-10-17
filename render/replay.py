"""Re-run chosen pixels on the engine that rendered them: plain run vs the stored values, plus rounding-sized nudges.

Used on the cluster (``alice/replay.py``) with the torch/CUDA engine, i.e. the same arithmetic as the tiles.  A
*plain* replay (full 24 loops) reproduces the stored settle time for nearly every pixel (batch size does not change
cuBLAS fp32 results, see RUNS row 4) and gives the loop-by-loop traces the explorer and the video show.  The
*nudged* replays multiply the latent by ``1 + eps * N(0, 1)`` (a few dozen float32 ulps at eps = 1e-6) and report
how much the settle time moves: small for pixels in flat domains, large in the speckled "dust".
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import outcomes as O
from . import slices as SL
from . import sudoku as S


def trace_strings(trace: np.ndarray) -> list[str]:
    """(T, 81) decoded loops -> T strings ('.' blank, '?' not a digit)."""
    return S.cells_to_str(trace)


def _flags(res: O.Outcomes) -> np.ndarray:
    """Same bits as the tile files: 1 solved, 2 valid, 4 censored."""
    return res.solved.astype(np.uint8) | (res.valid.astype(np.uint8) << 1) | (res.censored.astype(np.uint8) << 2)


def run_items(rollout: Callable[[np.ndarray], np.ndarray], plane: tuple[np.ndarray, np.ndarray], puzzle: str,
              solution: np.ndarray, items: list[dict], nudged: int = 3, eps: float = 1e-6, seed: int = 0,
              batch: int = 1024) -> list[dict]:
    """Replay ``items`` (dicts with ``id``, ``x``, ``y``, ``prec`` and ``stored`` = {settle, flags}); same order out.

    ``rollout(z)`` maps latents (B, 97, 512) to decoded traces (B, 24, 81) int8.  Items are grouped by latent
    precision (float32 for the notebook-mode hero, float64 for zoom windows) and sent ``batch`` at a time.
    Each result has ``settle``, ``flags``, ``trace`` (24 strings), ``nudged`` (settle of each nudged run) and
    ``equal`` (plain run == stored settle time and solved/censored flags).
    """
    u, v = plane
    rng = np.random.default_rng(seed)
    out: list[dict | None] = [None] * len(items)
    for prec in sorted({it["prec"] for it in items}):
        idx = [i for i, it in enumerate(items) if it["prec"] == prec]
        for s in range(0, len(idx), batch):
            chunk = idx[s : s + batch]
            coords = np.array([[items[i]["x"], items[i]["y"]] for i in chunk], dtype=np.float64)
            z = SL.latents(coords, u, v, precision=prec)
            tr = rollout(z)
            res = O.analyze(tr, puzzle, solution)
            flags = _flags(res)
            nud = []
            for _ in range(nudged):
                r = O.analyze(rollout(SL.perturb(z, eps, rng)), puzzle, solution)
                nud.append(r.settle_nb.astype(int))
            for k, i in enumerate(chunk):
                st = items[i]["stored"]
                equal = int(res.settle_nb[k]) == int(st["settle"]) and (int(flags[k]) & 5) == (int(st["flags"]) & 5)
                out[i] = dict(id=items[i]["id"], settle=int(res.settle_nb[k]), flags=int(flags[k]), trace=trace_strings(tr[k]),
                              nudged=[int(n[k]) for n in nud], equal=bool(equal))
    return out  # type: ignore[return-value]
