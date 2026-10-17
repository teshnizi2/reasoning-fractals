"""Early exit for rollouts: stop computing a trajectory once nothing about it can change any more.

A pixel is *retired* at a checkpoint loop s if (a) its decoded grid was identical over the last ``m`` loops
and (b) its latent moved less than ``tol`` (max |z_H(s) - z_H(s-1)|) in loop s.  Its remaining loops are
then filled with the last decoded grid.  ``simulate`` replays a policy on full rollouts so its mistakes can be
counted exactly before it is ever used to skip work.
"""
from __future__ import annotations

import numpy as np


def retire_mask(last_grids: np.ndarray, resid: np.ndarray, tol: float, solution: np.ndarray | None = None) -> np.ndarray:
    """last_grids (B, m, 81) decoded grids of the last m loops; resid (B,) latent change in the last loop.

    With ``solution`` a pixel may only retire while sitting on the true solution (a plateau on a wrong grid is
    the signature of a saddle the trajectory may still leave)."""
    same = (last_grids == last_grids[:, -1:, :]).all(axis=(1, 2))
    if solution is not None:
        same &= (last_grids[:, -1, :] == np.asarray(solution)[None, :]).all(axis=1)
    return same & (np.asarray(resid) <= tol)


def simulate(traces: np.ndarray, resid: np.ndarray, m: int, tol: float, checkpoints: tuple[int, ...],
             solution: np.ndarray | None = None) -> dict:
    """Replay a retirement policy on complete rollouts.

    traces (B, T, 81) decoded grids; resid (B, T) per-loop latent change.  ``checkpoints`` are 1-based loop
    counts at which the policy may fire.  Returns where each pixel retired (T if never), the traces the
    policy would have produced, the mean loops computed and how many pixels it got wrong.
    """
    b, t, _ = traces.shape
    retired_at = np.full(b, t, dtype=np.int64)
    for s in sorted(checkpoints):
        if s < m or s > t:
            continue
        open_ = retired_at == t
        if not open_.any():
            break
        idx = np.flatnonzero(open_)
        fire = retire_mask(traces[idx, s - m : s, :], resid[idx, s - 1], tol, solution)
        retired_at[idx[fire]] = s
    predicted = traces.copy()
    for k in np.flatnonzero(retired_at < t):
        predicted[k, retired_at[k] :, :] = traces[k, retired_at[k] - 1, :]
    n_wrong = int((predicted != traces).any(axis=(1, 2)).sum())
    mean_loops = float(retired_at.mean())
    return dict(retired_at=retired_at, predicted=predicted, mean_loops=mean_loops,
                saving=1.0 - mean_loops / t, n_wrong=n_wrong)
