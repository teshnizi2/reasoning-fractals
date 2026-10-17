"""Basin metrics (basin entropy, uncertainty exponent) via loopscape.metrics, plus outcome summaries.

Method, as in the paper (Lai et al., arXiv:2609.04963, App. B/D) and loopscape:
* basin entropy S_b: split the label field into non-overlapping 5 x 5 boxes, Shannon entropy (natural log) of
  the labels in each box, mean over boxes; the boundary entropy S_bb averages only boxes with >1 label;
  S_bb > ln 2 is the criterion for a fractal-like boundary (Daza et al. 2016);
* uncertainty exponent alpha: fraction f(eps) of pixel pairs separated by eps = 1..5 pixels (horizontal,
  vertical, both diagonals) that carry different labels, f ~ eps^alpha by a log-log fit; the boundary
  dimension is 2 - alpha (McDonald et al. 1985).
Labels are equality classes, so the settle-time field and the decoded asset colours (11 levels) are both valid.
"""
from __future__ import annotations

import numpy as np


def label_metrics(labels: np.ndarray, box_size: int = 5) -> dict:
    from loopscape.metrics import basin_entropy, uncertainty_exponent

    labels = np.asarray(labels)
    be = basin_entropy(labels, box_size=box_size)
    ue = uncertainty_exponent(labels, box_size=box_size)
    return dict(
        shape=list(labels.shape),
        n_labels=int(len(np.unique(labels))),
        basin_entropy=float(be["basin_entropy"]),
        boundary_basin_entropy=float(be["boundary_basin_entropy"]),
        boundary_box_fraction=float(be["boundary_box_fraction"]),
        log2_criterion=bool(be["log2_criterion"]),
        uncertainty_exponent=float(ue["uncertainty_exponent"]),
        boundary_dimension=float(ue["boundary_dimension"]),
        uncertainty_r2=float(ue["r_squared"]),
        uncertainty_fractions=[float(f) for f in ue["uncertainty_fractions"]],
        box_size=box_size,
    )


def multi_scale(labels: np.ndarray, strides=(1, 2, 4, 8), box_size: int = 5) -> dict:
    """Metrics of the same field sampled every ``stride`` pixels (a coarser render of the same plane)."""
    return {s: label_metrics(np.asarray(labels)[::s, ::s], box_size) for s in strides}


def outcome_summary(settle: np.ndarray, flags: np.ndarray, gid: np.ndarray, max_loops: int = 24) -> dict:
    settle, flags, gid = np.asarray(settle), np.asarray(flags), np.asarray(gid)
    solved = (flags & 1) > 0
    uniq = np.unique(gid)
    return dict(
        pixels=int(settle.size),
        frac_solved=float(solved.mean()),
        frac_censored=float(((flags & 4) > 0).mean()),
        n_outcomes=int(len(uniq)),
        n_wrong_outcomes=int(len(np.unique(gid[~solved]))) if (~solved).any() else 0,
        settle_hist=np.bincount(settle.ravel(), minlength=max_loops + 1).tolist(),
        mean_settle=float(settle.mean()),
        max_settle=int(settle.max()),
    )


def adjacent_gap_stats(settle: np.ndarray, thresholds=(1, 2, 5, 10)) -> dict:
    """How different are neighbouring starts?  Distribution of |settle(p) - settle(q)| over horizontal and vertical
    neighbour pairs: the fraction of pairs differing by at least each threshold, and the largest gap."""
    s = np.asarray(settle).astype(np.int64)
    gaps = np.concatenate([np.abs(np.diff(s, axis=1)).ravel(), np.abs(np.diff(s, axis=0)).ravel()])
    return dict(n_pairs=int(gaps.size), max_gap=int(gaps.max()), mean_gap=float(gaps.mean()),
                frac_ge={str(t): float((gaps >= t).mean()) for t in thresholds})
