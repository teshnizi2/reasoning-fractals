"""Turn decoded rollouts into per-pixel outcomes and a global attractor dictionary.

A *trace* is the decoded 9x9 grid (81 int8 cells, -1..10 as the model emits them) after each of the T
reasoning loops.  Two settle-time definitions are kept side by side:

* ``settle_nb``   -- loopscape notebook: ``T - #{loops whose grid equals the final grid}``.  This is what
                      the upstream asset is coloured by, so it is the one used for the reproduction.
* ``settle_last`` -- "the loop after which the decoded grid never changes again": the index of the last
                      loop whose grid differs from the final one, plus one (0 if it never differs).

They agree unless a trajectory leaves the final answer and later returns to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sudoku as S

_DIGITS = np.arange(1, 10, dtype=np.int8)


@dataclass
class Outcomes:
    final: np.ndarray         # (B, 81) int8
    settle_nb: np.ndarray     # (B,) uint8
    settle_last: np.ndarray   # (B,) uint8
    solved: np.ndarray        # (B,) bool   final grid == the unique solution
    valid_sudoku: np.ndarray  # (B,) bool   rows/cols/boxes are permutations of 1..9 (givens ignored)
    valid: np.ndarray         # (B,) bool   valid Sudoku AND every given kept
    n_wrong: np.ndarray       # (B,) uint8  cells differing from the solution
    n_changes: np.ndarray     # (B,) uint8  loop-to-loop changes of the decoded grid
    n_distinct: np.ndarray    # (B,) uint8  distinct grids visited along the trace
    censored: np.ndarray      # (B,) bool   still changing at the final loop


def _valid_sudoku(final: np.ndarray) -> np.ndarray:
    b = final.shape[0]
    oh = final.reshape(b, 9, 9, 1) == _DIGITS  # (B, row, col, digit)
    rows = (oh.sum(axis=2) == 1).all(axis=(1, 2))
    cols = (oh.sum(axis=1) == 1).all(axis=(1, 2))
    boxes = (oh.reshape(b, 3, 3, 3, 3, 9).sum(axis=(2, 4)) == 1).all(axis=(1, 2, 3))
    return rows & cols & boxes


def _n_distinct(traces: np.ndarray, chunk: int = 256) -> np.ndarray:
    b, t, _ = traces.shape
    out = np.empty(b, dtype=np.uint8)
    tril = np.tril(np.ones((t, t), dtype=bool), k=-1)
    for s in range(0, b, chunk):
        x = traces[s : s + chunk]
        eq = (x[:, :, None, :] == x[:, None, :, :]).all(-1)  # (c, T, T)
        out[s : s + chunk] = (~(eq & tril).any(-1)).sum(-1)
    return out


def analyze(traces: np.ndarray, puzzle: str, solution: np.ndarray) -> Outcomes:
    traces = np.asarray(traces, dtype=np.int8)
    b, t, _ = traces.shape
    final = traces[:, -1]
    eq = (traces == final[:, None, :]).all(-1)  # (B, T)
    settle_nb = (t - eq.sum(1)).astype(np.uint8)

    neq = ~eq
    last_diff = np.where(neq.any(1), t - 1 - np.argmax(neq[:, ::-1], axis=1), -1)
    settle_last = (last_diff + 1).astype(np.uint8)

    changed = (traces[:, 1:] != traces[:, :-1]).any(-1) if t > 1 else np.zeros((b, 0), dtype=bool)
    n_changes = changed.sum(1).astype(np.uint8)
    censored = changed[:, -1] if t > 1 else np.zeros(b, dtype=bool)

    sol = np.asarray(solution, dtype=np.int8)
    solved = (final == sol).all(-1)
    valid_sudoku = _valid_sudoku(final)
    given = np.array([0 if ch in "._0" else int(ch) for ch in "".join(puzzle.split())], dtype=np.int8)
    mask = given > 0
    givens_kept = (final[:, mask] == given[mask]).all(-1)

    return Outcomes(
        final=final.copy(),
        settle_nb=settle_nb,
        settle_last=settle_last,
        solved=solved,
        valid_sudoku=valid_sudoku,
        valid=valid_sudoku & givens_kept,
        n_wrong=(final != sol).sum(-1).astype(np.uint8),
        n_changes=n_changes,
        n_distinct=_n_distinct(traces),
        censored=censored,
    )


def unique_grids(final: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(unique grids (K, 81), inverse (N,), counts (K,)) for a batch of final grids."""
    u, inv, cnt = np.unique(np.asarray(final, dtype=np.int8), axis=0, return_inverse=True, return_counts=True)
    return u, inv.reshape(-1), cnt


@dataclass
class Dictionary:
    grids: np.ndarray            # (G, 81) int8; row 0 is the true solution
    counts: np.ndarray           # (G,) int64
    id_maps: list[np.ndarray]    # per input tile: local unique index -> global id


def build_dictionary(tiles_u: list[np.ndarray], tiles_n: list[np.ndarray], solution: np.ndarray) -> Dictionary:
    """Merge tile-local unique grids into one dictionary: id 0 = solution, then by descending count."""
    sol = np.asarray(solution, dtype=np.int8)
    allg = np.concatenate([np.asarray(u, dtype=np.int8) for u in tiles_u] + [sol[None, :]], axis=0)
    alln = np.concatenate([np.asarray(n, dtype=np.int64) for n in tiles_n] + [np.zeros(1, dtype=np.int64)])
    uniq, inv = np.unique(allg, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    counts = np.bincount(inv, weights=alln, minlength=len(uniq)).astype(np.int64)

    sol_idx = int(np.flatnonzero((uniq == sol).all(-1))[0])
    rest = np.array([i for i in range(len(uniq)) if i != sol_idx], dtype=np.int64)
    order_rest = rest[np.argsort(-counts[rest], kind="stable")]  # ties keep lexicographic grid order
    order = np.concatenate([[sol_idx], order_rest]).astype(np.int64)
    new_id = np.empty(len(uniq), dtype=np.int64)
    new_id[order] = np.arange(len(uniq))

    id_maps, off = [], 0
    for u in tiles_u:
        n = len(u)
        id_maps.append(new_id[inv[off : off + n]])
        off += n
    return Dictionary(grids=uniq[order], counts=counts[order], id_maps=id_maps)


def dictionary_records(d: Dictionary, puzzle: str) -> list[dict]:
    """JSON-ready attractor dictionary: id -> 81-char grid, count, validity, wrong-cell count."""
    sol = d.grids[0]
    given = np.array([0 if ch in "._0" else int(ch) for ch in "".join(puzzle.split())], dtype=np.int8)
    mask = given > 0
    strs = S.cells_to_str(d.grids)
    valid_sud = _valid_sudoku(d.grids)
    kept = (d.grids[:, mask] == given[mask]).all(-1)
    wrong = (d.grids != sol).sum(-1)
    out = []
    for i, g in enumerate(strs):
        out.append(
            {
                "id": i,
                "grid": g,
                "count": int(d.counts[i]),
                "valid": bool(valid_sud[i] and kept[i]),
                "solved": bool(wrong[i] == 0),
                "n_wrong": int(wrong[i]),
            }
        )
    return out
