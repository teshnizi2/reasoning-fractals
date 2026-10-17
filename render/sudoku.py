"""Sudoku ground truth: classical solver, validity test, compact grid packing.

Grids are 81-character strings (row-major, '1'-'9', blank '.' or '0') on the way in, and
uint8 arrays of shape (N, 81) with 0 = blank once packed.  Decoded model cells may also be
-1 (pad token) or >= 10; ``cells_to_str`` renders those as '?'.
"""
from __future__ import annotations

import numpy as np

_ROW = [r for r in range(9) for _ in range(9)]
_COL = [c for _ in range(9) for c in range(9)]
_BOX = [(r // 3) * 3 + c // 3 for r, c in zip(_ROW, _COL)]
_PEERS = [
    [j for j in range(81) if j != i and (_ROW[j] == _ROW[i] or _COL[j] == _COL[i] or _BOX[j] == _BOX[i])]
    for i in range(81)
]
_FULL = 0x1FF  # candidate bitmask 1..9


def _parse(puzzle: str) -> list[int] | None:
    s = "".join(puzzle.split())
    if len(s) != 81:
        raise ValueError(f"puzzle must have 81 cells, got {len(s)}")
    vals = [0 if ch in "._0" else int(ch) for ch in s]
    for i, v in enumerate(vals):  # reject contradictory givens
        if v and any(vals[j] == v for j in _PEERS[i]):
            return None
    return vals


def _search(vals: list[int], limit: int, out: list[list[int]]) -> int:
    """Count solutions (up to ``limit``), appending found grids to ``out``."""
    cand = [_FULL] * 81
    for i, v in enumerate(vals):
        if v:
            for j in _PEERS[i]:
                cand[j] &= ~(1 << (v - 1))
    count = 0

    def rec(vals: list[int], cand: list[int]) -> None:
        nonlocal count
        if count >= limit:
            return
        best, best_n = -1, 10
        for i in range(81):
            if vals[i] == 0:
                n = bin(cand[i]).count("1")
                if n < best_n:
                    best, best_n = i, n
                    if n <= 1:
                        break
        if best < 0:
            count += 1
            out.append(vals[:])
            return
        if best_n == 0:
            return
        m = cand[best]
        while m and count < limit:
            bit = m & -m
            m ^= bit
            d = bit.bit_length()
            saved = [(j, cand[j]) for j in _PEERS[best]]
            vals[best] = d
            for j in _PEERS[best]:
                cand[j] &= ~bit
            rec(vals, cand)
            vals[best] = 0
            for j, c in saved:
                cand[j] = c

    rec(vals[:], cand)
    return count


def solve(puzzle: str) -> str | None:
    vals = _parse(puzzle)
    if vals is None:
        return None
    out: list[list[int]] = []
    _search(vals, 1, out)
    return "".join(map(str, out[0])) if out else None


def count_solutions(puzzle: str, limit: int = 2) -> int:
    vals = _parse(puzzle)
    if vals is None:
        return 0
    return _search(vals, limit, [])


def is_valid_solution(grid: str) -> bool:
    """Every row, column and 3x3 box is a permutation of 1..9."""
    if len(grid) != 81 or any(ch not in "123456789" for ch in grid):
        return False
    d = [int(ch) for ch in grid]
    want = set(range(1, 10))
    return (
        all({d[r * 9 + c] for c in range(9)} == want for r in range(9))
        and all({d[r * 9 + c] for r in range(9)} == want for c in range(9))
        and all(
            {d[(br + i) * 9 + bc + j] for i in range(3) for j in range(3)} == want
            for br in (0, 3, 6)
            for bc in (0, 3, 6)
        )
    )


def pack(grids: list[str]) -> np.ndarray:
    """81-char strings -> uint8 (N, 81); '.' and '0' both become 0."""
    arr = np.empty((len(grids), 81), dtype=np.uint8)
    for k, g in enumerate(grids):
        g = "".join(g.split())
        arr[k] = [0 if ch in "._0" else int(ch) for ch in g]
    return arr


def unpack(arr: np.ndarray) -> list[str]:
    return ["".join("." if v == 0 else str(int(v)) for v in row) for row in arr]


def cells_to_str(cells: np.ndarray) -> list[str]:
    """Decoded model cells (int8, may include -1 or >= 10) -> strings; invalid cells are '?'."""
    out = []
    for row in np.asarray(cells):
        out.append("".join("?" if (v < 0 or v > 9) else ("." if v == 0 else str(int(v))) for v in row))
    return out


def n_wrong(cells: np.ndarray, solution: np.ndarray) -> np.ndarray:
    """Number of cells that differ from the solution, per grid. ``cells`` (N, 81)."""
    return (np.asarray(cells) != np.asarray(solution)[None, :]).sum(axis=1)
