"""Sudoku helpers: classical solver (ground truth), validity, packing."""
import numpy as np
import pytest

from render import sudoku as S

WIKI_PUZZLE = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
WIKI_SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
NOTEBOOK_PUZZLE = ".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2...."


def test_solve_known_puzzle():
    assert S.solve(WIKI_PUZZLE) == WIKI_SOLUTION


def test_solve_accepts_zero_blanks():
    assert S.solve(WIKI_PUZZLE.replace(".", "0")) == WIKI_SOLUTION


def test_count_solutions_unique_and_multiple():
    assert S.count_solutions(WIKI_PUZZLE, limit=2) == 1
    assert S.count_solutions("." * 81, limit=2) == 2  # empty grid: many, capped at limit


def test_unsolvable_returns_none():
    bad = "55" + "." * 79  # two 5s in one row
    assert S.solve(bad) is None


def test_notebook_puzzle_has_a_unique_solution():
    sol = S.solve(NOTEBOOK_PUZZLE)
    assert sol is not None
    assert S.count_solutions(NOTEBOOK_PUZZLE, limit=2) == 1
    assert S.is_valid_solution(sol)
    assert all(p == "." or p == s for p, s in zip(NOTEBOOK_PUZZLE, sol))  # givens kept


def test_is_valid_solution():
    assert S.is_valid_solution(WIKI_SOLUTION)
    broken = WIKI_SOLUTION[:80] + ("1" if WIKI_SOLUTION[80] != "1" else "2")
    assert not S.is_valid_solution(broken)
    assert not S.is_valid_solution(WIKI_PUZZLE)  # has blanks


def test_pack_unpack_roundtrip():
    # blanks may be written '.' or '0' on the way in; they come back as '.'
    arr = S.pack([WIKI_SOLUTION, WIKI_PUZZLE.replace(".", "0")])
    assert arr.dtype == np.uint8 and arr.shape == (2, 81)
    assert S.unpack(arr) == [WIKI_SOLUTION, WIKI_PUZZLE]


def test_pack_clips_model_garbage_cells_to_digits_or_zero():
    # a decoded cell of -1 (pad token) or 10+ must not crash or wrap
    arr = S.cells_to_str(np.array([[-1, 0, 5, 9, 10] + [1] * 76], dtype=np.int8))
    assert arr[0][:5] == "?.59?"


def test_hamming_to_solution_counts_wrong_cells():
    a = np.frombuffer(WIKI_SOLUTION.encode(), dtype=np.uint8) - ord("0")
    b = a.copy()
    b[[3, 40]] = (b[[3, 40]] % 9) + 1
    assert S.n_wrong(b[None, :], a)[0] == 2
    assert S.n_wrong(a[None, :], a)[0] == 0
