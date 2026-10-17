"""Per-pixel outcome extraction from decoded traces, and the global attractor dictionary."""
import json

import numpy as np

from render import outcomes as O
from render import sudoku as S

PUZZLE = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
SOL = S.pack([SOLUTION])[0].astype(np.int8)


def grid_with(*wrong):
    """The true solution with some cells corrupted: wrong = (cell, value) pairs."""
    g = SOL.copy()
    for cell, val in wrong:
        g[cell] = val
    return g


def trace(*grids):
    return np.stack(grids)[None, ...]  # (1, T, 81)


def test_solved_from_first_loop():
    t = trace(*[SOL] * 6)
    r = O.analyze(t, PUZZLE, SOL)
    assert r.solved[0] and r.valid[0] and r.n_wrong[0] == 0
    assert r.settle_nb[0] == 0 and r.settle_last[0] == 0
    assert r.n_changes[0] == 0 and r.n_distinct[0] == 1 and not r.censored[0]


def test_settles_after_two_loops():
    bad1, bad2 = grid_with((0, 9)), grid_with((0, 9), (1, 9))
    t = trace(bad1, bad2, SOL, SOL, SOL, SOL)
    r = O.analyze(t, PUZZLE, SOL)
    assert r.settle_nb[0] == 2 and r.settle_last[0] == 2
    assert r.solved[0] and r.n_changes[0] == 2 and r.n_distinct[0] == 3


def test_stuck_in_wrong_answer_is_not_solved_and_counts_wrong_cells():
    wrong = grid_with((10, 9), (20, 9), (30, 9))
    t = trace(SOL, SOL, wrong, wrong, wrong, wrong)
    r = O.analyze(t, PUZZLE, SOL)
    assert not r.solved[0] and r.n_wrong[0] == 3
    assert not r.valid[0]  # three repeated digits somewhere
    assert r.settle_nb[0] == 2 and r.settle_last[0] == 2


def test_valid_sudoku_that_breaks_a_given_is_valid_but_not_solved():
    # swap digit roles 1<->2 everywhere: still a valid Sudoku, but violates the givens
    swapped = np.where(SOL == 1, 2, np.where(SOL == 2, 1, SOL)).astype(np.int8)
    t = trace(*[swapped] * 6)
    r = O.analyze(t, PUZZLE, SOL)
    assert r.valid_sudoku[0] and not r.solved[0] and not r.valid[0]


def test_leave_and_return_distinguishes_the_two_settle_definitions():
    bad = grid_with((0, 9))
    t = trace(SOL, bad, SOL, SOL, SOL, SOL)  # right, wrong, right again
    r = O.analyze(t, PUZZLE, SOL)
    # notebook: T - (#steps equal to the final grid) = 6 - 5 = 1
    assert r.settle_nb[0] == 1
    # last-change definition: the grid last differed at step index 1 -> settled after 2 loops
    assert r.settle_last[0] == 2


def test_still_changing_at_the_cap_is_censored():
    a, b = grid_with((0, 9)), grid_with((0, 8))
    t = trace(a, a, a, a, a, b)
    r = O.analyze(t, PUZZLE, SOL)
    assert r.censored[0]
    assert r.settle_last[0] == 5  # T-1 loops, the maximum


def test_pad_and_garbage_cells_are_never_valid():
    g = SOL.copy()
    g[5] = -1
    r = O.analyze(trace(*[g] * 4), PUZZLE, SOL)
    assert not r.valid[0] and not r.solved[0] and r.n_wrong[0] == 1


def test_batch_shapes_and_dtypes():
    t = np.stack([np.stack([SOL] * 6), np.stack([grid_with((3, 9))] * 6)])  # (2, 6, 81)
    r = O.analyze(t, PUZZLE, SOL)
    assert r.final.shape == (2, 81) and r.final.dtype == np.int8
    assert r.settle_nb.dtype == np.uint8 and r.solved.tolist() == [True, False]


# ---- attractor dictionary ------------------------------------------------------------


def test_global_dictionary_orders_solution_first_then_by_count():
    a, b, c = grid_with((0, 9)), grid_with((1, 9)), grid_with((2, 9))
    # tile 1 saw SOL x5, a x2 ; tile 2 saw b x7, a x4, SOL x1 ; tile 3 saw c x1
    tiles_u = [np.stack([a, SOL]), np.stack([b, SOL, a]), np.stack([c])]
    tiles_n = [np.array([2, 5]), np.array([7, 1, 4]), np.array([1])]
    d = O.build_dictionary(tiles_u, tiles_n, SOL)
    assert d.grids.shape == (4, 81)
    assert (d.grids[0] == SOL).all() and d.counts.tolist() == [6, 7, 6, 1][:1] + sorted([7, 6, 1], reverse=True)
    # id 0 is the solution with its merged count; others by descending count
    assert d.counts[0] == 6
    assert d.counts[1:].tolist() == [7, 6, 1]
    assert (d.grids[1] == b).all() and (d.grids[2] == a).all() and (d.grids[3] == c).all()
    # per-tile maps send local indices to global ids
    assert d.id_maps[0].tolist() == [2, 0] and d.id_maps[1].tolist() == [1, 0, 2] and d.id_maps[2].tolist() == [3]


def test_dictionary_keeps_solution_slot_even_if_never_reached():
    a = grid_with((0, 9))
    d = O.build_dictionary([np.stack([a])], [np.array([3])], SOL)
    assert (d.grids[0] == SOL).all() and d.counts[0] == 0 and d.counts[1] == 3


def test_dictionary_is_deterministic_under_ties_and_tile_order():
    a, b = grid_with((0, 9)), grid_with((1, 9))
    d1 = O.build_dictionary([np.stack([a]), np.stack([b])], [np.array([2]), np.array([2])], SOL)
    d2 = O.build_dictionary([np.stack([b]), np.stack([a])], [np.array([2]), np.array([2])], SOL)
    assert (d1.grids == d2.grids).all()


def test_dictionary_json_records():
    a = grid_with((0, 9))
    d = O.build_dictionary([np.stack([SOL, a])], [np.array([9, 1])], SOL)
    recs = O.dictionary_records(d, PUZZLE)
    json.dumps(recs)  # must be JSON-serialisable
    assert recs[0] == {"id": 0, "grid": SOLUTION, "count": 9, "valid": True, "solved": True, "n_wrong": 0}
    assert recs[1]["grid"] == "9" + SOLUTION[1:] and recs[1]["count"] == 1
    assert recs[1]["valid"] is False and recs[1]["solved"] is False and recs[1]["n_wrong"] == 1
