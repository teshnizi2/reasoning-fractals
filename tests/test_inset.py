"""Mini Sudoku inset for the video: givens, model digits, wrong cells, loop ticks."""
import numpy as np

from render import inset as IN

Q = "2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3..."
A = "236184795974325618851697432769538124583412967142769583327956841498271356615843279"


def frame():
    return np.zeros((400, 400, 3), dtype=np.uint8)


def test_inset_draws_inside_its_box_only():
    f = frame()
    out = IN.draw_sudoku(f, grid=A, puzzle=Q, solution=A, xy=(50, 60), size=180, loop=24, settle=7, title="A")
    assert out.shape == f.shape and out is not f and f.sum() == 0
    ys, xs = np.nonzero(out.sum(-1))
    assert xs.min() >= 40 and xs.max() <= 250 and ys.min() >= 40 and ys.max() <= 300


def test_wrong_cells_are_amber_and_correct_cells_are_not():
    wrong = list(A)
    wrong[1] = "9"  # cell 1 is empty in the puzzle and its true digit is 3
    out = IN.draw_sudoku(frame(), grid="".join(wrong), puzzle=Q, solution=A, xy=(50, 60), size=180, loop=24, settle=7)
    out_ok = IN.draw_sudoku(frame(), grid=A, puzzle=Q, solution=A, xy=(50, 60), size=180, loop=24, settle=7)
    amber = np.all(np.abs(out.astype(int) - np.array([245, 158, 11])) < 40, axis=-1)
    amber_ok = np.all(np.abs(out_ok.astype(int) - np.array([245, 158, 11])) < 40, axis=-1)
    assert amber.sum() > amber_ok.sum() + 100  # a filled amber cell, not just the tick bar


def test_cell_geometry_has_nine_by_nine_cells_filling_the_box():
    cell = IN.cell_box((10, 20), 180, 4, 7)
    x0, y0, x1, y1 = cell
    assert (x1 - x0) == 20 and (y1 - y0) == 20 and x0 == 10 + 7 * 20 and y0 == 20 + 4 * 20 + IN.HEADER
