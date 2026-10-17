"""Batching + analysis glue: pixels in, per-pixel outcome arrays out (stub engine, no model)."""
import numpy as np

from render import runner as RU
from render import sudoku as S

PUZZLE = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
SOL = S.pack([SOLUTION])[0].astype(np.int8)


class StubEngine:
    """Reads a (latent[0,0]) and b (latent[0,1]) and answers: solved after int(a) loops, wrong cells = int(b)."""

    steps = 6

    def __init__(self):
        self.batches = []

    def rollout(self, z_h, z_l=None):
        self.batches.append(len(z_h))
        out = np.empty((len(z_h), self.steps, 81), dtype=np.int8)
        for k, z in enumerate(z_h):
            settle, n_bad = int(round(z[0, 0])), int(round(z[0, 1]))
            for t in range(self.steps):
                g = SOL.copy()
                if t < settle:
                    g[:] = 0
                elif n_bad:
                    g[:n_bad] = (g[:n_bad] % 9) + 1  # a wrong final answer
                out[k, t] = g
        return out


def stub_plane():
    u = np.zeros((97, 512), dtype=np.float32)
    v = np.zeros((97, 512), dtype=np.float32)
    u[0, 0] = 1.0
    v[0, 1] = 1.0
    return u, v


def test_run_pixels_returns_one_row_per_pixel_in_order_across_batches():
    eng = StubEngine()
    coords = np.array([[0, 0], [1, 0], [2, 0], [3, 1], [4, 2], [0, 3], [5, 0]], dtype=np.float32)
    res = RU.run_pixels(eng, coords, stub_plane(), PUZZLE, SOL, batch=3)
    assert eng.batches == [3, 3, 1]
    assert res["settle_nb"].tolist() == [0, 1, 2, 3, 4, 0, 5]
    assert res["solved"].tolist() == [True, True, True, False, False, False, True]
    assert res["n_wrong"].tolist() == [0, 0, 0, 1, 2, 3, 0]
    assert res["final"].shape == (7, 81) and res["final"].dtype == np.int8
    assert res["seconds"] > 0


def test_run_pixels_reports_progress_and_supports_float64_latents():
    eng = StubEngine()
    coords = np.array([[1.0, 0.0], [2.0, 0.0]])
    seen = []
    res = RU.run_pixels(eng, coords, stub_plane(), PUZZLE, SOL, batch=1, precision="float64",
                        progress=lambda done, total, dt: seen.append((done, total)))
    assert seen == [(1, 2), (2, 2)] and res["settle_nb"].tolist() == [1, 2]


class StubEarlyEngine(StubEngine):
    """Adds rollout_early: identical traces, plus a retirement loop count per pixel."""

    def __init__(self):
        super().__init__()
        self.early_calls = 0

    def rollout_early(self, z_h, z_l=None, solution=None, m=3, tokens=None):
        self.early_calls += 1
        tr = self.rollout(z_h, z_l)
        return tr, np.full(len(z_h), 4, dtype=np.int64)


def test_run_pixels_uses_the_early_exit_rollout_when_asked_and_reports_loops():
    eng = StubEarlyEngine()
    coords = np.array([[1, 0], [2, 0], [3, 1]], dtype=np.float32)
    res = RU.run_pixels(eng, coords, stub_plane(), PUZZLE, SOL, batch=2, early_m=3)
    assert eng.early_calls == 2 and res["settle_nb"].tolist() == [1, 2, 3]
    assert res["retired_at"].tolist() == [4, 4, 4]
    plain = RU.run_pixels(StubEarlyEngine(), coords, stub_plane(), PUZZLE, SOL, batch=2)
    assert "retired_at" not in plain
