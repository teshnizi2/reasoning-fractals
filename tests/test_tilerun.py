"""Tiled, resumable rendering: planning, worker split, atomic tile files, budget stop, merge."""
import json

import numpy as np
import pytest

from render import slices as SL
from render import sudoku as S
from render import tilerun as TR

PUZZLE = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
SOL = S.pack([SOLUTION])[0].astype(np.int8)


class StubEngine:
    """a = latent[0,0] -> loops before the right answer (capped), b = latent[0,1] -> wrong cells."""

    steps = 6

    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def rollout(self, z_h, z_l=None):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("boom")
        out = np.empty((len(z_h), self.steps, 81), dtype=np.int8)
        for k, z in enumerate(z_h):
            settle, n_bad = min(int(round(z[0, 0])), self.steps - 1), int(round(z[0, 1]))
            for t in range(self.steps):
                g = SOL.copy()
                if t < settle:
                    g[:] = 0
                elif n_bad:
                    g[:n_bad] = (g[:n_bad] % 9) + 1
                out[k, t] = g
        return out


def stub_plane():
    u = np.zeros((97, 512), dtype=np.float32)
    v = np.zeros((97, 512), dtype=np.float32)
    u[0, 0] = 1.0
    v[0, 1] = 1.0
    return u, v


def make_spec(**kw):
    spec = dict(
        name="t", puzzle=PUZZLE, mode="window", center=[3.5, 3.5], half_width=3.5, n=8, tile=4, batch=5,
        precision="fp32", latent_precision="float64", steps=6,
    )
    spec.update(kw)
    return spec


def expected_fields(n=8):
    j, i = np.meshgrid(np.arange(n), np.arange(n))  # x = col index, y = row index
    return np.minimum(j, 5), i  # settle loops = column (capped), wrong cells = row


def test_plan_tiles_cover_every_pixel_exactly_once_with_ragged_edges():
    tiles = TR.plan_tiles(n=10, tile=4)
    cover = np.zeros((10, 10), dtype=int)
    for t in tiles:
        cover[t.r0 : t.r0 + t.h, t.c0 : t.c0 + t.w] += 1
    assert (cover == 1).all() and len(tiles) == 9
    assert [t.index for t in tiles] == list(range(9))
    assert (tiles[2].h, tiles[2].w) == (4, 2) and (tiles[8].h, tiles[8].w) == (2, 2)


def test_workers_partition_the_tiles_round_robin():
    tiles = TR.plan_tiles(n=16, tile=4)
    parts = [TR.tiles_for_worker(tiles, w, 3) for w in range(3)]
    flat = sorted(t.index for p in parts for t in p)
    assert flat == list(range(16)) and all(len(p) in (5, 6) for p in parts)


def test_window_pixel_coords_match_the_window_class():
    spec = make_spec(n=9, tile=4, center=[0.25, -0.5], half_width=0.01)
    full = SL.Window((0.25, -0.5), 0.01, 9).coords().reshape(9, 9, 2)
    t = TR.plan_tiles(9, 4)[4]  # middle tile
    got = TR.pixel_coords(spec, t).reshape(t.h, t.w, 2)
    np.testing.assert_array_equal(got, full[t.r0 : t.r0 + t.h, t.c0 : t.c0 + t.w])


def test_asset_mode_uses_notebook_axis_at_stride_multiples():
    spec = make_spec(mode="asset", n=500, n_asset=1000, stride=2, tile=250)
    t = TR.plan_tiles(500, 250)[1]  # rows 0..249, cols 250..499
    got = TR.pixel_coords(spec, t).reshape(t.h, t.w, 2)
    ax = SL.axis(1000)
    assert got.dtype == np.float32
    assert got[0, 0, 0] == ax[2 * 250] and got[0, 0, 1] == ax[0] and got[3, 2, 1] == ax[2 * 3]


def run_all(tmp_path, spec=None, **kw):
    spec = spec or make_spec()
    eng = StubEngine()
    stats = TR.run_tiles(eng, spec, tmp_path, plane=stub_plane(), solution=SOL, **kw)
    return spec, eng, stats


def test_run_tiles_then_merge_reproduces_the_full_field(tmp_path):
    spec, eng, stats = run_all(tmp_path)
    assert stats["tiles_done"] == 4 and len(list((tmp_path / "tiles").glob("t_*.npz"))) == 4
    f = TR.merge(spec, tmp_path, solution=SOL)
    settle, wrong = expected_fields()
    np.testing.assert_array_equal(f["settle_nb"], settle)
    np.testing.assert_array_equal(f["n_wrong"], wrong)
    assert f["settle_nb"].dtype == np.uint8 and f["gid"].dtype == np.uint32
    assert ((f["flags"] & 1) == (wrong == 0)).all()  # bit0 = solved


def test_merge_builds_a_global_dictionary_with_solution_first(tmp_path):
    spec, _, _ = run_all(tmp_path)
    f = TR.merge(spec, tmp_path, solution=SOL)
    d = f["dictionary"]
    assert d[0]["grid"] == SOLUTION and d[0]["solved"] and d[0]["id"] == 0
    # one dictionary entry per wrong-cell count 0..7 (rows), each reached by one full row of 8 pixels
    assert sorted(r["n_wrong"] for r in d) == list(range(8))
    assert all(r["count"] == 8 for r in d) and sum(r["count"] for r in d) == 64
    # every pixel's gid points at a dictionary grid equal to what that pixel ended on
    settle, wrong = expected_fields()
    for r in range(8):
        for c in range(8):
            assert d[f["gid"][r, c]]["n_wrong"] == wrong[r, c]


def test_run_tiles_is_resumable_and_does_not_redo_finished_tiles(tmp_path):
    spec = make_spec()
    eng1 = StubEngine()
    TR.run_tiles(eng1, spec, tmp_path, plane=stub_plane(), solution=SOL, max_tiles=2)
    assert len(list((tmp_path / "tiles").glob("t_*.npz"))) == 2
    eng2 = StubEngine()
    stats = TR.run_tiles(eng2, spec, tmp_path, plane=stub_plane(), solution=SOL)
    assert stats["tiles_skipped"] == 2 and stats["tiles_done"] == 2
    assert eng2.calls == 2 * 4  # 2 tiles x ceil(16 px / batch 5) = 2 x 4 rollout calls


def test_an_interrupted_tile_leaves_no_partial_file(tmp_path):
    spec = make_spec()
    with pytest.raises(RuntimeError):
        TR.run_tiles(StubEngine(fail_on_call=6), spec, tmp_path, plane=stub_plane(), solution=SOL)
    files = sorted(p.name for p in (tmp_path / "tiles").iterdir())
    assert files == ["t_000_000.npz"]  # tile 0 finished (4 calls); tile 1 died mid-way and left nothing


def test_time_budget_stops_after_the_current_tile(tmp_path):
    spec = make_spec()
    ticks = iter(range(0, 1000, 60))  # every clock() call advances one minute
    stats = TR.run_tiles(StubEngine(), spec, tmp_path, plane=stub_plane(), solution=SOL,
                         max_minutes=2.5, clock=lambda: next(ticks))
    assert 1 <= stats["tiles_done"] < 4 and stats["stopped_for_budget"] is True


def test_worker_only_renders_its_own_tiles(tmp_path):
    spec = make_spec()
    stats = TR.run_tiles(StubEngine(), spec, tmp_path, plane=stub_plane(), solution=SOL, worker=1, workers=2)
    names = sorted(p.name for p in (tmp_path / "tiles").iterdir())
    assert names == ["t_000_001.npz", "t_001_001.npz"] and stats["tiles_done"] == 2


def test_merge_refuses_when_tiles_are_missing(tmp_path):
    spec = make_spec()
    TR.run_tiles(StubEngine(), spec, tmp_path, plane=stub_plane(), solution=SOL, max_tiles=3)
    with pytest.raises(FileNotFoundError, match="missing"):
        TR.merge(spec, tmp_path, solution=SOL)


def test_write_field_files_are_loadable(tmp_path):
    spec, _, _ = run_all(tmp_path)
    f = TR.merge(spec, tmp_path, solution=SOL)
    TR.write_field(f, tmp_path)
    z = np.load(tmp_path / "field.npz")
    assert set(["settle_nb", "settle_last", "n_wrong", "n_changes", "n_distinct", "flags", "gid"]) <= set(z.files)
    d = json.load(open(tmp_path / "dictionary.json"))
    assert d[0]["id"] == 0 and d[0]["grid"] == SOLUTION
