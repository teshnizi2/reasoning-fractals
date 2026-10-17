"""Which pixels go to the replay job: featured-start candidates of the hero and neighbouring pairs of the keyframes."""
import numpy as np

from render import replayplan as RPL
from render import tilerun as TR


def hero_like(n=64):
    rng = np.random.default_rng(1)
    s = rng.integers(3, 8, size=(n, n)).astype(np.uint8)
    f = np.ones((n, n), np.uint8)
    s[10, 10] = 2          # the fastest common value would be 3..7; a lone 2 is not "common"
    s[30, 30], s[30, 31] = 3, 13  # a neighbouring pair with a big gap
    s[40, 40] = 19         # slowest solved pixel below the cap
    s[50, 50] = 20         # a wrong, settled grid
    f[50, 50] = 0
    return s, f


def test_plan_has_stored_values_coordinates_and_roles_for_every_item():
    s, f = hero_like()
    spec = dict(mode="notebook", n=64, tile=64)
    plan = RPL.plan_layer("hero", spec, s, f, kinds=("fast", "slow", "wrong", "pair"), per_kind=10, pair_min_gap=6, pair_max_gap=15, margin=2)
    kinds = {p["kind"] for p in plan}
    assert kinds == {"fast", "slow", "wrong", "pair"}
    slow = next(p for p in plan if p["kind"] == "slow")
    assert (slow["row"], slow["col"]) == (40, 40) and slow["stored"] == dict(settle=19, flags=1)
    pair = [p for p in plan if p["kind"] == "pair"]
    assert {p["role"] for p in pair} == {"a", "b"}
    a = next(p for p in pair if p["role"] == "a" and (p["row"], p["col"]) == (30, 30))
    b = next(p for p in pair if p["role"] == "b" and p["group"] == a["group"])
    assert (b["row"], b["col"]) == (30, 31) and b["stored"]["settle"] == 13
    # coordinates are exactly what the renderer used for that pixel
    t = TR.Tile(0, 0, 0, a["row"], a["col"], 1, 1)
    assert [a["x"], a["y"]] == [float(v) for v in TR.pixel_coords(spec, t)[0]]
    assert a["prec"] == "float32"
    assert len({p["id"] for p in plan}) == len(plan)  # ids are unique


def test_window_layers_use_float64_latents_and_pair_candidates_respect_the_margin():
    s, f = hero_like(80)
    spec = dict(mode="window", center=[0.5, 0.25], half_width=0.01, nx=80, ny=80, n=80, tile=80)
    plan = RPL.plan_layer("kf07", spec, s, f, kinds=("pair",), per_kind=5, pair_min_gap=6, pair_max_gap=15, margin=24)
    assert all(p["prec"] == "float64" for p in plan)
    assert all(24 <= p["row"] < 80 - 24 for p in plan)


def test_replay_input_json_round_trips_the_coordinates_exactly():
    s, f = hero_like()
    spec = dict(mode="notebook", n=64, tile=64)
    plan = RPL.plan_layer("hero", spec, s, f, kinds=("fast",), per_kind=2, pair_min_gap=6, pair_max_gap=15, margin=2)
    doc = RPL.replay_input("PUZZLE", plan, nudged=3, eps=1e-6, seed=5)
    assert doc["nudged"] == 3 and doc["eps"] == 1e-6 and len(doc["items"]) == len(plan)
    assert [it["id"] for it in doc["items"]] == [p["id"] for p in plan]
    assert set(doc["items"][0]) == {"id", "x", "y", "prec", "stored"}


def test_roughness_strata_sample_solved_pixels_by_how_different_their_neighbours_are():
    rng = np.random.default_rng(0)
    s = np.full((60, 60), 4, np.uint8)
    s[:, 30:] = 5                                  # a border: roughness 1 along columns 29/30
    s[rng.random((60, 60)) < 0.08] = 7              # mild speckle: roughness 2-4
    s[rng.random((60, 60)) < 0.03] = 12             # strong speckle: roughness >= 5
    f = np.ones((60, 60), np.uint8)
    spec = dict(mode="notebook", n=60, tile=60)
    plan = RPL.plan_roughness("hero", spec, s, f, per_bin=15, seed=3)
    kinds = sorted({p["kind"] for p in plan})
    assert kinds == ["rough0", "rough1", "rough2", "rough5"]
    from collections import Counter
    assert all(c <= 15 for c in Counter(p["kind"] for p in plan).values())
    pad = np.pad(s.astype(int), 1, mode="edge")
    for p in plan:
        r, c = p["row"], p["col"]
        rough = max(abs(int(s[r, c]) - pad[r + 1 + dy, c + 1 + dx]) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)))
        lo = int(p["kind"][5:])
        assert (rough == 0) if lo == 0 else (rough == 1) if lo == 1 else (2 <= rough <= 4) if lo == 2 else rough >= 5
    assert len({p["id"] for p in plan}) == len(plan)
