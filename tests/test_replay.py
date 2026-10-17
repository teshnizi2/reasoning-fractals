"""Replay of chosen pixels on the same engine as the render: plain run vs stored values, plus rounding-sized nudges."""
import numpy as np

from render import replay as RP
from render import sudoku as S

PUZZLE = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
SOL = S.pack([SOLUTION])[0].astype(np.int8)


class StubRollout:
    """24 loops: blank for int(a) loops (a = latent[0, 0]), then the solution; a nudge of z shifts a by rounding noise."""

    def __init__(self):
        self.calls = []

    def __call__(self, z):
        self.calls.append(len(z))
        out = np.zeros((len(z), 24, 81), dtype=np.int8)
        for k, zz in enumerate(z):
            settle = int(round(float(zz[0, 0])))
            out[k, settle:] = SOL
        return out


def plane():
    u = np.zeros((97, 512), np.float32)
    v = np.zeros((97, 512), np.float32)
    u[0, 0], v[0, 1] = 1.0, 1.0
    return u, v


def items():
    return [
        dict(id="p0", x=3.0, y=0.0, prec="float32", stored=dict(settle=3, flags=1)),
        dict(id="p1", x=9.0, y=0.0, prec="float64", stored=dict(settle=9, flags=1)),
        dict(id="p2", x=5.0, y=0.0, prec="float32", stored=dict(settle=6, flags=1)),  # stored value the replay will not reproduce
    ]


def test_run_items_reports_settle_flags_traces_and_equality_with_the_stored_values():
    stub = StubRollout()
    out = RP.run_items(stub, plane(), PUZZLE, SOL, items(), nudged=2, eps=1e-6, seed=1, batch=2)
    assert [o["id"] for o in out] == ["p0", "p1", "p2"]  # order preserved although float32/float64 items are grouped
    assert [o["settle"] for o in out] == [3, 9, 5]
    assert [o["equal"] for o in out] == [True, True, False]
    assert all(o["flags"] & 1 for o in out)
    assert len(out[0]["trace"]) == 24 and out[0]["trace"][2] == "." * 81 and out[0]["trace"][3] == SOLUTION
    assert all(len(o["nudged"]) == 2 for o in out)
    assert max(stub.calls) <= 2  # never more than the batch size per rollout call


def test_nudged_runs_are_deterministic_given_the_seed():
    a = RP.run_items(StubRollout(), plane(), PUZZLE, SOL, items(), nudged=3, eps=1e-6, seed=7, batch=8)
    b = RP.run_items(StubRollout(), plane(), PUZZLE, SOL, items(), nudged=3, eps=1e-6, seed=7, batch=8)
    assert a == b


def test_trace_strings_use_dot_for_blank_and_question_mark_for_junk():
    t = np.zeros((2, 81), np.int8)
    t[0, 0], t[1, 1], t[1, 2] = 7, -1, 11
    s = RP.trace_strings(t)
    assert s[0].startswith("7.") and s[1].startswith(".??") and len(s[0]) == 81
