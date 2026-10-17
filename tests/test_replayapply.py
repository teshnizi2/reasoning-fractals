"""From replay results to explorer featured starts and the video's finale pair."""
from render import replayapply as RA

SOL = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
BLANK = "." * 81


def trace(settle: int) -> list[str]:
    return [BLANK] * settle + [SOL] * (24 - settle)


def item(layer, kind, group, role, row, col, settle, x=0.0, y=0.0, flags=1):
    return dict(id=f"{layer}:{kind}:{group}:{role}", kind=kind, layer=layer, row=row, col=col, role=role, group=group, x=x, y=y,
                prec="float64", stored=dict(settle=settle, flags=flags))


def result(it, settle=None, nudged=None, equal=True, tr=None):
    T = it["stored"]["settle"] if settle is None else settle
    return dict(id=it["id"], settle=T, flags=it["stored"]["flags"], trace=tr if tr is not None else trace(T), nudged=nudged if nudged is not None else [T, T, T], equal=equal)


def test_featured_picks_the_first_verified_candidate_of_each_kind_and_maps_rows_to_display():
    p = [
        item("hero", "fast", 0, "single", 10, 20, 3), item("hero", "fast", 1, "single", 11, 21, 3),
        item("hero", "slow", 0, "single", 30, 40, 19), item("hero", "slow", 1, "single", 31, 41, 17),
        item("hero", "wrong", 0, "single", 50, 60, 22, flags=4),
    ]
    r = [result(p[0], equal=False), result(p[1]), result(p[2], equal=False), result(p[3], nudged=[17, 11, 20]),
         result(p[4], tr=[BLANK] * 24)]
    feats = RA.featured(p, r, ny=100)
    by = {f["id"]: f for f in feats}
    assert set(by) == {"fast", "slow", "wrong"}
    assert (by["fast"]["row"], by["fast"]["col"]) == (100 - 1 - 11, 21) and by["fast"]["settle"] == 3 and by["fast"]["layer"] == 0
    assert by["fast"]["title"] == "settles in 3 loops" and by["fast"]["solved"] is True
    assert (by["slow"]["row"], by["slow"]["col"]) == (68, 41)  # first verified: the 17-loop one
    assert "11" in by["slow"]["note"] and "20" in by["slow"]["note"] and "ounding" in by["slow"]["note"]
    assert "agree" in by["fast"]["note"]
    assert by["wrong"]["solved"] is False and len(by["fast"]["trace"]) == 24


def test_featured_pair_needs_both_pixels_verified_and_reports_the_gap():
    p = [item("hero", "pair", 0, "a", 5, 5, 4), item("hero", "pair", 0, "b", 5, 6, 12),
         item("hero", "pair", 1, "a", 7, 7, 4), item("hero", "pair", 1, "b", 7, 8, 9)]
    r = [result(p[0]), result(p[1], equal=False), result(p[2]), result(p[3])]
    f = RA.featured(p, r, ny=50)
    assert [x["id"] for x in f] == ["pair"] and f[0]["title"] == "next to a pixel 5 loops away" and f[0]["col"] == 7


def finale_items(layer, g, ta, tb, row=40, col=40):
    return [item(layer, "pair", g, "a", row, col, ta, x=1.0, y=2.0), item(layer, "pair", g, "b", row, col + 1, tb, x=1.001, y=2.0)]


def test_finale_pick_prefers_the_pair_whose_ordering_and_gap_survive_the_nudges():
    a0, b0 = finale_items("kf07", 0, 3, 14)  # big stored gap but the slow one wobbles down to 5
    a1, b1 = finale_items("kf07", 1, 4, 9)   # smaller gap, rock solid
    a2, b2 = finale_items("kf07", 2, 3, 15)  # not reproduced by the plain replay
    plan = [a0, b0, a1, b1, a2, b2]
    res = [result(a0, nudged=[3, 3, 4]), result(b0, nudged=[5, 14, 9]), result(a1, nudged=[4, 4, 4]), result(b1, nudged=[9, 9, 8]),
           result(a2, equal=False), result(b2)]
    best = RA.pick_finale(plan, res, pitches={"kf07": 3e-5}, solution=SOL, min_robust_gap=2)
    assert best["layer"] == "kf07" and best["a"]["settle"] == 4 and best["b"]["settle"] == 9
    assert best["pitch"] == 3e-5 and best["gap"] == 5
    assert best["verified"]["a_range"] == [4, 4] and best["verified"]["b_range"] == [8, 9]
    assert len(best["a"]["trace"]) == 24 and best["a"]["label"] == "A" and best["b"]["x"] == 1.001


def test_finale_pick_requires_monotone_traces_and_returns_none_when_nothing_qualifies():
    a, b = finale_items("kf07", 0, 3, 10)
    bad = [SOL] + [BLANK] * 5 + [SOL] * 18  # leaves the solution again
    assert RA.pick_finale([a, b], [result(a, tr=bad), result(b)], pitches={"kf07": 3e-5}, solution=SOL) is None
    a2, b2 = finale_items("kf07", 1, 3, 10)
    assert RA.pick_finale([a2, b2], [result(a2, nudged=[3, 3, 9]), result(b2, nudged=[10, 10, 5])], pitches={"kf07": 3e-5},
                          solution=SOL, min_robust_gap=3) is None  # a nudge moves them to 9 vs 5: the ordering is lost


def test_roughness_summary_counts_reproduction_and_nudge_stability_per_stratum():
    p = [item("hero", "rough0", 0, "single", 1, 1, 4), item("hero", "rough0", 1, "single", 1, 2, 4), item("hero", "rough5", 0, "single", 5, 5, 9),
         item("hero", "rough5", 1, "single", 5, 6, 10), item("hero", "pair", 0, "a", 7, 7, 3)]
    r = [result(p[0], nudged=[4, 4, 4]), result(p[1], nudged=[4, 5, 4]), result(p[2], equal=False, settle=11, nudged=[7, 12, 9]),
         result(p[3], nudged=[10, 10, 10]), result(p[4])]
    s = RA.roughness_summary(p, r)
    assert set(s) == {"rough0", "rough5"}
    assert s["rough0"] == dict(n=2, reproduced=2, nudge_stable=1, mean_max_change=0.5)
    assert s["rough5"]["n"] == 2 and s["rough5"]["reproduced"] == 1 and s["rough5"]["nudge_stable"] == 1
    assert abs(s["rough5"]["mean_max_change"] - (4.0 + 0.0) / 2) < 1e-9  # max |nudged - plain| per item: 11 vs 7, 12, 9 -> 4; 10 vs 10, 10, 10 -> 0


def test_median_criterion_accepts_a_pair_whose_order_mostly_survives_the_nudges():
    a, b = finale_items("kf07", 0, 3, 14)
    plan = [a, b]
    res = [result(a, nudged=[3, 3, 3, 4, 3]), result(b, nudged=[2, 12, 9, 13, 11])]  # one nudge flips the order (b=2 < a=3)
    assert RA.pick_finale(plan, res, pitches={"kf07": 3e-5}, solution=SOL, min_robust_gap=3) is None  # strict: every run must keep the gap
    soft = RA.pick_finale(plan, res, pitches={"kf07": 3e-5}, solution=SOL, min_robust_gap=3, criterion="median")
    assert soft is not None and soft["verified"]["criterion"] == "median" and soft["verified"]["runs_keeping_order"] == 5  # plain + 4 of 5 nudged
    assert RA.pick_finale(plan, res, pitches={"kf07": 3e-5}, solution=SOL, min_robust_gap=3)  is None
