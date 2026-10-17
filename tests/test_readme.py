"""README numbers come from results/: formatting, strict substitution, and loading of the real JSON files."""
import json
from pathlib import Path

import pytest

from render import readme as RM

ROOT = Path(__file__).resolve().parents[1]


def test_fmt_covers_the_styles_the_readme_uses():
    assert RM.fmt(0.64424, "pct1") == "64.4%"
    assert RM.fmt(0.9994, "pct2") == "99.94%"
    assert RM.fmt(0.31, "pct0") == "31%"
    assert RM.fmt(1.41644, "f3") == "1.416"
    assert RM.fmt(62500, "int") == "62,500"
    assert RM.fmt(1.631e-5, "sci1") == "1.6×10^-5"
    assert RM.fmt(2.69, "x1") == "2.7×"
    assert RM.fmt(18.51, "h1") == "18.5 h"
    assert RM.fmt("abc", None) == "abc"
    with pytest.raises(ValueError):
        RM.fmt(1, "bogus")


def test_substitute_fills_numbers_and_tables_and_refuses_unknown_keys():
    out = RM.substitute("a {{x.y|pct1}} b {{ n|int }} c\n{{table:t}}", {"x.y": 0.5, "n": 12345}, {"t": "| T |"})
    assert out == "a 50.0% b 12,345 c\n| T |"
    with pytest.raises(KeyError):
        RM.substitute("{{missing|f1}}", {}, {})
    assert RM.substitute("no placeholders { } here", {}, {}) == "no placeholders { } here"


def test_load_numbers_reads_the_real_results_and_matches_the_json():
    N = RM.load_numbers(ROOT)
    r = json.loads((ROOT / "results/repro/repro250_summary.json").read_text())
    assert N["repro.agreement"] == r["agreement"] and N["repro.n_pixels"] == 62500
    assert abs(N["ee.speedup"] - 2.69) < 0.02 and 0.999 < N["ee.settle_equal"] < 1.0
    assert N["stock.n"] == 4096 and 0.6 < N["stock.fp32_agree"] < 0.7
    T = RM.tables(ROOT, N)
    assert "64.4%" in T["repro"] and "upstream image" in T["repro"]
    assert T["scale"].count("\n") >= 5 and T["cost"].startswith("| render |")


def test_md_table_shape():
    t = RM.md_table(["a", "b"], [["1", "2"], ["3", "4"]])
    assert t.splitlines()[1] == "|---|---|" and t.count("\n") == 3


def test_hero_gpu_and_scan_numbers_are_picked_up_from_their_json_files(tmp_path):
    (tmp_path / "results/alice").mkdir(parents=True)
    hero = dict(pixels=4194304, shape=[2048, 2048], frac_solved=0.999, frac_censored=0.0008, n_outcomes=40, n_wrong_outcomes=39,
                mean_settle=5.5, max_settle=23, gpu_seconds=36000.0, mean_loops_computed=9.1,
                adjacent_gap=dict(max_gap=20, frac_ge={"1": 0.4, "2": 0.2, "5": 0.05, "10": 0.01}),
                settle_time_metrics={"1": dict(basin_entropy=0.9, boundary_basin_entropy=1.4, uncertainty_exponent=0.15, boundary_dimension=1.85,
                                               boundary_box_fraction=0.6, shape=[2048, 2048])})
    (tmp_path / "results/metrics.json").write_text(json.dumps(dict(hero=hero, keyframes=[dict(pixels=100, gpu_seconds=3600.0, zoom=8)])))
    (tmp_path / "results/alice/gpu_hours.json").write_text(json.dumps(dict(total=dict(jobs=7, a100_hours=12.5, gpu_hours_raw=15.0))))
    (tmp_path / "results/scans.json").write_text(json.dumps(dict(
        pscan=dict(n_puzzles=5565, k=16, frac_solved=0.9, always_solved=4281, never_solved=115, partial=1169, multistable=664, asset=dict(frac_solved=1.0)),
        longrun=dict(n=400, loops=96, frac_solved_by_end=0.4675, still_wrong_and_changing=213, still_wrong_and_frozen=0, first_solved_percentiles={"50": 25.0}))))
    N = RM.load_numbers(tmp_path)
    assert N["hero.pixels"] == 4194304 and N["hero.side"] == 2048 and N["hero.gpu_hours"] == 10.0 and N["hero.n_unsolved"] == 4194
    assert N["hero.gap_ge5"] == 0.05 and N["hero.max_gap"] == 20 and N["hero.s1.d"] == 1.85 and N["kf.n"] == 1 and N["kf.zoom_max"] == 8
    assert N["gpu.a100_hours"] == 12.5 and N["pscan.starts"] == 5565 * 16 and N["long.frozen"] == 0
