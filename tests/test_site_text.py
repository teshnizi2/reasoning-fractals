"""Explorer copy: every number comes from the results dicts, none is typed in."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("site_text", Path(__file__).resolve().parents[1] / "scripts/site_text.py")
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)

PACK = {"layers": [dict(nx=2048, ny=2048, zoom=1.0), dict(nx=540, ny=675, zoom=64.0), dict(nx=540, ny=675, zoom=512.0)]}
METRICS = {"hero": dict(frac_solved=0.99912, pixels=4194304, shape=[2048, 2048])}
REPRO = dict(agreement=0.64424, metrics_ours=dict(basin_entropy=1.4164, boundary_dimension=1.8727),
             metrics_asset_same_pixels=dict(basin_entropy=1.4122, boundary_dimension=1.8722))
GPU = {"total": {"a100_hours": 41.6}}


def test_numbers_are_read_from_the_inputs():
    t = ST.build(PACK, METRICS, REPRO, GPU)
    joined = " ".join(t["limits"] + t["how"])
    assert "99.9%" in joined and "4,194,304" in joined and "64.4%" in joined
    assert "×512" in joined                      # deepest layer's zoom
    assert "1.416 vs 1.412" in joined and "1.873 vs 1.872" in joined
    assert "42 A100-hours" in joined             # 41.6 rounded
    assert "2048²" in joined and "2 zoom layers" in joined


def test_gpu_hours_sentence_is_omitted_without_accounting():
    t = ST.build(PACK, METRICS, REPRO, None)
    assert "A100-hours" not in " ".join(t["how"])


def test_unsolved_count_is_derived_not_assumed():
    t = ST.build(PACK, dict(hero=dict(frac_solved=1.0, pixels=1000, shape=[10, 100])), REPRO, None)
    assert "The 0 amber" in " ".join(t["limits"])
