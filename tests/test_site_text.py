"""Explorer copy: every number comes from the results dicts, none is typed in."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("site_text", Path(__file__).resolve().parents[1] / "scripts/site_text.py")
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)

PACK = {"layers": [dict(nx=2048, ny=2048, zoom=1.0), dict(nx=540, ny=675, zoom=64.0), dict(nx=540, ny=675, zoom=512.0)]}
METRICS = {"hero": dict(frac_solved=0.99912, frac_censored=0.00083, pixels=4194304, shape=[2048, 2048])}
REPRO = dict(agreement=0.64424, n_pixels=62500, metrics_ours=dict(basin_entropy=1.4164, boundary_dimension=1.8727),
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
    t = ST.build(PACK, dict(hero=dict(frac_solved=1.0, frac_censored=0.0, pixels=1000, shape=[10, 100])), REPRO, None)
    assert "The 0 pixels that do not end on the solution" in " ".join(t["limits"])


def test_unsettled_and_stuck_counts_add_up_to_the_unsolved_pixels():
    t = ST.build(PACK, METRICS, REPRO, None)
    joined = " ".join(t["limits"])
    # 4,194,304 * (1 - 0.99912) = 3,691 unsolved; 0.00083 of the pixels (3,481) are still changing at loop 24
    assert "3,691 pixels that do not end on the solution" in joined and "3,481 are starts still changing" in joined and "210 settled on a wrong grid" in joined


def test_speckle_and_zoom_sentences_use_the_measured_numbers_when_given():
    nums = {"pscan.plane_norm": 1.414, "pscan.native_norm": 222.9, "replay.flat": 0.97, "replay.dust": 0.26, "nudge.flat": 0.97, "nudge.dust": 0.18}
    scale = dict(rows=[
        dict(label="hero 2048x2048, every pixel", pitch=9.8e-4, shape=[2048, 2048], boundary_box_fraction=0.69),
        dict(label="keyframe x128 (540x675, centred on the zoom target)", pitch=2.9e-5, shape=[675, 540], boundary_box_fraction=0.31),
        dict(label="window x80 (160x160, chase2)", pitch=1.6e-4, shape=[160, 160], boundary_box_fraction=0.44)])
    t = ST.build(PACK, METRICS, REPRO, None, scale, nums)
    joined = " ".join(t["limits"])
    assert "at most 1.41" in joined and "about 223" in joined
    assert "97% of the pixels with flat surroundings" in joined and "only 26% of the speckle pixels" in joined and "leaves 97% of flat-region pixels unchanged and 18%" in joined
    assert "only 31% in the layer magnified ×128" in joined  # the unbiased keyframe, not the hand-picked chase window
    assert "a trial window at ×" in joined
