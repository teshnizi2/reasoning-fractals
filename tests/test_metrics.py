"""Basin metrics: known answers on synthetic label fields, plus the outcome summary."""
import numpy as np
import pytest

from render import metrics as M


def stripes(n=200, width=20):
    return np.tile((np.arange(n) // width) % 2, (n, 1))


def test_constant_field_has_zero_entropy_and_no_boundary():
    m = M.label_metrics(np.zeros((100, 100), dtype=np.uint8))
    assert m["basin_entropy"] == 0 and m["boundary_box_fraction"] == 0 and m["n_labels"] == 1


def test_smooth_stripes_have_boundary_dimension_one():
    m = M.label_metrics(stripes())
    assert 0.9 < m["uncertainty_exponent"] < 1.1 and 0.9 < m["boundary_dimension"] < 1.1


def test_white_noise_has_boundary_dimension_two():
    rng = np.random.default_rng(0)
    m = M.label_metrics(rng.integers(0, 11, size=(200, 200)))
    assert m["boundary_dimension"] > 1.9 and m["uncertainty_exponent"] < 0.1
    assert m["boundary_basin_entropy"] > np.log(2)  # the log 2 criterion for fractal-like boundaries


def test_multi_scale_subsamples_by_stride():
    out = M.multi_scale(stripes(240, 24), strides=(1, 2, 4))
    assert set(out) == {1, 2, 4} and out[1]["shape"] == [240, 240] and out[4]["shape"] == [60, 60]


def test_outcome_summary_counts_solved_censored_and_distinct():
    settle = np.array([[2, 3], [3, 12]], dtype=np.uint8)
    flags = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    gid = np.array([[0, 0], [0, 7]])
    s = M.outcome_summary(settle, flags, gid)
    assert s["frac_solved"] == 0.75 and s["n_outcomes"] == 2 and s["n_wrong_outcomes"] == 1
    assert s["settle_hist"][3] == 2 and s["mean_settle"] == pytest.approx(5.0)
