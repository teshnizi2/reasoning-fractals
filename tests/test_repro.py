"""Reproduction helpers: decode the upstream 11-level asset and compare fields pixel by pixel."""
from pathlib import Path

import numpy as np
import pytest

from render import repro as R
from render import slices as SL

ASSET = (
    Path(__file__).resolve().parents[1]
    / "vendor/loopscape/assets/basins_1000_0_eqr_sudoku_plane_rating47_7d68927f389fbdaa_rating47.png"
)
needs_asset = pytest.mark.skipif(not ASSET.exists(), reason="upstream asset not cloned into vendor/")


@needs_asset
def test_asset_decodes_to_eleven_ordered_levels_with_known_histogram():
    levels, colours = R.load_asset_levels(ASSET)
    assert levels.shape == (1000, 1000) and levels.dtype == np.uint8
    assert colours.shape == (11, 3)
    counts = np.bincount(levels.ravel(), minlength=11).tolist()
    # darkest -> lightest; counts measured directly from the PNG's 11 colours
    assert counts == [5981, 234256, 227025, 95713, 114082, 82673, 61447, 50919, 36164, 26536, 65204]
    lum = colours @ np.array([0.2126, 0.7152, 0.0722])
    assert (np.diff(lum) > 0).all()


def test_variants_are_the_eight_signed_permutations_of_the_plane_axes():
    assert len(R.VARIANTS) == 8
    n = 7
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    seen = set()
    for name in R.VARIANTS:
        xi, yi = R.variant_axes(name, r.ravel(), c.ravel(), n)
        # each variant is a bijection of the index grid onto itself
        assert len(set(zip(xi.tolist(), yi.tolist()))) == n * n
        seen.add(tuple(zip(xi.tolist(), yi.tolist())))
    assert len(seen) == 8  # all distinct


def test_identity_variant_is_notebook_orientation():
    n = 9
    r = np.array([0, 3, 8])
    c = np.array([1, 4, 2])
    xi, yi = R.variant_axes("xy", r, c, n)
    assert xi.tolist() == c.tolist() and yi.tolist() == r.tolist()


def test_axis_is_exactly_antisymmetric_so_flips_are_sign_changes():
    a = SL.axis(1000)
    np.testing.assert_array_equal(a[::-1], -a)


def test_best_shift_agreement_finds_the_offset():
    t = np.array([2, 3, 4, 5, 5, 6, 7, 2, 3])
    levels = t - 2
    shift, frac = R.best_shift_agreement(levels, t)
    assert shift == 2 and frac == 1.0
    noisy = levels.copy()
    noisy[0] = 9
    shift, frac = R.best_shift_agreement(noisy, t)
    assert shift == 2 and abs(frac - 8 / 9) < 1e-12


def test_contingency_counts_pairs():
    a = np.array([0, 0, 1, 2])
    b = np.array([3, 3, 4, 4])
    m = R.contingency(a, b, na=3, nb=5)
    assert m.shape == (3, 5) and m[0, 3] == 2 and m[1, 4] == 1 and m[2, 4] == 1 and m.sum() == 4


def test_spearman_is_one_for_monotone_relation():
    x = np.array([1, 2, 3, 4, 10])
    assert abs(R.spearman(x, x * 3 + 1) - 1.0) < 1e-12
    assert R.spearman(x, -x) < -0.99


def test_aligned_indices_include_both_edges_and_are_symmetric():
    idx = R.aligned_indices(24, n=1000)
    assert idx[0] == 0 and idx[-1] == 999 and len(idx) == 24
    assert (idx + idx[::-1] == 999).all()


def test_variant_sample_matches_brute_force_pixel_lookup():
    rng = np.random.default_rng(0)
    n, k = 49, 8
    image = rng.integers(0, 11, size=(n, n)).astype(np.uint8)
    idx = R.aligned_indices(k, n)
    a = image[np.ix_(idx, idx)]
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for name in R.VARIANTS:
        xi, yi = R.variant_axes(name, rr.ravel(), cc.ravel(), n)
        want = np.empty((k, k), dtype=np.uint8)
        for i in range(k):
            for j in range(k):
                hit = np.flatnonzero((xi == idx[j]) & (yi == idx[i]))
                assert len(hit) == 1
                want[i, j] = image.ravel()[hit[0]]
        np.testing.assert_array_equal(R.variant_sample(a, name), want, err_msg=name)


def test_best_shift_agreement_accepts_uint8_fields_and_negative_shifts():
    # regression: render outputs are uint8; `t - s` with s < 0 used to raise OverflowError
    t = np.array([0, 1, 2, 3, 3, 4], dtype=np.uint8)
    levels = np.array([1, 2, 3, 4, 4, 5], dtype=np.uint8)  # t + 1, i.e. shift = -1
    shift, frac = R.best_shift_agreement(levels, t)
    assert shift == -1 and frac == 1.0
