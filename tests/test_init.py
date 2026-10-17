"""EqR's native latent initialisation (truncated normal) and puzzle sampling for the scan."""
import numpy as np

from render import init as I


def test_trunc_normal_has_unit_std_and_is_bounded_like_eqr():
    x = I.trunc_normal((200_000,), std=1.0, rng=np.random.default_rng(0))
    assert x.dtype == np.float32
    assert abs(float(x.mean())) < 0.01 and abs(float(x.std()) - 1.0) < 0.01
    bound = 2.0 * I.comp_std(1.0)
    assert float(np.abs(x).max()) <= bound + 1e-6 and float(np.abs(x).max()) > 0.98 * bound
    assert abs(I.comp_std(1.0) - 1.1368) < 1e-3


def test_trunc_normal_scales_with_std_and_is_reproducible():
    a = I.trunc_normal((1000,), std=0.5, rng=np.random.default_rng(1))
    b = I.trunc_normal((1000,), std=0.5, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert abs(float(a.std()) - 0.5) < 0.03


def test_stratified_sample_takes_from_every_rating_bin_without_repeats():
    rows = [("s", f"q{i}", f"a{i}", r) for i, r in enumerate([0, 5, 12, 40, 41, 90, 120, 130, 300, 310, 320] * 20)]
    pick = I.stratified(rows, edges=(0, 10, 50, 100, 1000), per_bin=7, seed=0)
    assert len(pick) == 28 and len({p[1] for p in pick}) == 28
    ratings = np.array([p[3] for p in pick])
    assert ((ratings < 10).sum(), ((ratings >= 10) & (ratings < 50)).sum(), ((ratings >= 50) & (ratings < 100)).sum(),
            (ratings >= 100).sum()) == (7, 7, 7, 7)
