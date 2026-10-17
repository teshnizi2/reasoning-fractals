"""Colourways: ordering of luminance, outcome hue, series palette fit, linear-light downsampling."""
import colorsys

import numpy as np

from render import colour as C


def lum(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def hue(rgb):
    return colorsys.rgb_to_hsv(*(np.asarray(rgb) / 255.0))[0]


def hue_gap(a, b):
    d = abs(hue(a) - hue(b))
    return min(d, 1 - d)


def test_ramp_interpolates_anchors_exactly_and_is_uint8():
    lut = C.ramp([(0.0, (0, 0, 0)), (1.0, (255, 255, 255))], n=5)
    assert lut.dtype == np.uint8 and lut.shape == (5, 3)
    assert tuple(lut[0]) == (0, 0, 0) and tuple(lut[-1]) == (255, 255, 255)
    assert (np.diff(lum(lut)) > 0).all()


def test_both_colourways_get_lighter_as_settle_time_grows():
    for name in ("loopscape", "series"):
        cw = C.COLOURWAYS[name]
        ls = [lum(cw.solved_lut[t]) for t in range(cw.t_lo, cw.t_hi + 1)]
        assert all(b > a for a, b in zip(ls, ls[1:])), name


def test_loopscape_is_indigo_to_mint_and_series_starts_at_the_page_background():
    a = C.COLOURWAYS["loopscape"]
    assert hue(a.solved_lut[a.t_lo]) > 0.6 and 0.4 < hue(a.solved_lut[a.t_hi]) < 0.5  # indigo -> mint-green
    b = C.COLOURWAYS["series"]
    assert tuple(b.solved_lut[b.t_lo]) == (0x0E, 0x11, 0x17)  # #0e1117, the series background


def test_series_top_of_the_solved_ramp_is_amber():
    b = C.COLOURWAYS["series"]
    h = hue(b.solved_lut[(b.t_lo + b.t_hi) // 2 + 2]) * 360
    assert 25 < h < 50  # amber / gold hues


def test_wrong_answers_use_a_clearly_different_hue_at_the_same_settle_time():
    for name in ("loopscape", "series"):
        cw = C.COLOURWAYS[name]
        t = (cw.t_lo + cw.t_hi) // 2
        assert hue_gap(cw.solved_lut[t], cw.wrong_lut[t]) > 0.12, name


def test_colourize_picks_the_right_ramp_per_pixel():
    cw = C.COLOURWAYS["loopscape"]
    settle = np.array([[cw.t_lo, cw.t_hi], [cw.t_lo, cw.t_hi]], dtype=np.uint8)
    flags = np.array([[1, 1], [0, 0]], dtype=np.uint8)  # bit0 = solved
    img = C.colourize(settle, flags, cw)
    assert img.shape == (2, 2, 3) and img.dtype == np.uint8
    assert tuple(img[0, 0]) == tuple(cw.solved_lut[cw.t_lo]) and tuple(img[0, 1]) == tuple(cw.solved_lut[cw.t_hi])
    assert tuple(img[1, 0]) == tuple(cw.wrong_lut[cw.t_lo]) and tuple(img[1, 1]) == tuple(cw.wrong_lut[cw.t_hi])


def test_settle_times_outside_the_range_are_clipped():
    cw = C.COLOURWAYS["loopscape"]
    img = C.colourize(np.array([[0, 23]], dtype=np.uint8), np.array([[1, 1]], dtype=np.uint8), cw)
    assert tuple(img[0, 0]) == tuple(cw.solved_lut[cw.t_lo]) and tuple(img[0, 1]) == tuple(cw.solved_lut[cw.t_hi])


def test_censored_pixels_get_the_censored_colour():
    cw = C.COLOURWAYS["series"]
    img = C.colourize(np.array([[5]], dtype=np.uint8), np.array([[1 | 4]], dtype=np.uint8), cw)  # bit2 = censored
    assert tuple(img[0, 0]) == tuple(cw.censored)


def test_downsample_averages_in_linear_light():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0] = (255, 255, 255)
    out = C.downsample(img, 2)
    # one white of four pixels: linear-light mean 0.25 -> sRGB ~ 137, not the naive 64
    assert out.shape == (1, 1, 3) and 133 <= int(out[0, 0, 0]) <= 141


def test_downsample_keeps_flat_colours_and_rejects_ragged_sizes():
    img = np.full((4, 6, 3), 77, dtype=np.uint8)
    out = C.downsample(img, 2)
    assert out.shape == (2, 3, 3) and (out == 77).all()
    import pytest

    with pytest.raises(ValueError):
        C.downsample(np.zeros((5, 4, 3), dtype=np.uint8), 2)
