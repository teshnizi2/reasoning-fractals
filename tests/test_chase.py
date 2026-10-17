"""Zoom-centre selection: favour windows with rich but not saturated boundary structure."""
import numpy as np

from render import chase as CH


def smooth_field(n=160, level=3):
    return np.full((n, n), level, dtype=np.uint8)


def test_constant_field_scores_zero_everywhere():
    s = CH.window_scores(smooth_field(), win=32, step=8)
    assert s["score"].max() == 0


def test_white_noise_is_penalised_as_saturated_dust():
    rng = np.random.default_rng(0)
    noise = rng.integers(2, 13, size=(160, 160)).astype(np.uint8)
    structured = smooth_field()
    yy, xx = np.mgrid[0:160, 0:160]
    # nested ripples: moderate boundary density, several levels, smooth neighbours
    structured = (3 + 4 * (np.sin(xx / 6.0 + 2 * np.sin(yy / 9.0)) > 0) + 2 * (np.sin(yy / 5.0) > 0.3)).astype(np.uint8)
    sn = CH.window_scores(noise, win=32, step=8)["score"].max()
    ss = CH.window_scores(structured, win=32, step=8)["score"].max()
    assert ss > sn


def test_pick_centre_lands_on_the_localised_rich_patch():
    f = smooth_field(160, 3)
    yy, xx = np.mgrid[0:160, 0:160]
    patch = (3 + 4 * (np.sin(xx / 4.0 + np.sin(yy / 3.0)) > 0) + 2 * (np.sin(yy / 5.0) > 0.2)).astype(np.uint8)
    f[96:128, 32:64] = patch[96:128, 32:64]
    row, col, score = CH.pick_centre(f, win=32, step=4)
    assert 96 <= row < 128 and 32 <= col < 64 and score > 0


def test_plane_coordinates_of_a_picked_pixel_use_square_pixel_windows():
    # window centre (0.5, -0.25), half-width 0.1, 101 px: pixel (row 50, col 100) is the right edge, vertical middle
    x, y = CH.pixel_to_plane(50, 100, center=(0.5, -0.25), half_width=0.1, n=101)
    assert abs(x - 0.6) < 1e-12 and abs(y - (-0.25)) < 1e-12
    x, y = CH.pixel_to_plane(0, 0, center=(0.5, -0.25), half_width=0.1, n=101)
    assert abs(x - 0.4) < 1e-12 and abs(y - (-0.25 - 0.1)) < 1e-12  # row 0 is y_min in the field arrays
