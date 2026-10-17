"""Latent-plane slices: must reproduce the loopscape notebook's construction exactly."""
import numpy as np
import torch

from render import slices as SL


# ---- reference: verbatim from vendor/loopscape/demos/basins.ipynb ---------------------
def nb_mesh(res):
    g = torch.linspace(-1, 1, res)
    x, y = torch.meshgrid(g, g, indexing="xy")
    return torch.stack([x.ravel(), y.ravel()], -1)


def nb_plane(shape, seed=0):
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(int(np.prod(shape)), 2))
    return Q[:, 0].reshape(shape), Q[:, 1].reshape(shape)


def nb_latents(p, u, v):
    return p[:, 0, None, None] * u + p[:, 1, None, None] * v


# ------------------------------------------------------------------------------------


def test_plane_matches_notebook_bitwise():
    u, v = SL.plane((97, 512), seed=0)
    ru, rv = nb_plane((97, 512), seed=0)
    assert u.shape == (97, 512) and u.dtype == np.float32
    np.testing.assert_array_equal(u, ru.numpy())
    np.testing.assert_array_equal(v, rv.numpy())


def test_plane_is_orthonormal():
    u, v = SL.plane((97, 512), seed=3)
    assert abs(float((u.astype(np.float64) ** 2).sum()) - 1) < 1e-5
    assert abs(float((v.astype(np.float64) ** 2).sum()) - 1) < 1e-5
    assert abs(float((u.astype(np.float64) * v).sum())) < 1e-5


def test_axis_matches_torch_linspace():
    np.testing.assert_array_equal(SL.axis(1000), torch.linspace(-1, 1, 1000).numpy())


def test_mesh_order_is_row_major_x_fastest_like_notebook():
    res = 7
    ref = nb_mesh(res).numpy()
    got = SL.mesh(res)
    np.testing.assert_array_equal(got, ref)
    # pixel (row, col) -> (x = axis[col], y = axis[row])
    a = SL.axis(res)
    assert tuple(got[2 * res + 5]) == (a[5], a[2])


def test_latents_match_notebook_arithmetic_bitwise():
    u, v = SL.plane((97, 512), seed=0)
    p = SL.mesh(5)
    z = SL.latents(p, u, v)
    ref = nb_latents(torch.tensor(p), torch.tensor(u), torch.tensor(v)).numpy()
    assert z.dtype == np.float32 and z.shape == (25, 97, 512)
    np.testing.assert_array_equal(z, ref)


def test_pixel_coords_for_indices_uses_the_asset_grid():
    # the 1000x1000 asset samples axis(1000); a stride-10 render must hit a subset of it
    rows, cols = np.meshgrid(np.arange(0, 1000, 10), np.arange(0, 1000, 10), indexing="ij")
    c = SL.coords_from_indices(rows.ravel(), cols.ravel(), n=1000)
    a = SL.axis(1000)
    assert c.shape == (10000, 2)
    np.testing.assert_array_equal(c[:, 0], a[cols.ravel()])
    np.testing.assert_array_equal(c[:, 1], a[rows.ravel()])


def test_window_maps_pixels_to_scaled_axis():
    w = SL.Window(center=(0.25, -0.5), half_width=0.01, n=9)
    c = w.coords()  # float64, (81, 2), row-major x fastest
    assert c.dtype == np.float64 and c.shape == (81, 2)
    xs = c[:9, 0]
    assert abs(xs[0] - (0.25 - 0.01)) < 1e-15 and abs(xs[-1] - (0.25 + 0.01)) < 1e-15
    ys = c[::9, 1]
    assert abs(ys[0] - (-0.5 - 0.01)) < 1e-15 and abs(ys[-1] - (-0.5 + 0.01)) < 1e-15


def test_zoom_window_divides_width_by_factor_and_keeps_centre():
    w = SL.Window(center=(0.1, 0.2), half_width=1.0, n=64)
    z = w.zoomed(4)
    assert z.center == w.center and z.half_width == 0.25 and z.n == 64


def test_octave_zoom_is_two_to_the_k():
    w = SL.Window(center=(0.1, 0.2), half_width=1.0, n=64)
    assert w.octave(4).half_width == 1.0 / 16
    assert w.octave(0) == w


def test_float64_latents_agree_with_float32_notebook_mode_to_rounding():
    u, v = SL.plane((97, 512), seed=0)
    p = SL.mesh(4)
    z32 = SL.latents(p, u, v)
    z64 = SL.latents(p.astype(np.float64), u, v, precision="float64")
    assert z64.dtype == np.float32  # engine input is always float32
    np.testing.assert_allclose(z64, z32, rtol=0, atol=1e-7)


def test_distinct_latent_count_detects_precision_collapse():
    u, v = SL.plane((97, 512), seed=0)
    # absurdly deep window: neighbouring pixels differ by 1e-14 -> identical float32 latents
    w = SL.Window(center=(0.3, 0.3), half_width=1e-13, n=4)
    z = SL.latents(w.coords(), u, v, precision="float64")
    assert SL.n_distinct_latents(z) < 16
    w2 = SL.Window(center=(0.3, 0.3), half_width=1e-3, n=4)
    z2 = SL.latents(w2.coords(), u, v, precision="float64")
    assert SL.n_distinct_latents(z2) == 16
