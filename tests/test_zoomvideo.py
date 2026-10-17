"""Deep-zoom compositing: camera path, layer fade, and frames built from nested keyframe images."""
import numpy as np

from render import zoomvideo as ZV


def test_camera_hits_both_endpoints_and_zooms_monotonically():
    cam = ZV.Camera(c0=(0.0, 0.0), c1=(0.2, -0.3), zoom0=1.0, zoom1=4096.0)
    assert cam.at(0.0) == (0.0, 0.0, 1.0)
    cx, cy, z = cam.at(1.0)
    assert (cx, cy) == (0.2, -0.3) and abs(z - 4096.0) < 1e-6
    zs = [cam.at(t)[2] for t in np.linspace(0, 1, 50)]
    assert all(b >= a for a, b in zip(zs, zs[1:]))
    # equal steps in log-zoom at the midpoint of an ease-in-out: geometric mean
    assert abs(cam.at(0.5)[2] - 64.0) < 1e-6


def test_layer_alpha_fades_in_as_the_coarser_layer_gets_magnified():
    assert ZV.layer_alpha(prev_pitch_px=0.5) == 0.0
    assert ZV.layer_alpha(prev_pitch_px=1.0) == 0.0
    assert ZV.layer_alpha(prev_pitch_px=2.3) == 1.0
    assert 0.0 < ZV.layer_alpha(prev_pitch_px=1.65) < 1.0


def make_layers():
    base = ZV.Layer(center=(0.0, 0.0), half_width=1.0, img=np.full((64, 64, 3), (200, 0, 0), np.uint8))
    fine = ZV.Layer(center=(0.0, 0.0), half_width=0.125, img=np.full((64, 64, 3), (0, 0, 200), np.uint8))
    return [base, fine]


def test_frame_at_zoom_one_shows_only_the_base_layer():
    f = ZV.render_frame(make_layers(), cx=0.0, cy=0.0, zoom=1.0, size=(40, 50))
    assert f.shape == (50, 40, 3) and f.dtype == np.uint8
    assert tuple(f[25, 20]) == (200, 0, 0)


def test_deep_frame_shows_the_fine_layer_in_the_middle_and_base_outside_its_footprint():
    # at zoom 16 the fine layer (half-width 1/8) spans 2 * (1/8) * (frame px per unit) pixels
    f = ZV.render_frame(make_layers(), cx=0.0, cy=0.0, zoom=16.0, size=(96, 96), view_width_units=2.0)
    assert tuple(f[48, 48]) == (0, 0, 200)


def test_frames_are_deterministic():
    a = ZV.render_frame(make_layers(), 0.0, 0.0, 6.0, (32, 40))
    b = ZV.render_frame(make_layers(), 0.0, 0.0, 6.0, (32, 40))
    np.testing.assert_array_equal(a, b)


def test_to_screen_places_a_known_sample_where_the_renderer_draws_it():
    img = np.zeros((9, 9, 3), np.uint8)
    img[2, 6] = (255, 255, 255)  # display row 2 (from the top), column 6: x = -1 + 6 * 0.25 = 0.5, y = 1 - 2 * 0.25 = 0.5
    L = ZV.Layer(center=(0.0, 0.0), half_width=1.0, img=img)
    size = (160, 160)
    for cx, cy in ((0.5, 0.5), (0.4, 0.55)):
        sx, sy = ZV.to_screen(0.5, 0.5, cx, cy, 8.0, size, 2.0)
        f = ZV.render_frame([L], cx, cy, 8.0, size, view_width_units=2.0)
        half = 0.5 * 0.25 * 640  # half a sample = 80 screen px (640 px per plane unit)
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0)):  # corners just inside the block (those in the frame)
            x, y = int(sx + dx * (half - 10)), int(sy + dy * (half - 10))
            if 0 <= x < 160 and 0 <= y < 160:
                assert f[y, x].min() > 200
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # just outside it, still on the 9x9 layer: black
            x, y = int(sx + dx * (half + 10)), int(sy + dy * (half + 10))
            if 0 <= x < 160 and 0 <= y < 160:
                assert f[y, x].max() < 60
    assert ZV.to_screen(0.5, 0.5, 0.5, 0.5, 8.0, size, 2.0) == (80.0, 80.0)


def test_large_blocks_are_drawn_with_hard_edges_not_smoothed():
    img = np.zeros((2, 2, 3), np.uint8)
    img[:, 1] = 255
    L = ZV.Layer(center=(0.0, 0.0), half_width=1.0, img=img)
    f = ZV.render_frame([L], 0.0, 0.0, 1.0, (120, 120), view_width_units=2.0)  # 60 screen px per sample
    row = f[60, :, 0].astype(int)
    assert np.abs(np.diff(row)).max() >= 250  # a black block next to a white one, no ramp between them
    # while a fine, magnified-by-2 pattern stays smooth
    img2 = np.tile(np.array([[0, 255]], np.uint8)[..., None], (16, 8, 3))
    L2 = ZV.Layer(center=(0.0, 0.0), half_width=1.0, img=img2)
    g = ZV.render_frame([L2], 0.0, 0.0, 1.0, (32, 32), view_width_units=2.0)
    assert np.abs(np.diff(g[16, :, 0].astype(int))).max() < 250


def test_shifted_center_moves_a_target_to_the_requested_screen_offset():
    size = (1080, 1350)
    for zoom in (1.0, 37.0, 4000.0):
        cx, cy = ZV.shifted_center(0.3, -0.2, zoom, size, 1.6, (0.0, -245.0))
        sx, sy = ZV.to_screen(0.3, -0.2, cx, cy, zoom, size, 1.6)
        assert abs(sx - 540.0) < 1e-6 and abs(sy - (675.0 - 245.0)) < 1e-6
    assert ZV.shifted_center(0.3, -0.2, 5.0, size, 1.6, (0.0, 0.0)) == (0.3, -0.2)


def test_layers_sampled_from_one_smooth_function_reproduce_it_at_every_zoom():
    """A linear ramp is reproduced exactly by bicubic resampling, so any half-sample offset or scale error between
    layers would show up as a deviation from the ramp."""
    def ramp_layer(centre, hw, n):
        xs = centre[0] - hw + np.arange(n) * (2 * hw / (n - 1))
        row = np.clip(127.5 * (xs + 1.0), 0, 255)
        return ZV.Layer(center=centre, half_width=hw, img=np.tile(row[None, :, None], (n, 1, 3)).round().astype(np.uint8))

    # rounding to uint8 limits the accuracy to ~0.5 level; a half-sample error at the coarse pitch is ~16 levels
    layers = [ramp_layer((0.0, 0.0), 1.0, 33), ramp_layer((0.25, 0.1), 0.25, 33)]
    for zoom, cx in ((1.0, 0.0), (2.0, 0.25), (3.0, 0.25), (6.0, 0.25)):
        f = ZV.render_frame(layers, cx, 0.1, zoom, (120, 90), view_width_units=2.0)
        xs = cx + (np.arange(120) + 0.5 - 60) / (120 / (2.0 / zoom))
        want = np.clip(127.5 * (xs + 1.0), 0, 255)
        err = np.abs(f[45, :, 0].astype(float) - want)
        inner = slice(8, 112)  # away from the frame edge
        assert err[inner].max() < 2.5, (zoom, err[inner].max())
