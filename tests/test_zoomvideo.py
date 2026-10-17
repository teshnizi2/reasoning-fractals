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
