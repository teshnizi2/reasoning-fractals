"""Video overlay: zoom counter text and caption compositing."""
import numpy as np

from render import overlay as OV


def test_zoom_label_formats_magnification():
    assert OV.zoom_label(1.0) == "×1"
    assert OV.zoom_label(7.4) == "×7.4"
    assert OV.zoom_label(64.0) == "×64"
    assert OV.zoom_label(1500.0) == "×1,500"
    assert OV.zoom_label(4096.0) == "×4,096"


def test_caption_alpha_fades_in_holds_and_fades_out():
    f = lambda t: OV.caption_alpha(t, t_in=1.0, t_out=4.0, fade=0.5)  # noqa: E731
    assert f(0.5) == 0.0 and f(1.0) == 0.0
    assert 0.0 < f(1.25) < 1.0 and f(1.6) == 1.0 and f(3.0) == 1.0
    assert 0.0 < f(4.25) < 1.0 and f(5.0) == 0.0


def test_draw_text_changes_only_the_text_region_and_returns_a_copy():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    out = OV.draw_text(frame, "hello", (20, 20), size=24, font="sans", color=(255, 255, 255), alpha=1.0)
    assert out is not frame and out.shape == frame.shape and out.dtype == np.uint8
    assert frame.sum() == 0  # input untouched
    ys, xs = np.nonzero(out.sum(-1))
    assert ys.min() >= 15 and ys.max() < 70 and xs.min() >= 15 and xs.max() < 200
    assert OV.draw_text(frame, "hello", (20, 20), size=24, font="sans", color=(255, 255, 255), alpha=0.0).sum() == 0


def test_mono_and_sans_fonts_both_load():
    for name in ("sans", "mono", "sans-bold"):
        assert OV.load_font(name, 20) is not None
