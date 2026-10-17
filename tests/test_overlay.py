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


def test_caption_pill_sits_behind_the_text_and_grows_with_each_line():
    frame = np.full((300, 600, 3), (200, 230, 215), dtype=np.uint8)  # bright mint: the case that needs a pill
    one = OV.draw_caption(frame, "A small model", (40, 40), 36, "sans", (232, 236, 245))
    two = OV.draw_caption(frame, "A small model\nthinks in loops", (40, 40), 36, "sans", (232, 236, 245))
    assert (frame == (200, 230, 215)).all()  # input untouched
    dark = lambda im: ((im.astype(int).sum(-1)) < 300)  # noqa: E731  (pill is dark, the frame is bright)
    ys1, xs1 = np.nonzero(dark(one))
    ys2, _ = np.nonzero(dark(two))
    assert ys1.size > 2000 and xs1.min() < 40 and xs1.max() > 40 + 150  # pill extends around the text
    assert ys2.max() - ys2.min() > (ys1.max() - ys1.min()) + 25  # second line makes it taller
    assert OV.draw_caption(frame, "x", (40, 40), 36, "sans", (232, 236, 245), alpha=0.0).tolist() == frame.tolist()
    # text pixels are light on the dark pill
    assert one[40:80, 40:300].max() > 225


def test_draw_rect_outline_draws_only_the_outline_with_the_given_alpha():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    out = OV.draw_rect_outline(frame, (20, 30, 100, 90), (200, 100, 0), alpha=0.5, width=2)
    assert frame.sum() == 0 and out is not frame
    assert out[30, 50].tolist() == [100, 50, 0]          # on the top edge: half the colour
    assert out[60, 60].sum() == 0                          # the inside is untouched
    assert OV.draw_rect_outline(frame, (20, 30, 100, 90), (200, 100, 0), alpha=0.0).sum() == 0
    off = OV.draw_rect_outline(frame, (-500, -500, 5000, 5000), (200, 100, 0), alpha=1.0, width=2)  # edges far off screen: nothing, no crash
    assert off.sum() == 0
