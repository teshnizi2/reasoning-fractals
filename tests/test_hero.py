"""Hero crop geometry and resampling (scripts/make_hero.py)."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("make_hero", Path(__file__).resolve().parents[1] / "scripts/make_hero.py")
MH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MH)


def test_crop_box_stays_inside_the_image_and_has_the_requested_aspect():
    for cx, cy, hw in [(0, 0, 0.8), (0.9, 0.9, 0.5), (-1, -1, 0.7)]:
        x0, y0, x1, y1 = MH.crop_box(2048, cx, cy, hw, 1.25)
        assert 0 <= x0 < x1 <= 2048 and 0 <= y0 < y1 <= 2048
        assert abs((y1 - y0) / (x1 - x0) - 1.25) < 0.01


def test_crop_box_is_centred_when_it_fits_and_y_is_up():
    x0, y0, x1, y1 = MH.crop_box(1001, 0.0, 0.0, 0.5, 1.0)  # 1001 px over [-1, 1]: 500 px per unit
    assert (x0 + x1) // 2 == 500 and (y0 + y1) // 2 == 500
    _, ya, _, yb = MH.crop_box(1001, 0.0, 0.8, 0.2, 1.0)  # a window centred high in y is near the top rows
    assert (ya + yb) / 2 < 150


def test_to_size_returns_the_requested_size_and_keeps_flat_colours():
    img = np.full((250, 250, 3), 90, dtype=np.uint8)
    for w, h in [(1080, 1350), (500, 500), (125, 125)]:
        out = MH.to_size(img, w, h)
        assert out.shape == (h, w, 3) and abs(int(out[h // 2, w // 2, 0]) - 90) <= 1
