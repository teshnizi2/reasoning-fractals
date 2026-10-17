"""Text overlays for the video: zoom counter and fading captions (system fonts, no downloads)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONTS = {
    "sans": ("/System/Library/Fonts/Avenir Next.ttc", 5),        # Medium
    "sans-bold": ("/System/Library/Fonts/Avenir Next.ttc", 2),   # Demi Bold
    "mono": ("/System/Library/Fonts/Menlo.ttc", 0),
}


def load_font(name: str, size: int):
    path, index = _FONTS[name]
    try:
        return ImageFont.truetype(path, size, index=index)
    except OSError:  # no system font (e.g. on Linux): fall back to PIL's built-in font
        return ImageFont.load_default(size)


def zoom_label(zoom: float) -> str:
    if zoom < 10:
        s = f"{zoom:.1f}".rstrip("0").rstrip(".")
        return f"×{s}"
    return f"×{int(round(zoom)):,}"


def caption_alpha(t: float, t_in: float, t_out: float, fade: float = 0.5) -> float:
    """0 before t_in, ramps up over ``fade`` seconds, holds, ramps down ending at t_out + fade."""
    if t <= t_in or t >= t_out + fade:
        return 0.0
    if t < t_in + fade:
        return (t - t_in) / fade
    if t <= t_out:
        return 1.0
    return 1.0 - (t - t_out) / fade


def draw_text(frame: np.ndarray, text: str, xy: tuple[int, int], size: int, font: str, color: tuple[int, int, int],
              alpha: float = 1.0, shadow: bool = True) -> np.ndarray:
    """Return a copy of ``frame`` with ``text`` drawn at xy (top-left) with opacity ``alpha`` and a soft shadow."""
    if alpha <= 0.0:
        return frame.copy()
    base = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = load_font(font, size)
    a = int(round(255 * min(1.0, alpha)))
    if shadow:
        for dx, dy in ((0, 2), (0, 3), (1, 2), (-1, 2)):
            d.text((xy[0] + dx, xy[1] + dy), text, font=f, fill=(6, 8, 12, int(a * 0.55)))
    d.text(xy, text, font=f, fill=(*color, a))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB")).copy()


def draw_caption(frame: np.ndarray, text: str, xy: tuple[int, int], size: int, font: str, color: tuple[int, int, int],
                 alpha: float = 1.0, bg: tuple[int, int, int] = (14, 17, 23), bg_alpha: float = 0.80, line_gap: float = 1.3) -> np.ndarray:
    """``text`` (lines split on newlines) at xy (top-left of the first line) on a rounded dark pill, so it stays legible
    over bright parts of the fractal.  Returns a copy; ``alpha`` fades pill and text together."""
    if alpha <= 0.0:
        return frame.copy()
    base = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = load_font(font, size)
    lines = text.split("\n")
    boxes = [d.textbbox((xy[0], xy[1] + int(i * size * line_gap)), ln, font=f) for i, ln in enumerate(lines)]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad = int(size * 0.45)
    a = min(1.0, alpha)
    d.rounded_rectangle([x0 - pad, y0 - int(pad * 0.8), x1 + pad, y1 + int(pad * 0.8)], radius=int(pad * 0.9),
                        fill=(*bg, int(round(255 * bg_alpha * a))))
    for i, ln in enumerate(lines):
        d.text((xy[0], xy[1] + int(i * size * line_gap)), ln, font=f, fill=(*color, int(round(255 * a))))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB")).copy()
