#!/usr/bin/env python
"""Fly-in video from a data pack (nested keyframe images): scale + crossfade between keyframes, ffmpeg H.264.

    .venv/bin/python scripts/make_video.py --pack site/data --way series --out media/zoom-1080x1350.mp4

LinkedIn feed format 1080x1350 (4:5), yuv420p, no audio.  ``--square`` also writes a centre-cropped 1080x1080 cut.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import zoomvideo as ZV  # noqa: E402


def load_layers(pack: Path, way: str) -> tuple[list[ZV.Layer], dict]:
    man = json.loads((pack / "manifest.json").read_text())
    layers = []
    for m in sorted(man["layers"], key=lambda m: -m["half_width"]):
        img = np.asarray(Image.open(pack / m["images"][way]).convert("RGB"))
        # the image's width spans nx samples; ZV.Layer wants half_width in plane units over nx samples
        layers.append(ZV.Layer(center=tuple(m["center"]), half_width=m["half_width"], img=img))
    return layers, man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="site/data")
    ap.add_argument("--way", default="series")
    ap.add_argument("--out", default="media/zoom-1080x1350.mp4")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1080x1350")
    ap.add_argument("--view-width", type=float, default=1.6, help="plane units across the frame at zoom 1")
    ap.add_argument("--end-px", type=float, default=2.0, help="screen px per sample of the deepest layer at the end")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds to hold on the first and last frame")
    ap.add_argument("--square", action="store_true")
    ap.add_argument("--crf", type=int, default=16)
    a = ap.parse_args()

    pack = ROOT / a.pack
    layers, man = load_layers(pack, a.way)
    w, h = (int(v) for v in a.size.split("x"))
    deep = layers[-1]
    # end zoom: deepest layer shown at `end_px` screen pixels per sample
    zoom_end = (a.end_px / deep.pitch) / (w / a.view_width)
    cam = ZV.Camera(c0=layers[0].center, c1=deep.center, zoom0=1.0, zoom1=max(zoom_end, 2.0))
    n_move = int(round((a.seconds - 2 * a.hold) * a.fps))
    n_hold = int(round(a.hold * a.fps))
    ts = [0.0] * n_hold + list(np.linspace(0, 1, n_move)) + [1.0] * n_hold
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)

    def encoder(path: Path, size: tuple[int, int]) -> subprocess.Popen:
        return subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
             "-r", str(a.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    p1 = encoder(out, (w, h))
    p2 = encoder(out.with_name(out.stem.replace(f"{w}x{h}", "1080x1080") + ".mp4"), (w, w)) if a.square else None
    cache: dict[float, np.ndarray] = {}
    for i, t in enumerate(ts):
        if t not in cache or t not in (0.0, 1.0):
            cache = {} if t not in (0.0, 1.0) else cache
            cx, cy, z = cam.at(t)
            cache[t] = ZV.render_frame(layers, cx, cy, z, (w, h), view_width_units=a.view_width)
        f = cache[t]
        p1.stdin.write(f.tobytes())
        if p2:
            top = (h - w) // 2
            p2.stdin.write(np.ascontiguousarray(f[top : top + w]).tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{len(ts)}  zoom x{cam.at(t)[2]:.1f}", flush=True)
    for p in (p1, p2):
        if p:
            p.stdin.close()
            p.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
