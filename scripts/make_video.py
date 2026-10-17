#!/usr/bin/env python
"""Deep-zoom video from a data pack (nested keyframe images), with an optional pair finale.

    .venv/bin/python scripts/make_video.py --pack site/data --way series --out media/zoom-1080x1350.mp4 \
        --captions scripts/video_captions.json --pair data/pair.json --square

LinkedIn feed format 1080x1350 (4:5), yuv420p, H.264, no audio.  ``--square`` also writes a centre-cropped 1080x1080
cut.  The camera flies from the whole slice to the deepest keyframe and then parks; with ``--pair`` the last stretch
shows two neighbouring starts as big blocks (one pixel = one thought) with their Sudoku grids replaying beneath.
``--stills 0,6,12`` writes PNGs of those seconds (next to the video) instead of encoding anything.
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
from render import finale as FI  # noqa: E402
from render import overlay as OV  # noqa: E402
from render import zoomvideo as ZV  # noqa: E402

BG = (14, 17, 23)
AMBER = (245, 158, 11)
MUTED = (136, 147, 171)
PAIR_Y = 400  # screen y of the pair's centre in the 1080x1350 frame (render/finale.py PORTRAIT)
CREDIT = "EqR (Huang, Geng, Kolter) · loopscape (Lai, Bao, Quinn, Gilpin)"


def load_layers(pack: Path, way: str) -> tuple[list[ZV.Layer], dict]:
    man = json.loads((pack / "manifest.json").read_text())
    layers = []
    for m in sorted(man["layers"], key=lambda m: -m["half_width"]):
        img = np.asarray(Image.open(pack / m["images"][way]).convert("RGB"))
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
    ap.add_argument("--view-width", type=float, default=2.0, help="plane units across the frame at zoom 1 (2 = the whole slice)")
    ap.add_argument("--end-px", type=float, default=None, help="screen px per sample of the deepest layer at the end (default 56 with a pair, else 2)")
    ap.add_argument("--hold", type=float, default=1.2, help="seconds parked on the first frame")
    ap.add_argument("--finale", type=float, default=7.5, help="seconds parked at the end (the pair finale, or a plain hold)")
    ap.add_argument("--pair", default=None, help="pair.json from scripts/build_pair.py")
    ap.add_argument("--captions", default=None, help="JSON list of {text,size,font,color,x,y,t_in,t_out}; {a_settle}/{b_settle} come from --pair")
    ap.add_argument("--no-counter", action="store_true")
    ap.add_argument("--no-credit", action="store_true")
    ap.add_argument("--square", action="store_true", help="also write the centre-cropped 1080x1080 cut")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--stills", default=None, help="comma-separated seconds: write PNGs of those frames and stop")
    ap.add_argument("--stills-dir", default=None)
    a = ap.parse_args()

    pack = ROOT / a.pack
    layers, man = load_layers(pack, a.way)
    w, h = (int(v) for v in a.size.split("x"))
    pair = json.loads((ROOT / a.pair).read_text()) if a.pair else None
    deep = layers[-1]
    end_px = a.end_px if a.end_px else (56.0 if pair else 2.0)
    zoom_end = (end_px / deep.pitch) / (w / a.view_width)
    c_end = (pair["a"]["x"] / 2 + pair["b"]["x"] / 2, pair["a"]["y"] / 2 + pair["b"]["y"] / 2) if pair else deep.center
    cam = ZV.Camera(c0=layers[0].center, c1=c_end, zoom0=1.0, zoom1=max(zoom_end, 2.0))
    move = a.seconds - a.hold - a.finale
    assert move > 3, "not enough time for the move"
    ts = ZV.timeline(a.fps, a.hold, move, a.finale)
    t_arrive = a.hold + move
    if pair:
        pair = dict(pair, t_in=t_arrive + 0.5)
    caps = json.loads((ROOT / a.captions).read_text()) if a.captions else []
    fmt = {"a_settle": pair["a"]["settle"] if pair else "", "b_settle": pair["b"]["settle"] if pair else ""}
    for c in caps:
        c["text"] = c["text"].format(**fmt)

    def view_at(u: float, size: tuple[int, int]) -> tuple[float, float, float, float]:
        cx, cy, z = cam.at(u)
        if pair:  # slide the pair up to PAIR_Y (in the 1350-high frame) during the last part of the move
            k = size[1] / 1350.0
            m = ZV._smooth(0.5, 1.0, u * u * (3 - 2 * u))
            cx, cy = ZV.shifted_center(cx, cy, z, size, a.view_width, (0.0, (PAIR_Y - 1350 / 2) * k * m))
        return cx, cy, z, a.view_width

    def overlay(frame: np.ndarray, sec: float, view: tuple, square: bool) -> np.ndarray:
        fh, fw = frame.shape[:2]
        k = fw / 1080.0
        fin = 0.0 if not pair else min(1.0, max(0.0, (sec - pair["t_in"]) / FI.CARD_FADE))
        if pair:
            frame = FI.draw_finale(frame, pair, sec, view, FI.scaled(FI.SQUARE if square else FI.PORTRAIT, k))
        if not a.no_counter and fin < 1.0:
            frame = OV.draw_caption(frame, f"zoom {OV.zoom_label(view[2])}", (int(54 * k), fh - int(168 * k)), int(36 * k), "mono", AMBER, alpha=1.0 - fin)
        for c in caps:
            al = OV.caption_alpha(sec, c["t_in"], c["t_out"])
            y = c.get("y_square", c["y"]) if square else c["y"]
            if al > 0:
                frame = OV.draw_caption(frame, c["text"], (int(c["x"] * k), int(y * k)), int(c["size"] * k), c.get("font", "sans"),
                                        tuple(c.get("color", (232, 236, 245))), alpha=al)
        if not a.no_credit:
            frame = OV.draw_caption(frame, CREDIT, (int(54 * k), fh - int(66 * k)), int(21 * k), "mono", (170, 180, 202), alpha=1.0, bg_alpha=0.62)
        return frame

    raw_cache: dict[float, np.ndarray] = {}

    def raw_at(u: float) -> np.ndarray:  # the camera is parked at both ends: render those two frames once
        if u in (0.0, 1.0) and u in raw_cache:
            return raw_cache[u]
        view = view_at(u, (w, h))
        raw = ZV.render_frame(layers, view[0], view[1], view[2], (w, h), view_width_units=a.view_width, background=BG)
        if u in (0.0, 1.0):
            raw_cache[u] = raw
        return raw

    def frame_at(i: int):
        u = ts[i]
        sec = i / a.fps
        view = view_at(u, (w, h))
        raw = raw_at(u)
        full = overlay(raw, sec, view, square=False)
        top = (h - w) // 2
        sq = overlay(np.ascontiguousarray(raw[top : top + w]), sec, view, square=True) if a.square or a.stills else None
        return full, sq

    out = ROOT / a.out
    if a.stills:
        d = Path(a.stills_dir) if a.stills_dir else out.parent / "stills"
        d.mkdir(parents=True, exist_ok=True)
        for s in (float(v) for v in a.stills.split(",")):
            i = min(len(ts) - 1, int(round(s * a.fps)))
            full, sq = frame_at(i)
            Image.fromarray(full).save(d / f"frame_{s:05.2f}.png")
            Image.fromarray(sq).save(d / f"square_{s:05.2f}.png")
            print("still", s, "zoom x%.1f" % view_at(ts[i], (w, h))[2], flush=True)
        return

    out.parent.mkdir(parents=True, exist_ok=True)

    def encoder(path: Path, size: tuple[int, int]) -> subprocess.Popen:
        return subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
             "-r", str(a.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    p1 = encoder(out, (w, h))
    p2 = encoder(out.with_name(out.stem.replace(f"{w}x{h}", "1080x1080") + ".mp4"), (w, w)) if a.square else None
    for i in range(len(ts)):
        full, sq = frame_at(i)
        p1.stdin.write(full.tobytes())
        if p2:
            p2.stdin.write(np.ascontiguousarray(sq).tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{len(ts)}  zoom x{cam.at(ts[i])[2]:.1f}", flush=True)
    for p in (p1, p2):
        if p:
            p.stdin.close()
            p.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
