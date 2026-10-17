#!/usr/bin/env python
"""Quick look at a partially finished render: compose the tiles that exist (missing tiles stay black).

    alice/sync.sh pull hero && .venv/bin/python scripts/preview_partial.py --run data/cluster/hero --out /tmp/hero_partial.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import colour as C  # noqa: E402
from render import tilerun as TR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--way", default="loopscape")
ap.add_argument("--max", type=int, default=1024, help="longest side of the preview")
a = ap.parse_args()
run = ROOT / a.run
spec = json.loads((run / "spec.json").read_text())
ny, nx = TR.spec_dims(spec)
settle = np.zeros((ny, nx), dtype=np.uint8)
flags = np.zeros((ny, nx), dtype=np.uint8)
have = np.zeros((ny, nx), dtype=bool)
n = 0
for t in TR.plan_tiles(ny, spec["tile"], nx):
    p = run / "tiles" / TR._tile_name(t)
    if p.exists():
        z = np.load(p)
        sl = (slice(t.r0, t.r0 + t.h), slice(t.c0, t.c0 + t.w))
        settle[sl], flags[sl], have[sl] = z["settle_nb"], z["flags"], True
        n += 1
img = C.colourize(settle, flags, C.COLOURWAYS[a.way])[::-1].copy()
img[~have[::-1]] = (14, 17, 23)
im = Image.fromarray(img)
s = a.max / max(im.size)
if s < 1:
    im = im.resize((int(im.width * s), int(im.height * s)), Image.BOX)
im.save(a.out)
print(f"{n} tiles composed ({100 * have.mean():.1f}% of pixels) -> {a.out}")
