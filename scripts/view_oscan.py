#!/usr/bin/env python
"""Contact sheet of an outcome-scan run (alice/outcome_scan.py): one tile per (puzzle, seed, z_L mode, h).

    alice/sync.sh pull oscan && .venv/bin/python scripts/view_oscan.py --run data/cluster/oscan --out /tmp/oscan.png
Pixels: settle-time ramp where solved; the wrong-answer ramp where the model ended on another grid; white if still
changing at loop 24.  The title of each tile gives puzzle, seed, z_L mode, half-width h and the solved fraction.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import colour as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--scale", type=int, default=4)
a = ap.parse_args()
run = ROOT / a.run
rows = json.loads((run / "oscan.json").read_text())
cw = C.COLOURWAYS["loopscape"]
tiles = []
for r in rows:
    f = run / f"{r['puzzle']}_s{r['seed']}_{r['zl']}_h{r['h']:g}.npz"
    z = np.load(f)
    flags = z["solved"].astype(np.uint8) | (z["censored"].astype(np.uint8) << 2)
    img = C.colourize(z["settle_nb"], flags, cw)[::-1]
    tiles.append((np.repeat(np.repeat(img, a.scale, 0), a.scale, 1),
                  f"{r['puzzle']} s{r['seed']} zl={r['zl']} h={r['h']:g}  ok {100 * r['frac_solved']:.0f}%"))
n = tiles[0][0].shape[0]
cols = a.cols
nr = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * (n + 8), nr * (n + 26)), (14, 17, 23))
d = ImageDraw.Draw(sheet)
for i, (im, title) in enumerate(tiles):
    x, y = (i % cols) * (n + 8), (i // cols) * (n + 26)
    d.text((x + 2, y + 4), title, fill=(232, 236, 245))
    sheet.paste(Image.fromarray(im), (x, y + 20))
sheet.save(a.out)
print(len(tiles), "tiles ->", a.out, sheet.size)
