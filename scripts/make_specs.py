#!/usr/bin/env python
"""Write the production render specs (hero + zoom keyframes) into alice/specs/.

    .venv/bin/python scripts/make_specs.py --hero
    .venv/bin/python scripts/make_specs.py --keyframes --cx 0.88 --cy 0.87 --k-min 2 --k-max 12

Every spec: the asset puzzle, plane seed 0 (frozen file), true fp32, early exit after 3 loops on the solution.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import puzzles as P  # noqa: E402

COMMON = dict(puzzle=P.ASSET.question, steps=24, plane_seed=0, plane_file="data/plane_seed0.npz", precision="fp32",
              backend="torchfn", device="cuda", compile=False, early_exit={"m": 3}, batch=2048)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", action="store_true")
    ap.add_argument("--hero-n", type=int, default=2048)
    ap.add_argument("--keyframes", action="store_true")
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--nx", type=int, default=540)
    ap.add_argument("--ny", type=int, default=675)
    ap.add_argument("--tile", type=int, default=135)
    a = ap.parse_args()
    out = ROOT / "alice/specs"
    out.mkdir(exist_ok=True)
    if a.hero:
        spec = dict(COMMON, name="hero", mode="notebook", n=a.hero_n, tile=128,
                    note="loopscape's mesh(n) over [-1,1]^2 at RES=n: float32 axis and latent arithmetic, as in the notebook")
        (out / "hero.json").write_text(json.dumps(spec, indent=1))
        print("wrote hero.json", a.hero_n)
    if a.keyframes:
        for k in range(a.k_min, a.k_max + 1):
            spec = dict(COMMON, name=f"kf{k:02d}", mode="window", center=[a.cx, a.cy], half_width=2.0 ** -k, nx=a.nx, ny=a.ny,
                        tile=a.tile, latent_precision="float64", zoom=2 ** k)
            (out / f"kf{k:02d}.json").write_text(json.dumps(spec, indent=1))
        print("wrote keyframe specs", a.k_min, "..", a.k_max, "centre", a.cx, a.cy)


if __name__ == "__main__":
    main()
