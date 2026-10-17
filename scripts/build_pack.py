#!/usr/bin/env python
"""Assemble the explorer's data pack (site/data) from the merged hero and keyframe renders.

    .venv/bin/python scripts/build_pack.py --hero data/cluster/hero --keyframes data/cluster/kf03 data/cluster/kf04 ... --out site/data
    # featured starts + video pair come from the cluster replay: scripts/plan_replay.py -> alice/replay.slurm -> scripts/apply_replay.py
    .venv/bin/python scripts/site_text.py --pack site/data                                  # copy with numbers from results/

Every folder needs its finished tiles (``python -m render.tilerun merge`` is run here if field.npz is missing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import buildpack as BP  # noqa: E402
from render import puzzles as P  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", required=True)
    ap.add_argument("--keyframes", nargs="*", default=[])
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--ways", default="loopscape,series")
    ap.add_argument("--format", default="webp", choices=["webp", "png"])
    a = ap.parse_args()
    dirs = [ROOT / a.hero] + [ROOT / k for k in sorted(a.keyframes)]
    puzzle = dict(question=P.ASSET.question, solution=P.ASSET.answer, rating=P.ASSET.rating, id="7d68927f389fbdaa")
    man = BP.build(dirs, ROOT / a.out, puzzle, a.ways.split(","),
                   meta_extra=dict(precision="float32 (TF32 off), A100 GPUs, early exit after 3 loops on the solution"),
                   image_format=a.format)
    n = len(man["layers"])
    print(f"pack: {n} layers -> {ROOT / a.out}; zoom up to x{max(l['zoom'] for l in man['layers']):.0f}")


if __name__ == "__main__":
    main()
