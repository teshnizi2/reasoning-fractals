#!/usr/bin/env python
"""Basin metrics of rendered fields -> results/metrics.json (method: render/metrics.py, loopscape.metrics).

    .venv/bin/python scripts/compute_metrics.py --hero data/cluster/hero --keyframes data/cluster/kf* --out results/metrics.json

For each field (merged field.npz): fraction solved/censored, number of distinct final grids (attractors) and of
wrong ones, settle-time histogram, and basin entropy / boundary entropy / uncertainty exponent / boundary
dimension (5x5 boxes, eps = 1..5 pixels) of the settle-time labels at several strides, i.e. the same field sampled
every s-th pixel.  Metrics depend on the pixel pitch, so every number is stated with its stride.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import metrics as M  # noqa: E402


def field_report(run: Path, strides: tuple[int, ...]) -> dict:
    z = np.load(run / "field.npz")
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    spec = meta.get("spec", {})
    rep = M.outcome_summary(z["settle_nb"], z["flags"], z["gid"])
    rep.update(
        name=meta.get("name", run.name), shape=list(z["settle_nb"].shape), precision=meta.get("precision"),
        gpu_seconds=meta.get("gpu_seconds"), mean_loops_computed=meta.get("mean_loops_computed"),
        zoom=spec.get("zoom"), center=spec.get("center"), half_width=spec.get("half_width"),
        settle_time_metrics={str(s): M.label_metrics(z["settle_nb"][::s, ::s]) for s in strides if min(z["settle_nb"].shape) // s >= 20},
    )
    # outcome labels (which grid each start ends on) when there is more than one outcome
    if rep["n_outcomes"] > 1:
        rep["outcome_metrics"] = {str(s): M.label_metrics(z["gid"][::s, ::s]) for s in strides[:3]
                                  if min(z["gid"].shape) // s >= 20}
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero")
    ap.add_argument("--keyframes", nargs="*", default=[])
    ap.add_argument("--strides", default="1,2,4,8,10")
    ap.add_argument("--out", default="results/metrics.json")
    a = ap.parse_args()
    strides = tuple(int(s) for s in a.strides.split(","))
    res = {"script": "scripts/compute_metrics.py",
           "method": "basin entropy S_b and boundary entropy S_bb (Daza et al. 2016) over non-overlapping 5x5 boxes; "
                     "uncertainty exponent alpha from the fraction of pixel pairs at separation 1..5 px with different "
                     "labels, boundary dimension 2 - alpha (McDonald et al. 1985); labels = settle time (loops until the "
                     "decoded grid stops changing), as in Lai et al. arXiv:2609.04963; implemented by loopscape.metrics"}
    if a.hero:
        res["hero"] = field_report(ROOT / a.hero, strides)
    keys = sorted(p for pat in a.keyframes for p in glob.glob(str(ROOT / pat)))
    res["keyframes"] = [field_report(Path(p), strides[:2]) for p in keys]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
