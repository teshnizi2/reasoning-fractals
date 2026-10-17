#!/usr/bin/env python
"""Summaries of the native-scale scans (puzzle scan, long-run check, outcome scan) -> results/scans.json.

    .venv/bin/python scripts/summarize_scans.py            # after alice/sync.sh pull pscan / longrun / oscan

pscan  (job 5246311): 5565 Sudoku-Extreme puzzles x 16 EqR-style random starts (truncated normal, norm ~223), 24 loops.
longrun (job 5246477): 400 starts that ended unsolved in the scan, re-run for 96 loops.
oscan  (job 5246477): small outcome maps (48x48) of 2-D slices around native-scale base points.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSET = "2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3..."


def main() -> None:
    out: dict = {"script": "scripts/summarize_scans.py"}
    d = json.loads((ROOT / "data/cluster/pscan/pscan.json").read_text())
    ps = d["puzzles"]
    fs = np.array([p["frac_solved"] for p in ps])
    rt = np.array([p["rating"] for p in ps])
    bins = [(0, 1), (1, 5), (5, 20), (20, 100), (100, 10**6)]
    asset = next(p for p in ps if p["q"] == ASSET)
    out["pscan"] = dict(
        job="5246311", n_puzzles=len(ps), k=d["k"], steps=d["steps"], seconds=d["seconds"], frac_solved=float(fs.mean()),
        always_solved=int((fs == 1).sum()), never_solved=int((fs == 0).sum()), partial=int(((fs > 0) & (fs < 1)).sum()),
        multistable=int(((fs >= 0.15) & (fs <= 0.85)).sum()),
        by_rating=[dict(lo=lo, hi=hi, n=int(((rt >= lo) & (rt < hi)).sum()), frac_solved=float(fs[(rt >= lo) & (rt < hi)].mean()),
                        always=float((fs[(rt >= lo) & (rt < hi)] == 1).mean())) for lo, hi in bins if ((rt >= lo) & (rt < hi)).any()],
        asset=dict(rating=asset["rating"], frac_solved=asset["frac_solved"], mean_settle=asset["mean_settle"]),
        note="starts are EqR's own initialisation (truncated normal), not the notebook's near-zero plane",
    )
    lr = json.loads((ROOT / "data/cluster/longrun/longrun.json").read_text())
    out["longrun"] = dict(job="5246477", **lr["summary"],
                          note="same latents as the scan, but a different batch composition than the scan run: native-scale outcomes are "
                               "chaotic, so individual starts need not repeat their scan outcome")
    op = ROOT / "data/cluster/oscan/oscan.json"
    if op.exists():
        rows = json.loads(op.read_text())
        by = {}
        for r in rows:
            by.setdefault(r["puzzle"], []).append(r)
        out["oscan"] = dict(job="5246477", windows=len(rows),
                            puzzles={k: dict(windows=len(v), frac_solved_min=min(x["frac_solved"] for x in v), frac_solved_max=max(x["frac_solved"] for x in v),
                                             frac_censored_max=max(x["frac_censored"] for x in v), distinct_final_max=max(x["n_distinct_final"] for x in v))
                                     for k, v in by.items()},
                            rows=rows)
    (ROOT / "results/scans.json").write_text(json.dumps(out, indent=1))
    print("wrote results/scans.json", {k: v for k, v in out["pscan"].items() if k in ("n_puzzles", "frac_solved", "always_solved", "multistable")})


if __name__ == "__main__":
    main()
