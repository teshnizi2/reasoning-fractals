#!/usr/bin/env python
"""Time/cost estimates for full renders from the measured throughputs -> results/cost_estimate.json.

Reads results/throughput_m4.json (this Mac), results/alice/smoke_5246129_bench.json (A100) and, if present,
results/alice/eechk_*.json (early exit); every rate used is printed with its source.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


m4 = load("results/throughput_m4.json")
smoke = load("results/alice/smoke_5246129_bench.json")
ee = next(iter(sorted((ROOT / "results/alice").glob("eechk_*.json"))), None)
ee = json.loads(ee.read_text()) if ee else None

rates = {}
if m4:
    best = {}
    for p in m4["points"]:
        best[p["label"]] = max(best.get(p["label"], 0), p["rollouts_per_s"])
    rates["m4_mlx_fp32"] = dict(rollouts_per_s=best.get("mlx"), source="results/throughput_m4.json (best batch)")
    rates["m4_mps_torch_fp32"] = dict(rollouts_per_s=best.get("torchfn-mps"), source="results/throughput_m4.json")
if smoke:
    rates["a100_fp32_eager"] = dict(rollouts_per_s=smoke["throughput"]["fp32"]["B1024"]["rollouts_per_s"], source="smoke job 5246129, B=1024")
mean_loops = None
if ee:
    mean_loops = ee["early_B1024"]["mean_loops"]
    rates["a100_mig4g_fp32_full"] = dict(rollouts_per_s=ee["full_B1024_a"]["rollouts_per_s"], source="job 5246309 (A100 MIG 4g.40gb)")

jobs = {
    "1000x1000 (the upstream asset's resolution)": 1000 * 1000,
    "2048x2048 hero": 2048 * 2048,
    "zoom: 10 keyframes of 540x675": 10 * 540 * 675,
}
steps = 24
out = dict(script="scripts/cost_estimate.py", loops_per_rollout=steps, early_exit_mean_loops=mean_loops, rates=rates, estimates={})
for name, px in jobs.items():
    row = {}
    for rname, r in rates.items():
        if r["rollouts_per_s"]:
            row[rname] = dict(hours=px / r["rollouts_per_s"] / 3600)
            if mean_loops and rname.startswith(("a100", "mig")) or (mean_loops and "mig" in rname):
                row[rname + "_early_exit"] = dict(hours=px * (mean_loops / steps) / r["rollouts_per_s"] / 3600)
    out["estimates"][name] = dict(pixels=px, hours=row)
(ROOT / "results/cost_estimate.json").write_text(json.dumps(out, indent=1))
for name, e in out["estimates"].items():
    print(name, {k: round(v["hours"], 1) for k, v in e["hours"].items()})
