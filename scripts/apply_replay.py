#!/usr/bin/env python
"""Use the cluster's replay results: explorer featured starts (into the pack manifest) and the video's finale pair.

    .venv/bin/python scripts/apply_replay.py --pack site/data --keyframes data/cluster/kf03 ... data/cluster/kf07

Reads results/replay/plan_*.json and data/cluster/replay_*/replay_out.json (alice/sync.sh pull replay_hero / replay_kf); writes
results/replay/summary.json, results/video/pair.json and meta.featured of the pack.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import puzzles as P  # noqa: E402
from render import replayapply as RA  # noqa: E402
from render import tilerun as TR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", nargs="+", default=["results/replay/plan_hero.json", "results/replay/plan_kf.json"])
    ap.add_argument("--replay", nargs="+", default=["data/cluster/replay_hero/replay_out.json", "data/cluster/replay_kf/replay_out.json"])
    ap.add_argument("--pack", default="site/data")
    ap.add_argument("--keyframes", nargs="*", default=[])
    ap.add_argument("--pair-out", default="results/video/pair.json")
    ap.add_argument("--summary", default="results/replay/summary.json")
    ap.add_argument("--min-robust-gap", type=int, default=3)
    ap.add_argument("--criterion", default="median", choices=["strict", "median"], help="strict: every re-run keeps the gap; median: median gap and order in all but one run")
    ap.add_argument("--rank", default="depth", choices=["depth", "robust"], help="depth: the finest layer that passes wins")
    ap.add_argument("--loop-seconds", type=float, default=0.28)
    a = ap.parse_args()
    assert len(a.plan) == len(a.replay), "one replay output per plan"
    plan = [p for f in a.plan for p in json.loads((ROOT / f).read_text())]
    reps = [json.loads((ROOT / f).read_text()) for f in a.replay]
    rep = dict(meta=reps[0]["meta"])  # engine/nudges/eps: the same job script for every output
    res = [r for x in reps for r in x["items"]]
    man_path = ROOT / a.pack / "manifest.json"
    man = json.loads(man_path.read_text())
    ny = man["layers"][0]["ny"]

    feats = RA.featured(plan, res, ny)
    man["meta"]["featured"] = feats
    man_path.write_text(json.dumps(man))
    print("featured:", [(f["id"], f["settle"]) for f in feats])

    pitches = {}
    for k in a.keyframes:
        spec = json.loads((ROOT / k / "spec.json").read_text())
        pitches[Path(k).name] = 2 * spec["half_width"] / (TR.spec_dims(spec)[1] - 1)
    # the finest layer with a pair whose slow/fast order survives the nudges (median gap >= min_robust_gap, order kept in all but one run)
    best = RA.pick_finale(plan, res, pitches, P.ASSET.answer, min_robust_gap=a.min_robust_gap, criterion=a.criterion, rank=a.rank)
    print("finale pair:", None if best is None else (best["layer"], best["a"]["settle"], best["b"]["settle"], best["verified"]))

    by_id = {r["id"]: r for r in res}
    kinds: dict[str, dict] = {}
    for p in plan:
        r = by_id.get(p["id"])
        if r is None:
            continue
        if p["kind"].startswith("rough"):
            continue
        k = kinds.setdefault(f"{p['layer'] if p['layer'] == 'hero' else 'keyframes'}:{p['kind']}", dict(items=0, reproduced=0, nudge_stable=0))
        k["items"] += 1
        k["reproduced"] += int(r["equal"])
        k["nudge_stable"] += int(all(v == r["settle"] for v in r["nudged"]))
    summary = dict(script="scripts/apply_replay.py", engine=rep["meta"]["engine"], nudged=rep["meta"]["nudged"], eps=rep["meta"]["eps"],
                   items=len(res), reproduced=int(sum(r["equal"] for r in res)), by_kind=kinds, roughness=RA.roughness_summary(plan, res),
                   finale=None if best is None else {k: best[k] for k in ("layer", "gap", "pitch", "verified")})
    out = ROOT / a.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if best is None:
        sys.exit("no finale pair survived the replay checks; relax --min-robust-gap or add candidates")
    best.update(puzzle=P.ASSET.question, solution=P.ASSET.answer, loop_seconds=a.loop_seconds,
                loops_shown=int(min(24, max(best["a"]["settle"], best["b"]["settle"]) + 3)))
    best["verified"].update(engine=rep["meta"]["engine"], eps=rep["meta"]["eps"])
    po = ROOT / a.pair_out
    po.parent.mkdir(parents=True, exist_ok=True)
    po.write_text(json.dumps(best))
    print(f"wrote {po}: A {best['a']['settle']} loops vs B {best['b']['settle']} loops in {best['layer']}")


if __name__ == "__main__":
    main()
