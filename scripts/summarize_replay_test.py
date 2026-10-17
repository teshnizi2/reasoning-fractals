#!/usr/bin/env python
"""Summary of the replay smoke test (job 5246629): does a plain replay on the render engine reproduce stored settle times?

    alice/sync.sh pull replaytest    # (the job wrote out/replaytest)
    .venv/bin/python scripts/summarize_replay_test.py --plan data/cluster/replay_test_plan.json --out results/alice/replay_smoke_5246629.json

Items: featured-start candidates of the partially finished hero plus random solved pixels stratified by local roughness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="data/cluster/replay_test_plan.json")
    ap.add_argument("--replay", default="data/cluster/replaytest/replay_out.json")
    ap.add_argument("--out", default="results/alice/replay_smoke_5246629.json")
    a = ap.parse_args()
    plan = json.loads((ROOT / a.plan).read_text())
    rep = json.loads((ROOT / a.replay).read_text())
    by = {r["id"]: r for r in rep["items"]}
    kinds: dict[str, dict] = {}
    mism = []
    for p in plan:
        r = by[p["id"]]
        d = kinds.setdefault(p["kind"], dict(n=0, reproduced=0, nudge_stable=0, max_change=[], stored_settle=[]))
        d["n"] += 1
        d["reproduced"] += int(r["equal"])
        d["nudge_stable"] += int(all(v == r["settle"] for v in r["nudged"]))
        d["max_change"].append(max(abs(v - r["settle"]) for v in r["nudged"]))
        d["stored_settle"].append(p["stored"]["settle"])
        if not r["equal"]:
            mism.append(dict(id=p["id"], stored=p["stored"]["settle"], replay=r["settle"]))
    out = dict(script="scripts/summarize_replay_test.py", job="5246629", engine=rep["meta"]["engine"], items=rep["meta"]["items"],
               nudged=rep["meta"]["nudged"], eps=rep["meta"]["eps"], reproduced=rep["meta"]["equal"],
               by_kind={k: dict(n=v["n"], reproduced=v["reproduced"], nudge_stable=v["nudge_stable"], mean_max_change=float(np.mean(v["max_change"])),
                                mean_stored_settle=float(np.mean(v["stored_settle"]))) for k, v in kinds.items()},
               mismatches=mism)
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "mismatches"}, indent=1))
    print(len(mism), "mismatches, stored settle times:", sorted({m["stored"] for m in mism}))


if __name__ == "__main__":
    main()
