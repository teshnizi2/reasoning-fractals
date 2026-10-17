#!/usr/bin/env python
"""Choose pixels to replay on the cluster (explorer featured starts from the hero, finale pairs from the keyframes).

    .venv/bin/python scripts/plan_replay.py --hero data/cluster/hero --keyframes data/cluster/kf0{3,4,5,6,7}
    rsync results/replay/replay_in_hero.json alice2:reasoning-fractals/data/
    ssh alice2 'cd ~/reasoning-fractals && sbatch --export=ALL,INP=data/replay_in_hero.json,OUT=out/replay_hero alice/replay.slurm'
    alice/sync.sh pull replay_hero                                          # -> data/cluster/replay_hero/replay_out.json

Writes results/replay/plan_NAME.json (what each item is) and results/replay/replay_in_NAME.json (what the job needs); run it
twice (--name hero with --hero, --name kf with --keyframes) to replay the hero as soon as it is done and the keyframes later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import puzzles as P  # noqa: E402
from render import replayplan as RPL  # noqa: E402


def load(run: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    spec = json.loads((run / "spec.json").read_text())
    z = np.load(run / "field.npz")
    return spec, z["settle_nb"], z["flags"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", default=None, help="merged hero run (featured starts + roughness strata); omit for a keyframes-only plan")
    ap.add_argument("--keyframes", nargs="*", default=[])
    ap.add_argument("--out-dir", default="results/replay")
    ap.add_argument("--name", default="all", help="suffix of the written files: plan_NAME.json, replay_in_NAME.json")
    ap.add_argument("--per-kind", type=int, default=12)
    ap.add_argument("--finale-per-layer", type=int, default=60)
    ap.add_argument("--rough-per-bin", type=int, default=150)
    ap.add_argument("--nudged", type=int, default=5)
    ap.add_argument("--eps", type=float, default=1e-6)
    a = ap.parse_args()
    plan: list[dict] = []
    if a.hero:
        spec, s, f = load(ROOT / a.hero)
        plan += RPL.plan_layer("hero", spec, s, f, kinds=("fast", "slow", "wrong", "pair"), per_kind=a.per_kind, pair_min_gap=4, pair_max_gap=15, margin=4)
        plan += RPL.plan_roughness("hero", spec, s, f, per_bin=a.rough_per_bin)  # reproducibility vs local roughness, same engine
    for k in sorted(a.keyframes):
        run = ROOT / k
        spec, s, f = load(run)
        # finale candidates: neighbours far apart in settle time (any layer; the best is chosen after the replays)
        plan += RPL.plan_layer(run.name, spec, s, f, kinds=("pair",), per_kind=a.finale_per_layer, pair_min_gap=4, pair_max_gap=20, margin=24)
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"plan_{a.name}.json").write_text(json.dumps(plan))
    (out / f"replay_in_{a.name}.json").write_text(json.dumps(RPL.replay_input(P.ASSET.question, plan, nudged=a.nudged, eps=a.eps, seed=2026)))
    kinds: dict[str, int] = {}
    for p in plan:
        kinds[f"{p['layer']}:{p['kind']}"] = kinds.get(f"{p['layer']}:{p['kind']}", 0) + 1
    print(len(plan), "items;", kinds)


if __name__ == "__main__":
    main()
