#!/usr/bin/env python
"""Pick featured starts from the rendered hero, re-run them locally to get loop-by-loop traces, and write them
into the explorer pack's manifest (meta.featured).  Only starts whose local replay reproduces the stored settle
time and outcome are kept, so the replay shown in the explorer is exactly what the stored pixel says.

    .venv/bin/python scripts/build_featured.py --run data/cluster/hero --pack site/data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import featured as F  # noqa: E402
from render import outcomes as O  # noqa: E402
from render import puzzles as P  # noqa: E402
from render import slices as SL  # noqa: E402
from render import sudoku as S  # noqa: E402
from render.engine import Engine  # noqa: E402


def trace_strings(trace: np.ndarray) -> list[str]:
    return ["".join("?" if (v < 0 or v > 9) else ("." if v == 0 else str(int(v))) for v in g) for g in trace]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pack", default="site/data")
    ap.add_argument("--tries", type=int, default=16, help="candidates replayed per kind")
    a = ap.parse_args()
    z = np.load(ROOT / a.run / "field.npz")
    spec = json.loads((ROOT / a.run / "spec.json").read_text())
    assert spec["mode"] == "notebook", "featured starts are picked on the notebook-mode hero"
    s, f = z["settle_nb"], z["flags"]
    n = s.shape[0]
    sol = S.pack([P.ASSET.answer])[0].astype(np.int8)
    ax = SL.axis(n)
    plane = SL.get_plane(0, ROOT / "data/plane_seed0.npz")
    eng = Engine(P.ASSET.question, backend="mlx")
    kinds = [
        ("fast", F.fast(s, f, k=a.tries), lambda T: f"settles in {T} loops", "a quick thought", 6),
        ("slow", F.slow(s, f, k=a.tries * 3), lambda T: f"takes {T} loops", "a slow thought", 24),
        ("pair", F.pair(s, f, k=a.tries * 3, min_gap=6), lambda T: f"next to a pixel {T} loops away", "neighbours that disagree", 40),
        ("wrong", F.wrong(s, f, k=a.tries), lambda T: "ends on a wrong grid", "a wrong ending", 24),
    ]
    out = []
    for kind, cands, title_fn, label, zoom in kinds:
        if not cands:
            print(kind, "no candidates")
            continue
        coords = np.array([[ax[c], ax[r]] for r, c in cands], dtype=np.float32)
        tr = eng.rollout(SL.latents(coords, *plane))
        res = O.analyze(tr, P.ASSET.question, sol)
        for (r, c), t, T, ok, cens in zip(cands, tr, res.settle_nb, res.solved, res.censored):
            same = int(T) == int(s[r, c]) and bool(ok) == bool(f[r, c] & 1) and bool(cens) == bool(f[r, c] & 4)
            if not same:
                continue
            rd, cd = F.to_display(r, c, n)
            gap = int(abs(int(s[r, c]) - int(s[r, c + 1]))) if kind == "pair" else int(T)
            out.append(dict(id=kind, label=label, title=title_fn(gap), layer=0, col=int(cd), row=int(rd), zoom=zoom,
                            settle=int(T), solved=bool(ok), trace=trace_strings(t)))
            print(f"{kind}: kept pixel row {r} col {c} (settle {int(T)})")
            break
        else:
            print(kind, "no candidate replayed identically locally (numerically touchy); skipped")
    man_path = ROOT / a.pack / "manifest.json"
    man = json.loads(man_path.read_text())
    man["meta"]["featured"] = out
    man_path.write_text(json.dumps(man))
    print("wrote", len(out), "featured starts into", man_path)


if __name__ == "__main__":
    main()
