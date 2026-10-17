#!/usr/bin/env python
"""Pick the pair of neighbouring starts for the video finale and replay both locally, loop by loop.

    .venv/bin/python scripts/build_pair.py --run data/cluster/kf07 --out results/video/pair.json

Candidates come from the rendered deepest keyframe (adjacent samples, both solved and settled, settle times far
apart, near the centre of the window).  Each candidate is re-run on this Mac (MLX); it is kept only if both replays
reproduce the stored settle time and outcome and each trace is monotone (wrong until loop T, the solution from
loop T + 1), so the cards in the video show exactly what the stored pixels say.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import pairpick as PP  # noqa: E402
from render import puzzles as P  # noqa: E402
from render import runner as RU  # noqa: E402
from render import slices as SL  # noqa: E402
from render import sudoku as S  # noqa: E402
from render import tilerun as TR  # noqa: E402
from render.engine import Engine  # noqa: E402


def trace_strings(trace: np.ndarray) -> list[str]:
    return ["".join("?" if (v < 0 or v > 9) else ("." if v == 0 else str(int(v))) for v in g) for g in trace]


def monotone(trace: np.ndarray, final: np.ndarray, settle: int) -> bool:
    """Wrong (not the final grid) for loops 1..settle, the final grid from loop settle + 1 on."""
    eq = (trace == final[None]).all(-1)
    return bool((~eq[:settle]).all() and eq[settle:].all())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="merged keyframe folder (field.npz, spec.json)")
    ap.add_argument("--out", default="results/video/pair.json")
    ap.add_argument("--min-gap", type=int, default=6)
    ap.add_argument("--max-gap", type=int, default=14, help="skip absurdly slow partners (they would need a long replay)")
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--loop-seconds", type=float, default=0.28)
    a = ap.parse_args()
    run = ROOT / a.run
    spec = json.loads((run / "spec.json").read_text())
    z = np.load(run / "field.npz")
    s, f = z["settle_nb"], z["flags"]
    ny, nx = s.shape
    cands = [c for c in PP.candidates(s, f, min_gap=a.min_gap, margin=a.margin, k=a.tries * 4) if c[2] <= a.max_gap]
    print(f"{len(cands)} candidates (gap {a.min_gap}..{a.max_gap}) in {run.name}", flush=True)
    sol = S.pack([P.ASSET.answer])[0].astype(np.int8)
    plane = SL.get_plane(0, ROOT / "data/plane_seed0.npz")
    eng = Engine(P.ASSET.question, backend="mlx")
    prec = TR._latent_precision(spec)
    pitch = 2 * spec["half_width"] / (nx - 1)
    kept = None
    for n_try, (r, c, gap) in enumerate(cands[: a.tries]):
        coords = TR.pixel_coords(spec, TR.Tile(0, 0, 0, r, c, 1, 2))
        res = RU.run_pixels(eng, coords, plane, P.ASSET.question, sol, batch=2, precision=prec, keep_traces=True)
        ok = True
        for j, cj in enumerate((c, c + 1)):
            T = int(res["settle_nb"][j])
            same = (T == int(s[r, cj]) and bool(res["solved"][j]) == bool(f[r, cj] & 1) and not bool(res["censored"][j])
                    and monotone(res["traces"][j], sol, T))
            ok &= same
        print(f"try {n_try}: row {r} col {c} gap {gap}: stored {int(s[r, c])}/{int(s[r, c + 1])} replay "
              f"{int(res['settle_nb'][0])}/{int(res['settle_nb'][1])} -> {'KEEP' if ok else 'skip'}", flush=True)
        if ok:
            kept = (r, c, gap, coords, res)
            break
    if kept is None:
        sys.exit("no candidate replayed identically; render more keyframe pixels or loosen --min-gap")
    r, c, gap, coords, res = kept
    ta, tb = int(res["settle_nb"][0]), int(res["settle_nb"][1])
    spec_out = dict(
        puzzle=P.ASSET.question, solution=P.ASSET.answer, loop_seconds=a.loop_seconds, loops_shown=int(min(24, max(ta, tb) + 3)),
        pitch=pitch, layer=run.name, render_row=int(r), display_row=int(ny - 1 - r), col=int(c), gap=int(gap),
        a=dict(x=float(coords[0, 0]), y=float(coords[0, 1]), settle=ta, trace=trace_strings(res["traces"][0]), label="A"),
        b=dict(x=float(coords[1, 0]), y=float(coords[1, 1]), settle=tb, trace=trace_strings(res["traces"][1]), label="B"),
    )
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec_out))
    print(f"wrote {out}: A settles in {ta} loops, B in {tb}; pixel pitch {pitch:.3e} plane units")


if __name__ == "__main__":
    main()
