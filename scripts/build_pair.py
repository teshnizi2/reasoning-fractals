#!/usr/bin/env python
"""Pick the pair of neighbouring starts for the video finale and replay both locally, loop by loop.

    .venv/bin/python scripts/build_pair.py --run data/cluster/kf07 --out results/video/pair.json

Candidates come from a rendered field (adjacent samples, both solved and settled, settle times far apart, away from
the border).  Each candidate is re-run on this Mac (MLX, a different float32 stack from the A100 render) once as is
and ``--runs`` times with a relative nudge of ``--eps`` on the latents (a few dozen float32 ulps).  It is kept only if
every run reproduces the stored settle time of both pixels and the traces are monotone (wrong until loop T, the
solution from loop T + 1): then the pair is a property of the map rather than of one float32 rounding, and the cards
in the video show exactly what the stored pixels say.  Everything tried goes to ``<out>.search.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import outcomes as O  # noqa: E402
from render import pairpick as PP  # noqa: E402
from render import puzzles as P  # noqa: E402
from render import slices as SL  # noqa: E402
from render import sudoku as S  # noqa: E402
from render import tilerun as TR  # noqa: E402
from render.engine import Engine  # noqa: E402


def trace_strings(trace: np.ndarray) -> list[str]:
    return ["".join("?" if (v < 0 or v > 9) else ("." if v == 0 else str(int(v))) for v in g) for g in trace]


def monotone(trace: np.ndarray, final: np.ndarray, settle: int) -> bool:
    """Not the final grid for loops 1..settle, the final grid from loop settle + 1 on."""
    eq = (trace == final[None]).all(-1)
    return bool((~eq[:settle]).all() and eq[settle:].all())


def load_field(run: Path, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """(settle, flags) of a merged run, or of the tiles that exist so far (missing tiles: flags 0, so never candidates)."""
    if (run / "field.npz").exists():
        z = np.load(run / "field.npz")
        return z["settle_nb"], z["flags"]
    ny, nx = TR.spec_dims(spec)
    s = np.zeros((ny, nx), np.uint8)
    f = np.zeros((ny, nx), np.uint8)
    for t in TR.plan_tiles(ny, spec["tile"], nx):
        p = run / "tiles" / TR._tile_name(t)
        if p.exists():
            z = np.load(p)
            sl = (slice(t.r0, t.r0 + t.h), slice(t.c0, t.c0 + t.w))
            s[sl], f[sl] = z["settle_nb"], z["flags"]
    return s, f


def spec_pitch(spec: dict) -> float:
    """Plane units between neighbouring samples."""
    if spec["mode"] == "notebook":
        return 2.0 / (spec["n"] - 1)
    return 2.0 * spec["half_width"] / (TR.spec_dims(spec)[1] - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="keyframe/hero folder (field.npz or tiles, and spec.json)")
    ap.add_argument("--out", default="results/video/pair.json")
    ap.add_argument("--min-gap", type=int, default=6)
    ap.add_argument("--max-gap", type=int, default=15, help="skip absurdly slow partners")
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--clean", action="store_true", help="only clean steps (borders between flat domains), not speckle")
    ap.add_argument("--runs", type=int, default=3, help="nudged replays per candidate (plus the plain one)")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--loop-seconds", type=float, default=0.28)
    ap.add_argument("--stop-after", type=int, default=1, help="stop after this many robust pairs (the first is used)")
    a = ap.parse_args()
    run = ROOT / a.run
    spec = json.loads((run / "spec.json").read_text())
    s, f = load_field(run, spec)
    ny, nx = s.shape
    cands = PP.candidates(s, f, min_gap=a.min_gap, margin=a.margin, k=a.tries * 6, max_gap=a.max_gap, clean=a.clean)
    print(f"{len(cands)} candidates (gap {a.min_gap}..{a.max_gap}) in {run.name}", flush=True)
    sol = S.pack([P.ASSET.answer])[0].astype(np.int8)
    u, v = SL.get_plane(0, ROOT / "data/plane_seed0.npz")
    eng = Engine(P.ASSET.question, backend="mlx")
    prec = TR._latent_precision(spec)
    pitch = spec_pitch(spec)
    rng = np.random.default_rng(2026)
    tried, kept = [], []
    for n_try, (r, c, gap) in enumerate(cands[: a.tries]):
        coords = TR.pixel_coords(spec, TR.Tile(0, 0, 0, r, c, 1, 2))
        z = SL.latents(coords, u, v, precision=prec)
        Z = np.concatenate([z] + [SL.perturb(z, a.eps, rng) for _ in range(a.runs)])
        tr = eng.rollout(Z)
        res = O.analyze(tr, P.ASSET.question, sol)
        stored = [int(s[r, c]), int(s[r, c + 1])]
        got = res.settle_nb.reshape(a.runs + 1, 2).astype(int).tolist()
        ok = all(
            got[k][j] == stored[j] and bool(res.solved[2 * k + j]) and not bool(res.censored[2 * k + j])
            for k in range(a.runs + 1) for j in range(2)
        ) and all(monotone(tr[j], sol, stored[j]) for j in range(2))
        tried.append(dict(row=r, col=c, gap=gap, stored=stored, replay=got, kept=ok))
        print(f"try {n_try}: row {r} col {c} gap {gap}: stored {stored} replays {got} -> {'KEEP' if ok else 'skip'}", flush=True)
        if ok:
            kept.append((r, c, gap, coords, tr[:2], stored))
            if len(kept) >= a.stop_after:
                break
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    search = dict(run=str(a.run), min_gap=a.min_gap, runs=a.runs, eps=a.eps, engine="MLX float32, Apple M4",
                  tried=len(tried), kept=len(kept), rows=tried)
    out.with_name(out.stem + ".search.json").write_text(json.dumps(search, indent=1))
    if not kept:
        sys.exit(f"no candidate survived {a.runs + 1} replays; {len(tried)} tried (see {out.stem}.search.json)")
    r, c, gap, coords, tr, (ta, tb) = kept[0]
    spec_out = dict(
        puzzle=P.ASSET.question, solution=P.ASSET.answer, loop_seconds=a.loop_seconds, loops_shown=int(min(24, max(ta, tb) + 3)),
        pitch=pitch, layer=run.name, render_row=int(r), display_row=int(ny - 1 - r), col=int(c), gap=int(gap),
        verified=dict(engine="MLX float32 (Apple M4)", plain_runs=1, nudged_runs=a.runs, eps=a.eps, tried=len(tried)),
        a=dict(x=float(coords[0, 0]), y=float(coords[0, 1]), settle=ta, trace=trace_strings(tr[0]), label="A"),
        b=dict(x=float(coords[1, 0]), y=float(coords[1, 1]), settle=tb, trace=trace_strings(tr[1]), label="B"),
    )
    out.write_text(json.dumps(spec_out))
    print(f"wrote {out}: A settles in {ta} loops, B in {tb}; pixel pitch {pitch:.3e} plane units; "
          f"{len(kept)} robust of {len(tried)} tried")


if __name__ == "__main__":
    main()
