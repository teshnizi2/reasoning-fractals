"""Choose which pixels to replay on the cluster, and write the replay job's input.

``plan_layer`` returns plain dicts (kind, layer, row, col in render orientation, role/group for pairs, coordinates,
latent precision, stored settle/flags); ``replay_input`` strips them down to what ``alice/replay.py`` needs.
"""
from __future__ import annotations

import numpy as np

from . import featured as F
from . import pairpick as PP
from . import tilerun as TR


def _item(kind: str, layer: str, spec: dict, s: np.ndarray, f: np.ndarray, r: int, c: int, role: str = "single", group: int = 0) -> dict:
    x, y = (float(v) for v in TR.pixel_coords(spec, TR.Tile(0, 0, 0, int(r), int(c), 1, 1))[0])
    return dict(id=f"{layer}:{kind}:{group}:{role}", kind=kind, layer=layer, row=int(r), col=int(c), role=role, group=group, x=x, y=y,
                prec=TR._latent_precision(spec), stored=dict(settle=int(s[r, c]), flags=int(f[r, c])))


def plan_layer(layer: str, spec: dict, settle: np.ndarray, flags: np.ndarray, kinds: tuple[str, ...] = ("fast", "slow", "wrong", "pair"),
               per_kind: int = 10, pair_min_gap: int = 4, pair_max_gap: int = 15, margin: int = 4) -> list[dict]:
    out: list[dict] = []
    picks = {
        "fast": lambda: F.fast(settle, flags, k=per_kind),
        "slow": lambda: F.slow(settle, flags, k=per_kind),
        "wrong": lambda: F.wrong(settle, flags, k=per_kind),
    }
    for kind in kinds:
        if kind == "pair":
            for g, (r, c, _) in enumerate(PP.candidates(settle, flags, min_gap=pair_min_gap, margin=margin, k=per_kind, max_gap=pair_max_gap)):
                out.append(_item("pair", layer, spec, settle, flags, r, c, "a", g))
                out.append(_item("pair", layer, spec, settle, flags, r, c + 1, "b", g))
        else:
            for g, (r, c) in enumerate(picks[kind]()):
                out.append(_item(kind, layer, spec, settle, flags, r, c, "single", g))
    return out


def replay_input(puzzle: str, plan: list[dict], nudged: int = 3, eps: float = 1e-6, seed: int = 0) -> dict:
    return dict(puzzle=puzzle, nudged=nudged, eps=eps, seed=seed,
                items=[dict(id=p["id"], x=p["x"], y=p["y"], prec=p["prec"], stored=p["stored"]) for p in plan])
