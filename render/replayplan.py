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


ROUGHNESS_BINS = ((0, 0), (1, 1), (2, 4), (5, 99))


def plan_roughness(layer: str, spec: dict, settle: np.ndarray, flags: np.ndarray, per_bin: int = 100, seed: int = 5) -> list[dict]:
    """Random solved, settled pixels stratified by roughness (largest settle-time difference to the 4 neighbours):
    kinds ``rough0``, ``rough1``, ``rough2`` (2-4) and ``rough5`` (5+).  Used to measure how reproducible a pixel is."""
    s = np.asarray(settle).astype(np.int64)
    ok = ((np.asarray(flags) & 1) > 0) & ((np.asarray(flags) & 4) == 0)
    pad = np.pad(s, 1, mode="edge")
    ny, nx = s.shape
    rough = np.max([np.abs(s - pad[1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx]) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))], axis=0)
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for lo, hi in ROUGHNESS_BINS:
        cand = np.argwhere(ok & (rough >= lo) & (rough <= hi))
        if len(cand) == 0:
            continue
        pick = cand[rng.choice(len(cand), size=min(per_bin, len(cand)), replace=False)]
        for g, (r, c) in enumerate(pick):
            out.append(_item(f"rough{lo}", layer, spec, settle, flags, int(r), int(c), "single", g))
    return out
