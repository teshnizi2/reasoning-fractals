"""Tiled, resumable rendering of one slice of the initial-latent plane (runs on the cluster).

A *spec* (JSON) names the puzzle, plane seed and either a zoom window (``mode: window``: centre,
half_width, n x n pixels, float64 coordinates) or the upstream asset's own grid (``mode: asset``: every
``stride``-th pixel of the 1000-pixel notebook axis, float32 notebook arithmetic).  The image is cut into
square tiles; each finished tile is one small npz written atomically, so a job killed at the 4 h limit
loses at most the tile it was on and a re-submission continues where it stopped.

    python -m render.tilerun run   --spec spec.json --out out/NAME --worker $SLURM_ARRAY_TASK_ID --workers 8
    python -m render.tilerun merge --spec spec.json --out out/NAME
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import outcomes as O
from . import runner as RU
from . import slices as SL
from . import sudoku as S

ARRAYS = ("settle_nb", "settle_last", "n_wrong", "n_changes", "n_distinct", "flags", "gid")


@dataclass(frozen=True)
class Tile:
    index: int
    ti: int
    tj: int
    r0: int
    c0: int
    h: int
    w: int


def spec_dims(spec: dict) -> tuple[int, int]:
    """(rows, cols) of the image: ``ny``/``nx`` if given, else ``n`` x ``n``."""
    return (spec["ny"], spec["nx"]) if "nx" in spec else (spec["n"], spec["n"])


def plan_tiles(n: int, tile: int, nx: int | None = None) -> list[Tile]:
    """Row-major tiles covering an ``n`` (rows) x ``nx`` (cols, default ``n``) image exactly once."""
    nx = n if nx is None else nx
    tiles, k = [], 0
    for ti, r0 in enumerate(range(0, n, tile)):
        for tj, c0 in enumerate(range(0, nx, tile)):
            tiles.append(Tile(k, ti, tj, r0, c0, min(tile, n - r0), min(tile, nx - c0)))
            k += 1
    return tiles


def tiles_for_worker(tiles: list[Tile], worker: int, workers: int) -> list[Tile]:
    return [t for t in tiles if t.index % workers == worker]


def pixel_coords(spec: dict, t: Tile) -> np.ndarray:
    """(h*w, 2) plane coordinates (x, y) of a tile's pixels, row-major with x fastest."""
    rr, cc = np.meshgrid(np.arange(t.r0, t.r0 + t.h), np.arange(t.c0, t.c0 + t.w), indexing="ij")
    if spec["mode"] == "window":
        ny, nx = spec_dims(spec)
        ax_x = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
        ax_y = np.linspace(-1.0, 1.0, ny, dtype=np.float64)
        cx, cy = spec["center"]
        hw = spec["half_width"]
        hh = hw * (ny - 1) / (nx - 1) if nx > 1 else hw  # square pixels
        x, y = cx + hw * ax_x[cc], cy + hh * ax_y[rr]
        return np.stack([x.ravel(), y.ravel()], axis=-1)
    if spec["mode"] == "notebook":  # loopscape's mesh(n) verbatim: float32 axis, float32 latent arithmetic
        ax = SL.axis(spec["n"])
        return np.stack([ax[cc].ravel(), ax[rr].ravel()], axis=-1)
    if spec["mode"] == "asset":
        ax = SL.axis(spec["n_asset"])
        s = spec["stride"]
        return np.stack([ax[cc * s].ravel(), ax[rr * s].ravel()], axis=-1)
    raise ValueError(f"unknown mode {spec['mode']!r}")


def _tile_name(t: Tile) -> str:
    return f"t_{t.ti:03d}_{t.tj:03d}.npz"


def _latent_precision(spec: dict) -> str:
    return spec.get("latent_precision", "float64" if spec["mode"] == "window" else "float32")


def run_tiles(engine, spec: dict, out_dir, plane, solution: np.ndarray, worker: int = 0, workers: int = 1,
              max_minutes: float | None = None, max_tiles: int | None = None, clock=time.time, log=print) -> dict:
    out = Path(out_dir)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    spec_path = out / "spec.json"
    if not spec_path.exists():
        spec_path.write_text(json.dumps(spec, indent=1))
    ny, nx = spec_dims(spec)
    mine = tiles_for_worker(plan_tiles(ny, spec["tile"], nx), worker, workers)
    stats = dict(tiles_total=len(mine), tiles_done=0, tiles_skipped=0, seconds=0.0, rollouts=0,
                 stopped_for_budget=False)
    t0 = clock()
    for t in mine:
        path = out / "tiles" / _tile_name(t)
        if path.exists():
            stats["tiles_skipped"] += 1
            continue
        if max_minutes is not None and (clock() - t0) / 60.0 >= max_minutes:
            stats["stopped_for_budget"] = True
            break
        if max_tiles is not None and stats["tiles_done"] >= max_tiles:
            break
        res = RU.run_pixels(
            engine, pixel_coords(spec, t), plane, spec["puzzle"], solution,
            batch=spec.get("batch", 512), precision=_latent_precision(spec),
            early_m=(spec.get("early_exit") or {}).get("m"),
        )
        shp = (t.h, t.w)
        flags = (res["solved"].astype(np.uint8) | (res["valid"].astype(np.uint8) << 1)
                 | (res["censored"].astype(np.uint8) << 2) | (res["valid_sudoku"].astype(np.uint8) << 3))
        ug, inv, cnt = O.unique_grids(res["final"])
        tmp = path.with_name(path.stem + ".tmp.npz")
        try:
            np.savez(tmp, settle_nb=res["settle_nb"].reshape(shp), settle_last=res["settle_last"].reshape(shp),
                     n_wrong=res["n_wrong"].reshape(shp), n_changes=res["n_changes"].reshape(shp),
                     n_distinct=res["n_distinct"].reshape(shp), flags=flags.reshape(shp),
                     gidx=inv.astype(np.uint32).reshape(shp), ugrids=ug, ucounts=cnt,
                     seconds=np.float64(res["seconds"]), n_px=np.int64(t.h * t.w),
                     loops_mean=np.float64(res["retired_at"].mean() if "retired_at" in res else spec.get("steps", 24)))
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        stats["tiles_done"] += 1
        stats["seconds"] += res["seconds"]
        stats["rollouts"] += t.h * t.w
        log(f"[worker {worker}/{workers}] tile {t.index} ({t.ti},{t.tj}) {t.h}x{t.w} "
            f"{res['seconds']:.0f}s {t.h * t.w / res['seconds']:.1f} rollouts/s  "
            f"done {stats['tiles_done']} skipped {stats['tiles_skipped']} of {len(mine)}")
    return stats


def merge(spec: dict, out_dir, solution: np.ndarray) -> dict:
    out = Path(out_dir)
    ny, nx = spec_dims(spec)
    tiles = plan_tiles(ny, spec["tile"], nx)
    missing = [t.index for t in tiles if not (out / "tiles" / _tile_name(t)).exists()]
    if missing:
        raise FileNotFoundError(f"missing tiles: {missing[:20]}{'...' if len(missing) > 20 else ''} ({len(missing)})")
    dt = dict(settle_nb=np.uint8, settle_last=np.uint8, n_wrong=np.uint8, n_changes=np.uint8,
              n_distinct=np.uint8, flags=np.uint8)
    full = {k: np.zeros((ny, nx), dtype=v) for k, v in dt.items()}
    gidx_tiles, ug_tiles, uc_tiles, seconds, px, loops_sum = [], [], [], 0.0, 0, 0.0
    for t in tiles:
        z = np.load(out / "tiles" / _tile_name(t))
        sl = (slice(t.r0, t.r0 + t.h), slice(t.c0, t.c0 + t.w))
        for k in dt:
            full[k][sl] = z[k]
        gidx_tiles.append(z["gidx"])
        ug_tiles.append(z["ugrids"])
        uc_tiles.append(z["ucounts"])
        seconds += float(z["seconds"])
        px += int(z["n_px"])
        loops_sum += float(z["loops_mean"]) * int(z["n_px"]) if "loops_mean" in z.files else spec.get("steps", 24) * int(z["n_px"])
    d = O.build_dictionary(ug_tiles, uc_tiles, solution)
    gid = np.zeros((ny, nx), dtype=np.uint32)
    for t, g, idmap in zip(tiles, gidx_tiles, d.id_maps):
        gid[t.r0 : t.r0 + t.h, t.c0 : t.c0 + t.w] = idmap[g]
    full["gid"] = gid
    full["dictionary"] = O.dictionary_records(d, spec["puzzle"])
    full["dict_grids"], full["dict_counts"] = d.grids, d.counts
    full["meta"] = dict(name=spec.get("name"), ny=ny, nx=nx, pixels=px, gpu_seconds=seconds,
                        precision=spec.get("precision"), mean_loops_computed=loops_sum / max(px, 1), spec=spec)
    return full


def write_field(f: dict, out_dir) -> None:
    out = Path(out_dir)
    np.savez_compressed(out / "field.npz", **{k: f[k] for k in ARRAYS})
    (out / "dictionary.json").write_text(json.dumps(f["dictionary"]))
    (out / "meta.json").write_text(json.dumps(f["meta"], indent=1))


def _load_spec(path) -> dict:
    return json.loads(Path(path).read_text())


def apply_overrides(spec: dict, backend: str | None = None, device: str | None = None) -> dict:
    """Copy of ``spec`` for a different machine: ``--backend mlx`` (a Mac) drops the CUDA-only early exit and compile flags."""
    out = dict(spec)
    if backend:
        out["backend"] = backend
        if backend == "mlx":
            out.pop("early_exit", None)
            out["compile"] = False
    if device:
        out["device"] = device
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "merge"])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--backend", default=None, help="override the spec's engine, e.g. mlx on a Mac (early exit is cuda-only and is dropped)")
    ap.add_argument("--device", default=None, help="override the spec's torch device (cpu, mps, cuda)")
    a = ap.parse_args()
    spec = apply_overrides(_load_spec(a.spec), a.backend, a.device)
    solution = S.pack([S.solve(spec["puzzle"])])[0].astype(np.int8)
    if a.cmd == "merge":
        f = merge(spec, a.out, solution)
        write_field(f, a.out)
        print("merged", a.out, "dictionary size", len(f["dictionary"]), "gpu_seconds", f["meta"]["gpu_seconds"])
        return
    from .engine import Engine

    engine = Engine(spec["puzzle"], backend=spec.get("backend", "torchfn"), device=spec.get("device", "cuda"),
                    precision=spec.get("precision", "fp32"), compile=spec.get("compile", False))
    plane = SL.get_plane(seed=spec["plane_seed"], path=spec.get("plane_file"))
    stats = run_tiles(engine, spec, a.out, plane, solution, worker=a.worker,
                      workers=a.workers, max_minutes=a.max_minutes, max_tiles=a.max_tiles)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
