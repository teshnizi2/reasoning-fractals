"""Explorer data pack: what the static site loads.

* ``layerN.bin.gz`` -- 4 bytes per pixel, display orientation (row 0 = top = largest y):
  ``[settle, flags, gid_lo, gid_hi]``.  flags: bit0 solved, bit1 valid, bit2 censored.
* ``attractors.json`` -- one dictionary shared by all layers: id -> 81-char grid, count, validity, wrong cells.
* ``manifest.json`` -- puzzle, plane geometry of each layer (centre, half-width, pixel pitch), image names.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import outcomes as O

MAX_ID = 65535  # gid is stored in 16 bits; everything at or above collapses into this "other" id


@dataclass(frozen=True)
class LayerMeta:
    id: int
    center: tuple[float, float]
    half_width: float
    nx: int
    ny: int
    kind: str = "zoom"


def clamp_gid(gid: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(gid), MAX_ID).astype(np.uint16)


def encode_layer(settle: np.ndarray, flags: np.ndarray, gid: np.ndarray) -> bytes:
    ny, nx = settle.shape
    g = clamp_gid(gid)[::-1]
    arr = np.empty((ny, nx, 4), dtype=np.uint8)
    arr[..., 0] = settle[::-1]
    arr[..., 1] = flags[::-1]
    arr[..., 2] = g & 0xFF
    arr[..., 3] = g >> 8
    return arr.tobytes()


def decode_layer(raw: bytes, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(ny, nx, 4)
    return arr[..., 0], arr[..., 1], (arr[..., 2].astype(np.uint16) | (arr[..., 3].astype(np.uint16) << 8))


def unify(solution: np.ndarray, layers: list[dict]) -> tuple[O.Dictionary, list[np.ndarray]]:
    """Merge every layer's own dictionary into one; return it and each layer's gid remapped to shared ids."""
    d = O.build_dictionary([l["dict_grids"] for l in layers], [l["dict_counts"] for l in layers], solution)
    return d, [m[np.asarray(l["gid"])] for m, l in zip(d.id_maps, layers)]


def manifest(layers: list[LayerMeta], puzzle: dict, meta: dict, images: dict | None = None) -> dict:
    out = []
    for lm in layers:
        pitch = 2 * lm.half_width / (lm.nx - 1)
        entry = dict(
            id=lm.id, kind=lm.kind, center=list(lm.center), half_width=lm.half_width,
            half_height=lm.half_width * (lm.ny - 1) / (lm.nx - 1), pitch=pitch, nx=lm.nx, ny=lm.ny,
            zoom=1.0 / lm.half_width, data=f"layer{lm.id}.bin.gz",
        )
        if images and lm.id in images:
            entry["images"] = images[lm.id]
        out.append(entry)
    return dict(puzzle=puzzle, meta=meta, layers=out)


def write_pack(out_dir, layers: list[LayerMeta], fields: list[dict], dictionary: O.Dictionary, puzzle: dict,
               meta: dict, images: dict | None = None) -> None:
    assert len(layers) == len(fields), "one field per layer"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for lm, f in zip(layers, fields):
        raw = encode_layer(f["settle_nb"], f["flags"], f["gid"])
        (out / f"layer{lm.id}.bin.gz").write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    (out / "attractors.json").write_text(json.dumps(O.dictionary_records(dictionary, puzzle["question"])))
    (out / "manifest.json").write_text(json.dumps(manifest(layers, puzzle, meta, images)))


def write_images(out_dir, layer_id: int, field: dict, colourways: list[str], fmt: str = "webp",
                 quality: int = 90, supersample: int = 1) -> dict:  # quality is kept for API compatibility; webp is lossless
    """Colour images of a layer in display orientation (y up), one file per colourway."""
    from PIL import Image

    from . import colour as C

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {}
    for name in colourways:
        img = C.colourize(field["settle_nb"], field["flags"], C.COLOURWAYS[name])[::-1]
        if supersample > 1:
            img = C.downsample(np.ascontiguousarray(img), supersample)
        fn = f"layer{layer_id}_{name}.{fmt}"
        im = Image.fromarray(np.ascontiguousarray(img))
        if fmt == "webp":
            im.save(out / fn, lossless=True, quality=100, method=6)  # exact colours: single pixels are the data
        else:
            im.save(out / fn)
        names[name] = fn
    return names


def colourway_manifest(names: list[str]) -> dict:
    from . import colour as C

    out = {}
    for n in names:
        cw = C.COLOURWAYS[n]
        ts = range(cw.t_lo, cw.t_hi + 1)
        out[n] = dict(t_lo=cw.t_lo, t_hi=cw.t_hi, solved=[cw.solved_lut[t].tolist() for t in ts],
                      wrong=[cw.wrong_lut[t].tolist() for t in ts], censored=list(cw.censored))
    return out
