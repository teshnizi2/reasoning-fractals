"""Assemble the explorer pack (site/data) from rendered layer directories (tiles + spec.json each)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import pack as PK
from . import sudoku as S
from . import tilerun as TR


def build(layer_dirs: list, out_dir, puzzle: dict, ways: list[str], meta_extra: dict | None = None,
          image_format: str = "webp", supersample: int = 1) -> dict:
    """Merge every layer's tiles, unify the answer dictionary, write images, binary layers and the manifest.

    ``layer_dirs[0]`` is the base layer (the whole plane); the rest are zoom layers.  All must be window-mode
    renders of the same puzzle/plane.  Returns the manifest dict.
    """
    sol = S.pack([puzzle["solution"]])[0].astype(np.int8)
    fields, metas = [], []
    for i, d in enumerate(layer_dirs):
        d = Path(d)
        spec = json.loads((d / "spec.json").read_text())
        if spec["mode"] == "notebook":  # loopscape's mesh(n): the window [-1, 1]^2
            center, half_width = (0.0, 0.0), 1.0
        elif spec["mode"] == "window":
            center, half_width = tuple(spec["center"]), spec["half_width"]
        else:
            raise ValueError(f"{d}: only window/notebook-mode renders go into an explorer pack")
        ny, nx = TR.spec_dims(spec)
        fields.append(TR.merge(spec, d, sol))
        metas.append(PK.LayerMeta(id=i, center=center, half_width=half_width, nx=nx, ny=ny,
                                  kind="base" if i == 0 else "zoom"))
    dictionary, remapped = PK.unify(sol, fields)
    for f, g in zip(fields, remapped):
        f["gid"] = g
    images = {m.id: PK.write_images(out_dir, m.id, f, ways, fmt=image_format, supersample=supersample)
              for m, f in zip(metas, fields)}
    meta = dict(loops=24, colourways=PK.colourway_manifest(ways), **(meta_extra or {}))
    PK.write_pack(out_dir, metas, fields, dictionary, puzzle, meta, images)
    return json.loads((Path(out_dir) / "manifest.json").read_text())
