"""Assemble an explorer pack from several rendered layer directories (stub engine, no model)."""
import gzip
import json

import numpy as np

from render import buildpack as BP
from render import pack as PK
from render import sudoku as S
from render import tilerun as TR

from test_tilerun import PUZZLE, SOL, SOLUTION, StubEngine, make_spec, stub_plane


def make_layer(tmp_path, name, center, half_width, n=8):
    spec = make_spec(name=name, center=list(center), half_width=half_width, n=n, tile=4)
    d = tmp_path / name
    TR.run_tiles(StubEngine(), spec, d, plane=stub_plane(), solution=SOL)
    return d


def test_build_assembles_layers_images_dictionary_and_manifest(tmp_path):
    base = make_layer(tmp_path, "base", (3.5, 3.5), 3.5)
    zoom = make_layer(tmp_path, "z1", (3.5, 3.5), 1.75)
    out = tmp_path / "pack"
    BP.build([base, zoom], out, puzzle=dict(question=PUZZLE, solution=SOLUTION, rating=0), ways=["loopscape", "series"],
             meta_extra=dict(subtitle="hello"), image_format="png")
    man = json.loads((out / "manifest.json").read_text())
    assert [l["id"] for l in man["layers"]] == [0, 1] and man["layers"][0]["kind"] == "base"
    assert man["layers"][1]["kind"] == "zoom" and abs(man["layers"][1]["zoom"] - 1 / 1.75) < 1e-9
    assert man["meta"]["subtitle"] == "hello" and set(man["meta"]["colourways"]) == {"loopscape", "series"}
    assert (out / "layer0_loopscape.png").exists() and (out / "layer1_series.png").exists()
    att = json.loads((out / "attractors.json").read_text())
    assert att[0]["grid"] == SOLUTION and sum(a["count"] for a in att) == 128  # two 8x8 layers
    s, f, g = PK.decode_layer(gzip.decompress((out / "layer0.bin.gz").read_bytes()), 8, 8)
    assert s.shape == (8, 8) and int(g.max()) < len(att)
