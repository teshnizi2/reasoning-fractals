"""Explorer data pack: display-oriented binary layers, one dictionary shared by all layers, manifest."""
import gzip
import json

import numpy as np
import pytest

from render import outcomes as O
from render import pack as PK
from render import sudoku as S

SOLUTION = "236184795974325618851697432769538124583412967142769583327956841498271356615843279"
ASSET_Q = "2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3..."
SOL = S.pack([SOLUTION])[0].astype(np.int8)


def grid_with(*wrong):
    g = SOL.copy()
    for cell, val in wrong:
        g[cell] = val
    return g


def test_layer_bytes_round_trip_in_display_orientation():
    settle = np.arange(12, dtype=np.uint8).reshape(3, 4)
    flags = (settle % 2).astype(np.uint8)
    gid = (np.arange(12, dtype=np.uint32) * 1000).reshape(3, 4)
    raw = PK.encode_layer(settle, flags, gid)
    assert len(raw) == 3 * 4 * 4
    s2, f2, g2 = PK.decode_layer(raw, ny=3, nx=4)
    # stored with row 0 = top = highest y, i.e. the array flipped vertically
    np.testing.assert_array_equal(s2, settle[::-1])
    np.testing.assert_array_equal(f2, flags[::-1])
    np.testing.assert_array_equal(g2, gid[::-1].astype(np.uint16))


def test_ids_beyond_uint16_collapse_into_one_overflow_id():
    gid = np.array([[0, 1, 65534, 65535, 70000, 4_000_000]], dtype=np.uint32)
    out = PK.clamp_gid(gid)
    assert out.dtype == np.uint16 and out.tolist() == [[0, 1, 65534, 65535, 65535, 65535]]


def test_unify_layer_dictionaries_gives_shared_ids_and_remapped_pixels():
    a, b, c = grid_with((0, 9)), grid_with((1, 9)), grid_with((2, 9))
    # layer 0 saw: SOL x10, a x4 ; layer 1 saw: b x7, a x1, SOL x2
    l0 = dict(dict_grids=np.stack([SOL, a]), dict_counts=np.array([10, 4]), gid=np.array([[0, 1], [0, 0]], dtype=np.uint32))
    l1 = dict(dict_grids=np.stack([SOL, b, a]), dict_counts=np.array([2, 7, 1]), gid=np.array([[1, 2], [0, 1]], dtype=np.uint32))
    d, remapped = PK.unify(SOL, [l0, l1])
    assert (d.grids[0] == SOL).all() and d.counts[0] == 12
    assert (d.grids[1] == b).all() and (d.grids[2] == a).all()  # by total count: b 7, a 5
    assert remapped[0].tolist() == [[0, 2], [0, 0]] and remapped[1].tolist() == [[1, 2], [0, 1]]


def test_manifest_lists_layers_with_plane_geometry():
    layers = [
        PK.LayerMeta(id=0, center=(0.0, 0.0), half_width=1.0, nx=1000, ny=1000, kind="base"),
        PK.LayerMeta(id=1, center=(0.25, -0.5), half_width=1 / 16, nx=640, ny=800, kind="zoom"),
    ]
    m = PK.manifest(layers, puzzle=dict(question="q", solution="s", rating=47), meta={"loops": 24})
    assert m["layers"][1]["zoom"] == 16.0 and m["layers"][0]["zoom"] == 1.0
    assert abs(m["layers"][1]["half_height"] - (1 / 16) * 799 / 639) < 1e-12  # square pixels
    assert abs(m["layers"][1]["pitch"] - 2 * (1 / 16) / 639) < 1e-12
    assert m["puzzle"]["rating"] == 47 and m["meta"]["loops"] == 24
    json.dumps(m)


def test_write_pack_writes_gz_layers_and_attractor_json(tmp_path):
    a = grid_with((0, 9))
    settle = np.array([[2, 3], [4, 5]], dtype=np.uint8)
    flags = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    gid = np.array([[0, 0], [1, 0]], dtype=np.uint32)
    d = O.build_dictionary([np.stack([SOL, a])], [np.array([3, 1])], SOL)
    meta = PK.LayerMeta(id=0, center=(0, 0), half_width=1.0, nx=2, ny=2, kind="base")
    PK.write_pack(tmp_path, [meta], [dict(settle_nb=settle, flags=flags, gid=gid)], d,
                  puzzle=dict(question=ASSET_Q, solution=SOLUTION, rating=47), meta={})
    raw = gzip.decompress((tmp_path / "layer0.bin.gz").read_bytes())
    s2, f2, g2 = PK.decode_layer(raw, 2, 2)
    np.testing.assert_array_equal(s2, settle[::-1])
    att = json.loads((tmp_path / "attractors.json").read_text())
    assert att[0]["grid"] == SOLUTION and att[1]["n_wrong"] == 1 and att[1]["count"] == 1
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["layers"][0]["data"] == "layer0.bin.gz" and man["layers"][0]["nx"] == 2
    with pytest.raises(AssertionError):
        PK.write_pack(tmp_path, [meta], [], d, puzzle={}, meta={})  # layer/field count mismatch


def test_write_images_makes_display_oriented_colour_images_per_colourway(tmp_path):
    from PIL import Image

    from render import colour as C

    settle = np.array([[2, 12, 2], [2, 2, 2]], dtype=np.uint8)  # row 0 (lowest y) holds the slow pixel
    flags = np.ones((2, 3), dtype=np.uint8)
    names = PK.write_images(tmp_path, 7, dict(settle_nb=settle, flags=flags), ["loopscape", "series"], fmt="png")
    assert names == {"loopscape": "layer7_loopscape.png", "series": "layer7_series.png"}
    im = np.asarray(Image.open(tmp_path / names["loopscape"]).convert("RGB"))
    assert im.shape == (2, 3, 3)
    cw = C.COLOURWAYS["loopscape"]
    # flipped vertically: the slow pixel is now in the bottom row of the image
    assert tuple(im[1, 1]) == tuple(cw.solved_lut[12]) and tuple(im[0, 1]) == tuple(cw.solved_lut[2])


def test_colourway_manifest_exposes_luts_for_the_legend():
    m = PK.colourway_manifest(["loopscape"])
    cw = m["loopscape"]
    assert cw["t_lo"] == 2 and cw["t_hi"] == 12 and len(cw["solved"]) == 11 and len(cw["wrong"]) == 11
    assert len(cw["solved"][0]) == 3 and len(cw["censored"]) == 3
    json.dumps(m)
