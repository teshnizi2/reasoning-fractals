"""The whole finishing pipeline on a tiny fake render (stub engine): merge -> pack -> replay plan -> replay -> featured + pair -> video stills.

Catches glue errors (file names, argument names, coordinate conventions) before the real renders exist.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from render import puzzles as P
from render import replay as RP
from render import sudoku as S
from render import tilerun as TR

ROOT = Path(__file__).resolve().parents[1]
SOL = S.pack([P.ASSET.answer])[0].astype(np.int8)


class Stub:
    """Settle time from the latent's first two entries (a, b): a hash-like staircase (neighbours differ by several loops), never unsolved."""

    def rollout(self, z_h, z_l=None):
        out = np.zeros((len(z_h), 24, 81), dtype=np.int8)
        for k, z in enumerate(z_h):
            settle = 2 + ((7 * math.floor(float(z[0, 0]) * 4000) + 13 * math.floor(float(z[0, 1]) * 4000)) % 12)
            out[k, :settle] = (SOL % 9) + 1  # wrong until it settles
            out[k, settle:] = SOL
        return out


def plane():
    u = np.zeros((97, 512), dtype=np.float32)
    v = np.zeros((97, 512), dtype=np.float32)
    u[0, 0], v[0, 1] = 1.0, 1.0
    return u, v


def render(tmp, name, **kw):
    spec = dict(name=name, puzzle=P.ASSET.question, steps=24, plane_seed=0, precision="fp32", tile=48, batch=64)
    spec.update(kw)
    d = tmp / name
    TR.run_tiles(Stub(), spec, d, plane=plane(), solution=SOL)
    f = TR.merge(spec, d, SOL)
    TR.write_field(f, d)
    return d


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return r.stdout


def test_finishing_pipeline_end_to_end_on_a_fake_render(tmp_path):
    hero = render(tmp_path, "hero", mode="notebook", n=96)
    kf = [render(tmp_path, f"kf0{k}", mode="window", center=[0.5, 0.4], half_width=0.1 / k, nx=96, ny=120, n=96, tile=48,
                 latent_precision="float64") for k in (3, 4)]
    pack = tmp_path / "pack"
    run("scripts/build_pack.py", "--hero", hero, "--keyframes", *kf, "--out", pack, "--format", "png")
    man = json.loads((pack / "manifest.json").read_text())
    assert [l["kind"] for l in man["layers"]] == ["base", "zoom", "zoom"] and (pack / "layer2_series.png").exists()

    rdir = tmp_path / "replay"
    out = run("scripts/plan_replay.py", "--hero", hero, "--keyframes", *kf, "--out-dir", rdir, "--per-kind", "6", "--finale-per-layer", "20")
    assert "items" in out
    plan = json.loads((rdir / "plan.json").read_text())
    spec_in = json.loads((rdir / "replay_in.json").read_text())
    assert len(plan) == len(spec_in["items"]) > 10 and {p["layer"] for p in plan} >= {"hero", "kf03"}

    # what alice/replay.py does on the cluster, with the stub engine
    res = RP.run_items(Stub().rollout, plane(), spec_in["puzzle"], SOL, spec_in["items"], nudged=spec_in["nudged"], eps=spec_in["eps"],
                       seed=spec_in["seed"], batch=64)
    assert sum(r["equal"] for r in res) == len(res)  # the stub is deterministic: every plain replay reproduces the stored value
    (tmp_path / "replay_out.json").write_text(json.dumps(dict(meta=dict(engine="stub", nudged=spec_in["nudged"], eps=spec_in["eps"]), items=res)))

    pair = tmp_path / "pair.json"
    run("scripts/apply_replay.py", "--plan", rdir / "plan.json", "--replay", tmp_path / "replay_out.json", "--pack", pack, "--keyframes", *kf,
        "--pair-out", pair, "--summary", tmp_path / "summary.json", "--min-robust-gap", "3")
    man = json.loads((pack / "manifest.json").read_text())
    assert man["meta"]["featured"] and all(len(f["trace"]) == 24 for f in man["meta"]["featured"])
    pj = json.loads(pair.read_text())
    assert pj["layer"].startswith("kf") and pj["a"]["x"] < pj["b"]["x"] and len(pj["a"]["trace"]) == 24 and pj["gap"] >= 3

    stills = tmp_path / "stills"
    run("scripts/make_video.py", "--pack", pack, "--way", "series", "--pair", pair, "--captions", "scripts/video_captions.json",
        "--stills", "0,6,12.5,16,19", "--stills-dir", stills, "--out", tmp_path / "v.mp4")
    for sec in ("00.00", "06.00", "12.50", "16.00", "19.00"):
        assert Image.open(stills / f"frame_{sec}.png").size == (1080, 1350) and Image.open(stills / f"square_{sec}.png").size == (1080, 1080)
