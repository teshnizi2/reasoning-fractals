#!/usr/bin/env python
"""Smoke + benchmark on one cluster GPU (run inside alice/smoke.slurm, from ~/reasoning-fractals).

Stages (results are written to <out>/bench.json after every stage, so a time-out keeps what finished):
  1. environment + GPU facts
  2. per-precision throughput (fp32 / tf32 / bf16) at several batch sizes
  3. agreement on a few thousand pixels of the upstream asset's grid (pixel-by-pixel against the asset's own
     levels when data/asset_samples.npz exists) and fp32-vs-tf32/bf16 agreement
  4. per-loop latent residuals for a pilot (to design a safe early exit)
  5. torch.compile attempt
  6. end-to-end tiled render + merge on a 64 x 64 window
  7. outcome-vs-scale scan: 48 x 48 windows of half-width 1, 4, 16, 64, 256 (does the model ever end on a wrong
     grid, and how many different ones?)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

import torch  # noqa: E402

from render import repro as R  # noqa: E402
from render import runner as RU  # noqa: E402
from render import slices as SL  # noqa: E402
from render import sudoku as S  # noqa: E402
from render import tilerun as TR  # noqa: E402
from render.engine import Engine  # noqa: E402

PUZZLE = "2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3..."  # the asset's puzzle
OUT: Path
RES: dict = {}


def save() -> None:
    (OUT / "bench.json").write_text(json.dumps(RES, indent=1, default=float))


def sync() -> None:
    torch.cuda.synchronize()


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/smoke")
    ap.add_argument("--batches", default="256,512,1024")
    ap.add_argument("--skip-compile", action="store_true")
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)

    props = torch.cuda.get_device_properties(0)
    RES["env"] = dict(host=os.uname().nodename, gpu=props.name, total_gb=props.total_memory / 2**30,
                      torch=torch.__version__, cuda=torch.version.cuda,
                      driver=subprocess.run("nvidia-smi --query-gpu=driver_version --format=csv,noheader", shell=True,
                                            capture_output=True, text=True).stdout.strip())
    print(RES["env"], flush=True)
    save()

    plane = SL.get_plane(0, "data/plane_seed0.npz")
    sol = S.pack([S.solve(PUZZLE)])[0].astype(np.int8)
    ax = SL.axis(1000)
    sp = Path("data/asset_samples.npz")
    if sp.exists():
        z = np.load(sp)
        xi, yi, levels, nbr = z["xi"], z["yi"], z["levels"], z["nbr_diff"]
    else:
        rng = np.random.default_rng(0)
        xi, yi, levels, nbr = rng.integers(0, 1000, 4096), rng.integers(0, 1000, 4096), None, None
    coords = np.stack([ax[xi], ax[yi]], axis=-1)
    RES["n_sample_pixels"] = int(len(coords))

    # ---- stage 2 + 3: throughput and agreement, per precision
    batches = [int(b) for b in a.batches.split(",")]
    base = {}
    for prec in ("fp32", "tf32", "bf16"):
        eng = Engine(PUZZLE, backend="torchfn", device="cuda", precision=prec)
        row = {}
        for b in batches:
            z = SL.latents(coords[:b], *plane)
            eng.rollout(z)
            sync()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            eng.rollout(z)
            sync()
            dt = time.time() - t0
            row[f"B{b}"] = dict(rollouts_per_s=b / dt, seconds=dt, peak_gb=torch.cuda.max_memory_allocated() / 2**30)
            print(prec, f"B={b}", f"{b / dt:.1f} rollouts/s", flush=True)
        RES.setdefault("throughput", {})[prec] = row
        save()
        r = RU.run_pixels(eng, coords, plane, PUZZLE, sol, batch=max(batches))
        agree = dict(seconds=r["seconds"], frac_solved=float(r["solved"].mean()),
                     frac_censored=float(r["censored"].mean()), settle_hist=np.bincount(r["settle_nb"], minlength=25).tolist())
        if levels is not None:
            shift, frac = R.best_shift_agreement(levels, r["settle_nb"], clip_top=10)
            clipped = np.clip(r["settle_nb"].astype(int) - shift, 0, 10)
            interior = nbr == 0
            agree.update(
                asset_best_shift=shift, asset_agreement=frac, asset_spearman=R.spearman(levels, r["settle_nb"]),
                asset_agreement_interior=float((levels == clipped)[interior].mean()),
                asset_agreement_boundary=float((levels == clipped)[~interior].mean()),
                frac_interior=float(interior.mean()),
                agreement_by_level=[float((clipped[levels == k] == k).mean()) if (levels == k).any() else None
                                    for k in range(11)],
            )
        if prec == "fp32":
            base = r
        else:
            agree.update(
                vs_fp32_settle_equal=float((r["settle_nb"] == base["settle_nb"]).mean()),
                vs_fp32_final_equal=float((r["final"] == base["final"]).all(-1).mean()),
            )
        RES.setdefault("agreement", {})[prec] = agree
        print(prec, agree, flush=True)
        save()
        np.savez_compressed(OUT / f"sample_{prec}.npz", settle_nb=r["settle_nb"], settle_last=r["settle_last"],
                            solved=r["solved"], final=r["final"], xi=xi, yi=yi)

    # ---- stage 4: latent residual pilot (fp32)
    eng = Engine(PUZZLE, backend="torchfn", device="cuda", precision="fp32")
    npil = min(1024, len(coords))
    tr, resid = eng.rollout_ex(SL.latents(coords[:npil], *plane))
    np.savez_compressed(OUT / "resid_pilot_fp32.npz", traces=tr, resid=resid, xi=xi[:npil], yi=yi[:npil])
    RES["resid_pilot"] = dict(n=int(npil), median_by_loop=np.median(resid, axis=0).tolist(),
                              p99_by_loop=np.percentile(resid, 99, axis=0).tolist(),
                              max_by_loop=resid.max(axis=0).tolist())
    save()

    # ---- stage 5: torch.compile
    if not a.skip_compile:
        try:
            eng_c = Engine(PUZZLE, backend="torchfn", device="cuda", precision="tf32", compile=True)
            b = batches[-1]
            z = SL.latents(coords[:b], *plane)
            t0 = time.time()
            eng_c.rollout(z)
            sync()
            compile_s = time.time() - t0
            t0 = time.time()
            eng_c.rollout(z)
            sync()
            dt = time.time() - t0
            RES["compile_tf32"] = dict(first_call_s=compile_s, rollouts_per_s=b / dt)
        except Exception as e:  # noqa: BLE001
            RES["compile_tf32"] = dict(error=repr(e)[:500])
        print("compile:", RES["compile_tf32"], flush=True)
        save()

    # ---- stage 6: end-to-end tiled render on a 64 x 64 window
    spec = dict(name="smoke", puzzle=PUZZLE, mode="window", center=[0.0, 0.0], half_width=1.0, n=64, tile=32,
                batch=1024, precision="tf32", latent_precision="float64", steps=24, plane_seed=0,
                plane_file="data/plane_seed0.npz")
    eng = Engine(PUZZLE, backend="torchfn", device="cuda", precision="tf32")
    t0 = time.time()
    stats = TR.run_tiles(eng, spec, OUT / "pipeline", plane, sol)
    f = TR.merge(spec, OUT / "pipeline", sol)
    TR.write_field(f, OUT / "pipeline")
    RES["pipeline"] = dict(stats=stats, seconds=time.time() - t0, dictionary_size=len(f["dictionary"]),
                           frac_solved=float(((f["flags"] & 1) > 0).mean()),
                           settle_hist=np.bincount(f["settle_nb"].ravel(), minlength=25).tolist())
    save()

    # ---- stage 7: outcome-vs-scale scan (tf32, 48 x 48 windows around the origin of the plane)
    scan = {}
    for hw in (1.0, 4.0, 16.0, 64.0, 256.0):
        w = SL.Window(center=(0.0, 0.0), half_width=hw, n=48)
        r = RU.run_pixels(eng, w.coords(), plane, PUZZLE, sol, batch=2304, precision="float64")
        np.savez_compressed(OUT / f"scan_hw{hw:g}.npz", settle_nb=r["settle_nb"].reshape(48, 48),
                            settle_last=r["settle_last"].reshape(48, 48), n_wrong=r["n_wrong"].reshape(48, 48),
                            solved=r["solved"].reshape(48, 48), final=r["final"])
        scan[f"hw{hw:g}"] = dict(
            seconds=r["seconds"], frac_solved=float(r["solved"].mean()), frac_valid=float(r["valid"].mean()),
            frac_censored=float(r["censored"].mean()), n_distinct_final=int(len(np.unique(r["final"], axis=0))),
            median_n_wrong_of_unsolved=float(np.median(r["n_wrong"][~r["solved"]])) if (~r["solved"]).any() else None,
            settle_hist=np.bincount(r["settle_nb"], minlength=25).tolist())
        print("scan", hw, scan[f"hw{hw:g}"], flush=True)
        RES["scale_scan"] = scan
        save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
