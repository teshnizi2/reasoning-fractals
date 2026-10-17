# ALICE runs for reasoning-fractals

Everything here runs on one A100 per job (`gpu-short`, `--gres=gpu:a100:1`, 4 h limit), inside
`~/reasoning-fractals` on the cluster and nowhere else. Approved by Reza on 17 Oct for the high-res fractal renders
only (see `../../visual-series-RULES.md`). Every experiment gets one row in `../RUNS.md`.

## One-time setup (login node, has internet)

```bash
alice/sync.sh push                         # code + weights npz + frozen plane + frozen asset pixels -> ~/reasoning-fractals
ssh alice2 'bash ~/reasoning-fractals/alice/setup_env.sh'   # conda env with torch (CUDA wheels from PyPI)
```

The cluster never unpickles the upstream checkpoint: `data/eqr_sudoku_weights.npz` is exported locally by
`render/weights.py` after a pickle-global scan (see `tests/test_weights.py`).
The stock-path check (`stock_check.py`) additionally needs `vendor/loopscape` and `vendor/EqR` (copied with rsync,
see the commands in `RUNS.md` row for that job).

## Jobs

| script | what | GPU time |
|---|---|---|
| `smoke.slurm` + `bench_cuda.py` | throughput of fp32/tf32/bf16, agreement with the asset on 4096 frozen pixels, residual pilot, scale scan | ~15 min |
| `stock.slurm` + `stock_check.py` | the authors' own loopscape code path on the same pixels, several numeric configs | ~20 min |
| `eechk.slurm` + `earlyexit_check.py` | early exit vs full rollouts vs batch-size changes (same pixels) | ~10 min |
| `pscan.slurm` + `puzzle_scan.py` | multistability scan: 5565 puzzles x 16 native random starts | ~30 min |
| `chase.slurm` + `chase.py` | zoom chase: small windows, richest sub-window each level, to choose the keyframe centre | ~30 min |
| `render.slurm` | tiled, resumable window/asset-grid renders (hero, keyframes, reproduction grid) from a spec | see below |

## Tiled renders (`render.slurm`)

A spec (`alice/specs/*.json`) describes a window of the (a, b) plane or the upstream asset's grid, the tile size,
precision and early exit. The image is cut into tiles; every finished tile is one small `.npz`, written atomically,
so a job killed at the limit loses at most one tile and re-submitting the same command continues where it stopped.

```bash
ssh alice2 'cd ~/reasoning-fractals && sbatch --array=0-7 \
   --export=ALL,SPEC=alice/specs/hero.json,OUT=out/hero,WORKERS=8 alice/render.slurm'
alice/sync.sh pull hero                    # -> data/cluster/hero (tiles, spec.json)
.venv/bin/python -m render.tilerun merge --spec data/cluster/hero/spec.json --out data/cluster/hero
```

Array task *k* of *W* renders every *W*-th tile. Never mix different `WORKERS` values on the same output folder
while jobs are running (two workers could pick the same tile). Each tile must finish well inside the margin the
worker keeps before the 4 h limit (`MAX_MIN`, default 215 min); tiles here take 2-12 min.

## Numerics (why everything is fp32)

On 4096 pixels of the upstream asset's grid, true fp32 agrees with the published image on 65% of pixels (99.9% where
the image is locally flat, 48% in the fractal filaments); TF32 agrees on 22% and bf16 on 14%, so reduced precision
is a different realisation even in flat regions. All deliverable layers are fp32. Throughput of the eager torch
backend on an A100-80GB-PCIe is ~23 rollouts/s in fp32 (a rollout = 24 loops of one starting thought); early exit
(retire a pixel after its decoded grid has equalled the true solution for 3 loops) cuts the loops computed per pixel
to ~9 (`RUNS.md`).
