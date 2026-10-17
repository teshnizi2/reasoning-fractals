# reasoning-fractals: the fractal "mind" of a reasoning model

Give a recurrent reasoning model (EqR, a small Sudoku solver that thinks by iterating) the **same Sudoku** many
times and change only its **initial latent "thought"**. Colour every starting point by what happens next. The map
of starting points is a fractal: a tiny nudge to the first thought can change how long the model needs to settle on
its answer by many loops.

This project reproduces the central figure of **GilpinLab/loopscape** (Lai, Bao, Quinn, Gilpin, *Fractal basins trap
latent reasoning*, arXiv:2609.04963), renders it at high resolution on the ALICE cluster, adds a deep-zoom video and a
click-a-pixel explorer, and measures how far the claims reach.

> **Status of this file:** numbers below come from `results/` and `RUNS.md`; anything marked *pending* has not been
> measured yet. See `STATE.md` for the current state of the work.

## What is in the box

| path | what |
|---|---|
| `media/hero.png`, `media/zoom-1080x1350.mp4` | hero image and deep-zoom video (*pending, see STATE.md*) |
| `site/` | the explorer: static page (canvas, ES modules, no build), `python3 -m http.server` from `site/` |
| `render/` | the pipeline (tested): engine, slices, outcomes, tiled renders, colourways, pack builder, zoom compositor |
| `alice/` | Slurm scripts, specs and the cluster README (everything that ran on ALICE) |
| `results/` | every number quoted here, as JSON, with the script that produced it |
| `RUNS.md` | one row per ALICE experiment (job ids, config, metric, verdict, GPU-hours) |
| `tests/` | `pytest` suite (about 120 tests; the engine tests need the EqR weights under `data/raw`) |

## What was reproduced

The upstream repository ships a 1000x1000 basin image of EqR on one Sudoku-Extreme puzzle
(`assets/basins_1000_0_eqr_sudoku_plane_rating47_7d68927f389fbdaa_rating47.png`). Its file name carries the puzzle:
`blake2s(question, digest_size=8)` of the puzzle string
`2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3...` (rating 47, Sudoku-Extreme test
split) is exactly `7d68927f389fbdaa`. (The notebook's own EqR puzzle is a different one, rating 162.)

With the notebook's recipe (`z_H = a*u + b*v`, `z_L = 0`, plane seed 0, `(a, b)` in [-1, 1]^2, 24 loops, noise off,
float32) the image is the settle-time field with 11 levels (settle time 2..12, the top level saturated), drawn with
y pointing up.

Pixel-by-pixel comparison on every 4th pixel of the upstream image (250x250 = 62,500 pixels, fp32 on an A100,
`RUNS.md` row 2, `results/repro/repro250_summary.json`):

| | ours | upstream image |
|---|---|---|
| pixels identical | **64.4%** (99.94% of the pixels where the upstream image is locally flat, 48.1% inside the fractal filaments) | |
| Spearman correlation of settle time | 0.889 | |
| basin entropy S_b (5x5 boxes) | 1.416 | 1.412 |
| boundary basin entropy S_bb | 1.557 | 1.552 |
| boundary dimension 2 - alpha | 1.873 | 1.872 |

![reproduction](results/repro/repro250_side_by_side.png)

Why not 100%? The filaments are numerically touchy. The authors' *own* code (loopscape's stock solver, notebook
batching) run on the same A100 agrees with their image on the same 64.7% (flat regions 99.9%, filaments 47.7%), and
our fast engine agrees with the stock solver on 63.7%, i.e. different but equally valid fp32 stacks agree with each
other exactly as much as with the upstream image. Reduced precision is a different realisation even in flat
regions (TF32: 21.9% identical, bf16: 13.6%), which is why every layer here is fp32.

## Honest limits

*(to be completed with the final numbers; see STATE.md)*

## Credits and licences

* **loopscape** and the paper (Lai, Bao, Quinn, Gilpin), https://github.com/GilpinLab/loopscape. The repository carries
  no licence, so none of its code or images is redistributed here (`vendor/` is gitignored); the upstream image is read
  locally only for the comparison above.
* **EqR** (Huang, Geng, Kolter, arXiv:2605.21488), Apache-2.0 code and weights from locuslab
  (`locuslab/EqR`, pinned commit e9934826; `locuslab/EqR-model`).
* **Sudoku-Extreme** puzzles (HRM, Wang et al.; HF `sapientinc/sudoku-extreme`).
* FPRM (Movahedi et al., arXiv:2606.18206) is the other model loopscape wraps; it is not used here.
