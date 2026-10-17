<!-- Generated: edit docs/README.template.md, then `.venv/bin/python scripts/build_readme.py`. Every number below comes from results/*.json. -->
# reasoning-fractals: the fractal "mind" of a reasoning model

![hero](media/hero-1080x1350.png)

Give a recurrent reasoning model (EqR, a small Sudoku solver that thinks by iterating) the **same Sudoku** many
times and change only its **initial latent "thought"**. Colour every starting point by how many loops the model
needed before its answer stopped changing. The map of starting points is a fractal: a tiny nudge to the first
thought can change how long the model thinks by many loops.

This project reproduces the central figure of **GilpinLab/loopscape** (Lai, Bao, Quinn, Gilpin, *Fractal basins trap
latent reasoning*, arXiv:2609.04963), renders it at {{hero.side|int}} x {{hero.side|int}} pixels on the ALICE GPU cluster,
adds a deep-zoom video and a click-a-pixel explorer, and measures how far the claims reach.

**In numbers**

* The upstream figure reproduces: **{{repro.agreement|pct1}}** of {{repro.n_pixels|int}} pixels identical, {{repro.interior|pct2}} where
  the picture is smooth, {{repro.boundary|pct1}} inside the filaments. The statistics match (boundary dimension
  {{repro.d_ours|f3}} vs {{repro.d_asset|f3}}). The filament pixels are rounding-sensitive; the authors' own code agrees with their
  image no better than ours does ({{stock.fp32_agree|pct1}}).
* The hero has {{hero.pixels|int}} starting thoughts. {{hero.frac_solved|pct2}} of them end on the true solution: the fractal is in
  *how long* the model thinks, not in whether it solves this puzzle.
* Neighbouring pixels differ a lot: {{hero.gap_ge5|pct1}} of neighbour pairs differ by 5 loops or more, the largest gap is
  {{hero.max_gap|int}} loops.
* Cost: {{gpu.a100_hours|f1}} A100-hours on {{gpu.jobs|int}} cluster jobs for everything in this repository.

## What is in the box

| path | what |
|---|---|
| `media/hero.png` (+ `hero-1080x1350.png`, `hero-1080x1080.png`) | the hero image, {{hero.side|int}}² native |
| `media/zoom-1080x1350.mp4`, `media/zoom-1080x1080.mp4` | deep-zoom video, 20 s, H.264, no audio |
| `site/` | the explorer: a static page (canvas, ES modules, no build). `python3 -m http.server` from `site/` |
| `render/` | the pipeline (tested): engine, slices, outcomes, tiled renders, colourways, pack builder, zoom compositor |
| `alice/` | Slurm scripts, specs and the cluster README (everything that ran on ALICE) |
| `results/` | every number quoted here, as JSON, with the script that produced it (`results/README.md`) |
| `RUNS.md` | one row per ALICE experiment (job ids, config, metric, verdict, GPU-hours) |
| `tests/` | `pytest` suite (the engine tests need the EqR weights under `data/raw`) |

## How the picture is made

![how it works](docs/how-it-works.svg)

* **The model.** EqR (Equilibrium Reasoners, Huang, Geng & Kolter) keeps two latent states `z_H`, `z_L` of 97 x 512
  numbers (16 register tokens + the 81 Sudoku cells). One *loop* is 3 x (6 updates of `z_L`, then one of `z_H`) with a
  0.95 relaxation; after every loop `lm_head(z_H)` is decoded into a 9x9 grid. With its noise switched off (as loopscape
  does) the whole thing is a deterministic map of the initial latent, run for 24 loops.
* **The slice.** `z_H = a*u + b*v`, `z_L = 0`, with `u, v` a random orthonormal pair of directions in the 49,664-dimensional
  space (`torch.manual_seed(0)`, QR), `(a, b)` on a grid in [-1, 1]^2. Every pixel is one starting thought for the same
  puzzle.
* **Settle time** of a pixel: `24 - #{loops whose decoded grid equals the final grid}` (loopscape's definition; the
  loop after which the grid never changes again is stored as well). Pixels still changing at the last loop are flagged
  *censored*.
* **Outcome** of a pixel: the final decoded grid. Each distinct grid gets an id (0 = the true solution), stored with its
  count and whether it is a valid Sudoku.
* **Colour.** Settle time is the luminance (dark = fast), the outcome picks the hue: two colourways in
  `render/colour.py`, an indigo-teal-mint one close to loopscape's look and one in the series palette (midnight and
  amber). A pixel that ends on a wrong grid is drawn from a separate, clearly different ramp.
* **Early exit.** A start is retired once its decoded grid has equalled the true solution for three loops in a row; the
  remaining loops are filled in. Checked against full 24-loop runs on {{stock.n|int}} pixels (`RUNS.md` row 4): identical settle
  time for {{ee.settle_equal|pct2}} of the pixels, {{ee.speedup|x1}} faster ({{ee.mean_loops|f1}} loops computed per pixel instead of 24).
* **float32 only.** TF32 gives a different picture even in smooth regions ({{smoke.tf32_agree|pct1}} of pixels identical to the
  upstream image, bfloat16 {{smoke.bf16_agree|pct1}}, true float32 {{smoke.fp32_agree|pct1}}), so every layer here is fp32 with TF32 off.

## Quickstart

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements.txt           # Python 3.12
git clone https://github.com/GilpinLab/loopscape vendor/loopscape           # read it before you run it; no licence file
uv pip install -e vendor/loopscape && source scripts/paths.sh               # EqR source pinned at commit e9934826
.venv/bin/python -m pytest                                                  # unit tests (engine tests are marked slow)
(cd site && python3 -m http.server 8772)                                    # the explorer: http://127.0.0.1:8772/
```

Render a small window yourself (any machine; about {{m4.mlx_rps|f1}} rollouts/s on an Apple M4 with MLX, see the throughput table):

```bash
# a spec file describes the window, e.g. alice/specs/kf05.json; --max-tiles keeps a laptop run short
.venv/bin/python -m render.tilerun run --spec alice/specs/kf05.json --out data/local/kf05 --max-tiles 1
```

The cluster runs (hero, keyframes, scans) are in `alice/README.md`; the pack, video and README builders are
`scripts/build_pack.py`, `scripts/make_video.py`, `scripts/build_readme.py`.

## Results

### 1. The upstream figure reproduces

The upstream repository ships a 1000x1000 basin image of EqR on one Sudoku-Extreme puzzle. Its file name carries the
puzzle: `blake2s(question, digest_size=8)` of
`2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3...` (rating 47, Sudoku-Extreme test split)
is exactly `7d68927f389fbdaa`. (The notebook's own EqR puzzle is a different one, rating 162.)

With the notebook's recipe (plane seed 0, `(a, b)` in [-1, 1]^2, 24 loops, noise off, float32) the image is the
settle-time field with 11 levels (settle time 2..12, the top level saturated), drawn with y pointing up. Comparison on
every 4th pixel of the upstream image (250x250 grid, fp32 on an A100, `RUNS.md` row 2):

{{table:repro}}

Why not 100%? The filaments are numerically touchy. The authors' *own* code (loopscape's stock solver, notebook batching)
run on the same A100 agrees with their image on {{stock.fp32_agree|pct1}} (smooth regions {{stock.fp32_interior|pct1}}, filaments
{{stock.fp32_boundary|pct1}}), and our fast engine agrees with that stock solver on {{stock.vs_ours|pct1}}: two valid float32 stacks agree
with each other exactly as much as with the upstream image. The upstream image is not committed here (the repository has
no licence); the side-by-side figure is regenerated locally by `scripts/repro_report.py` when `vendor/loopscape` is present.

### 2. The hero: what the map looks like

The full plane `[-1, 1]^2` at {{hero.side|int}} x {{hero.side|int}} (`media/hero.png`, `alice/specs/hero.json`, jobs in `RUNS.md`). Colour is settle time; nearly
everything ends on the true solution, so the picture is a map of *how long* the model thinks.

{{table:hero}}

The last row is the upstream image analysed with the same code (`loopscape.metrics`, 5x5 boxes, eps = 1..5 px); our map at the
same pixel pitch (every 2nd pixel of the hero) lands close to it. All metrics depend on the pixel pitch, so each row says which one.

### 3. Zoom: the fractal is "slim"

Zooming into the map does not reveal the same complexity again and again: the share of 5x5 boxes that touch a basin boundary falls as the pitch gets
finer, and the finest windows turn into flat domains next to fine speckle. This is the scale dependence the paper calls a *slim* fractal (appendix E).
The zoom video and the explorer therefore stop at ×{{kf.zoom_max|int}}; a chase to ×5,120 found no coherent structure beyond that, only speckle
next to a flat domain.

{{table:scale}}

### 4. How far can a single pixel be trusted?

Random pixels of the chase windows and of the hero were re-run on a second float32 implementation (MLX on an Apple M4; the renders are torch on
A100s) and compared with the stored settle times ({{replay.n_per_field|int}} pixels per field, {{replay.n_fields|int}} fields, `scripts/replay_agreement.py`).
In flat regions the second implementation gives the same number. In the speckled "dust" it mostly does not, and the slow dust pixels
essentially never do. The *statistics* of the dust (histograms, basin entropy, boundary dimension) are stable; individual dust pixels
are one realisation of float32 rounding. Nudging a start by a relative 1e-6 (a few dozen float32 ulps) moves such pixels by several loops
(`results/replay/summary.json`).

{{table:replay}}

### 5. Native-scale starts: when does EqR really fail?

The hero slice stays within distance 1.4 of the all-zero state, while EqR's own random initialisation has norm about 223. At that scale the picture changes:

{{table:scans}}

So EqR is far from perfect on Sudoku-Extreme from its own random starts, but the *wrong* endings are not stable wrong answers: they wander. That is why the
explorer calls a pixel that is still changing after 24 loops `UNSETTLED`, and reserves `STUCK` for a wrong grid that stopped changing.

### 6. Throughput and cost

Measured on this Apple M4 (shared with other jobs) and on the ALICE A100s; one rollout is 24 loops of one starting thought
(about {{m4.gflop_per_rollout|f1}} GFLOP).

{{table:throughput}}

Time to render, from those rates (`scripts/cost_estimate.py`; early exit computes {{ee.mean_loops|f1}} loops per pixel instead of 24):

{{table:cost}}

The hero and keyframes actually cost {{hero.gpu_hours|f1}} and {{kf.gpu_hours|f1}} GPU-hours of tile time (summed over workers; full A100s and MIG slices mixed);
everything in this repository, scans and checks included, used {{gpu.a100_hours|f1}} A100-equivalent hours ({{gpu.jobs|int}} jobs, `results/alice/gpu_hours.json`).

## Honest limits

* **One puzzle, one model.** EqR's released Sudoku checkpoint on one rating-47 puzzle, the one behind loopscape's figure. Other puzzles give other maps.
* **One slice, tiny thoughts.** A random 2-D plane (seed 0) through the all-zero start. At EqR's own start scale the puzzle scan found failures on
  {{pscan.multistable|int}} of {{pscan.puzzles|int}} puzzles, but this puzzle was solved from all {{pscan.k|int}} of its native starts.
* **Almost every start ends right.** {{hero.frac_solved|pct2}} of the hero's starts reach the true solution; the {{hero.n_unsolved|int}} others are mostly starts still
  changing at loop 24. The fractal is in the settle time.
* **Pixel-exact only where the map is smooth.** See section 4: a second float32 implementation agrees on {{replay.flat|pct0}} of pixels with flat surroundings
  and on {{replay.dust|pct0}} of speckle pixels. The reproduction of the upstream figure is statistical in the filaments for the same reason.
* **float32 only.** TF32 and bfloat16 give different pictures ({{smoke.tf32_agree|pct1}} and {{smoke.bf16_agree|pct1}} pixel agreement with the upstream image).
* **Slim fractal.** The structure thins out under magnification (section 3); nothing was rendered beyond ×{{kf.zoom_max|int}}.
* **Early exit** retires a start after 3 loops on the solution: identical settle times for {{ee.settle_equal|pct2}} of 4,096 test pixels. Batch size matters (B=2048 is a
  different cuBLAS kernel regime and agrees on only {{ee.b2048_equal|pct1}}), so production used the validated batch of 1,024.
* **Metrics depend on the pixel pitch** (and on the labels: here settle time, as in the paper).
* **The upstream repository has no licence**, so none of its code or images is redistributed; see Credits.

## Credits and licences

* **loopscape** and the paper (Lai, Bao, Quinn, Gilpin), https://github.com/GilpinLab/loopscape. The repository carries
  no licence, so none of its code or images is redistributed here (`vendor/` is gitignored); the upstream image and
  numbers derived pixel-by-pixel from it stay local and out of git.
* **EqR** (Huang, Geng, Kolter, arXiv:2605.21488), Apache-2.0 code and weights from locuslab
  (`locuslab/EqR`, pinned commit e9934826; `locuslab/EqR-model`).
* **Sudoku-Extreme** puzzles (HRM, Wang et al.; HF `sapientinc/sudoku-extreme`).
* FPRM (Movahedi et al., arXiv:2606.18206) is the other model loopscape wraps; it is not used here.
* Compute: ALICE, Leiden University (A100 GPUs, partition `gpu-short`).
