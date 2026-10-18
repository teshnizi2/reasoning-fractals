<!-- Generated: edit docs/README.template.md, then `.venv/bin/python scripts/build_readme.py`. Every number below comes from results/*.json. -->
# reasoning-fractals: the fractal "mind" of a reasoning model

**Live explorer: <https://fractals.teshnizi.dev>** (click any pixel to see that starting thought's loops and Sudoku; "Fly in" zooms through the layers) · MIT licence

![hero](media/hero-1080x1350.png)

Give a recurrent reasoning model (EqR, a small Sudoku solver that thinks by iterating) the **same Sudoku** many
times and change only its **initial latent "thought"**. Colour every starting point by how many loops the model
needed before its answer stopped changing. The map of starting points is a fractal: a tiny nudge to the first
thought can change how long the model thinks by many loops.

This project reproduces the central figure of **GilpinLab/loopscape** (Lai, Bao, Quinn, Gilpin, *Fractal basins trap
latent reasoning*, arXiv:2609.04963), renders it at 2,048 x 2,048 pixels on the ALICE GPU cluster,
adds a deep-zoom video and a click-a-pixel explorer, and measures how far the claims reach.

**In numbers**

* The upstream figure reproduces: **64.4%** of 62,500 pixels identical, 99.94% where
  the picture is smooth, 48.1% inside the filaments. The statistics match (boundary dimension
  1.873 vs 1.872). The filament pixels are rounding-sensitive; the authors' own code agrees with their
  image no better than ours does (64.7%).
* At the hero's full resolution the statistics still match: with the same 11 labels and the same pixel pitch the boundary dimension is
  1.853 against the upstream image's 1.852, basin entropy 1.139 against 1.141.
* The hero has 4,194,304 starting thoughts. 99.92% of them end on the true solution: the fractal is in
  *how long* the model thinks, not in whether it solves this puzzle.
* Neighbouring pixels differ a lot: 12.3% of neighbour pairs differ by 5 loops or more, the largest gap is
  21 loops.
* Cost: 31.6 A100-equivalent hours (49.1 GPU-slot hours; a MIG slice counts as 4/7 of an A100) over 174 cluster job records (array tasks counted one by one) for everything in this repository.

## What is in the box

| path | what |
|---|---|
| `media/hero.png` (+ `hero-1080x1350.png`, `hero-1080x1080.png`) | the hero image, 2,048² native |
| `media/zoom-1080x1350.mp4`, `media/zoom-1080x1080.mp4` | deep-zoom video, 20 s, H.264, no audio |
| `site/` | the explorer: a static page (canvas, ES modules, no build). `python3 -m http.server` from `site/` |
| `render/` | the pipeline (tested): engine, slices, outcomes, tiled renders, colourways, pack builder, zoom compositor |
| `alice/` | Slurm scripts, specs and the cluster README (everything that ran on ALICE) |
| `results/` | every number quoted here, as JSON, with the script that produced it (`results/README.md`) |
| `RUNS.md` | one row per ALICE experiment (job ids, config, metric, verdict, GPU-hours) |
| `tests/` | `pytest` suite (the engine tests need the EqR weights under `data/raw`) |

## The explorer

![the explorer on a desktop](media/explorer-desktop.png)

`site/` is a static page (no build step): the hero as a zoomable canvas with 5 nested zoom layers up to ×128.
Click or tap any pixel to see the Sudoku that starting thought led to: the 9x9 grid, the givens, the cells that differ from the true solution
(amber, true digit small), a `SOLVED` / `UNSETTLED` / `STUCK` badge, and how many loops it needed. A few featured starts replay their thinking loop by loop
(traces from the cluster's own engine). The inspector also says whether the pixel sits in a flat region or in the rounding-sensitive speckle.
The "What am I looking at?" drawer explains the picture and repeats the limits below; on a phone the inspector is a bottom sheet.

<img src="media/explorer-phone.png" alt="the explorer on a phone" width="320">

## The video

`media/zoom-1080x1350.mp4` (4:5 LinkedIn feed) and `media/zoom-1080x1080.mp4`: 20 s, H.264, yuv420p, 30 fps, no audio. The camera starts on the whole map and swings to the vortex
at the top right, then dives through the 5 keyframe layers (the layers end at ×128; the zoom counter is the magnification of the whole map). In the last seconds the picture is the
deepest layer magnified into single pixels, each block one starting thought. The finale rings two neighbouring starts, 2.9×10^-5 apart in the plane, and replays
their Sudoku loop by loop beside each other: A settles in 5 loops, B in 21. B sits in the speckle, so a footnote on screen gives what 1e-6 nudges did to it (6 to 21 loops);
the slow/fast order held in 5 of 6 runs. The traces are the cluster engine's own (`results/video/pair.json`, RUNS rows 12 and 14).
Rebuild with `scripts/finish.sh 6` (about ten minutes of CPU); the files are not in git (50-65 MB each at CRF 16).

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
  `render/colour.py`, an indigo-teal-mint one close to loopscape's look (`media/loopscape/`) and one in the series
  palette (midnight and amber), which is the one used for the hero, the video and the explorer's default: it keeps
  the flat domains calm and lets the speckle read as gold leaf. A pixel that ends on a wrong grid is drawn from a
  separate, clearly different ramp.
* **Early exit.** A start is retired once its decoded grid has equalled the true solution for three loops in a row; the
  remaining loops are filled in. Checked against full 24-loop runs on 4,096 pixels (`RUNS.md` row 4): identical settle
  time for 99.93% of the pixels, 2.7× faster (8.8 loops computed per pixel instead of 24).
* **float32 only.** TF32 gives a different picture even in smooth regions (21.9% of pixels identical to the
  upstream image, bfloat16 13.6%, true float32 65.0%), so every layer here is fp32 with TF32 off.

## Quickstart

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements.txt           # Python 3.12
git clone https://github.com/GilpinLab/loopscape vendor/loopscape           # read it before you run it; no licence file
uv pip install -e vendor/loopscape && source scripts/paths.sh               # EqR source pinned at commit e9934826
.venv/bin/python -m pytest                                                  # unit tests (engine tests are marked slow)
(cd site && python3 -m http.server 8772)                                    # the explorer: http://127.0.0.1:8772/
```

Render a small window yourself (about 1.2 rollouts/s on an Apple M4 with MLX, so this 16 x 16 window takes a few minutes; see the throughput table):

```bash
# needs the EqR weights under data/raw (see requirements.txt) and the frozen plane data/plane_seed0.npz
.venv/bin/python -m render.tilerun run --spec alice/specs/demo16.json --out data/local/demo16 --backend mlx
.venv/bin/python -m render.tilerun merge --spec alice/specs/demo16.json --out data/local/demo16   # field.npz + dictionary.json
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

|  | ours (fp32, A100) | upstream image |
|---|---|---|
| pixels identical | **64.4%** of 62,500 |  |
| ...where the upstream image is locally flat | 99.94% (on 31% of the pixels) |  |
| ...inside the filaments | 48.1% |  |
| Spearman correlation of settle time | 0.889 |  |
| basin entropy S_b (5x5 boxes) | 1.416 | 1.412 |
| boundary basin entropy S_bb | 1.557 | 1.552 |
| uncertainty exponent alpha | 0.127 | 0.128 |
| boundary dimension 2 - alpha | 1.873 | 1.872 |

Why not 100%? The filaments are numerically touchy. The authors' *own* code (loopscape's stock solver, notebook batching)
run on the same A100 agrees with their image on 64.7% (smooth regions 99.9%, filaments
47.7%), and our fast engine agrees with that stock solver on 63.7%: two valid float32 stacks agree
with each other exactly as much as with the upstream image. The upstream image is not committed here (the repository has
no licence); the side-by-side figure is regenerated locally by `scripts/repro_report.py` (it needs `vendor/loopscape` and the cluster pull `data/cluster/repro250`; our own render of that grid is `results/repro/repro250_field.npz`).

### 2. The hero: what the map looks like

The full plane `[-1, 1]^2` at 2,048 x 2,048 (`media/hero.png`, `alice/specs/hero.json`, jobs in `RUNS.md`). Colour is settle time; nearly
everything ends on the true solution, so the picture is a map of *how long* the model thinks.

| hero (full plane) |  |
|---|---|
| starting thoughts (pixels) | 4,194,304 (2048 x 2048 over [-1, 1]^2) |
| end on the true solution | 99.92% (4,190,863) |
| still changing at loop 24 | 0.12% |
| distinct final grids (wrong ones) | 2,402 (2,401) |
| settle time, mean / max (loops) | 5.90 / 23 |
| loops computed per pixel (early exit) | 8.9 |
| neighbour pairs differing by >= 1 loop | 44.2% |
| neighbour pairs differing by >= 2 loops | 32.4% |
| neighbour pairs differing by >= 5 loops | 12.3% |
| neighbour pairs differing by >= 10 loops | 2.1% |
| largest neighbour gap (loops) | 21 |

| settle-time labels, 5x5 boxes | pixel pitch | boxes touching a boundary | S_b | S_bb | alpha | boundary dim. 2 - alpha |
|---|---|---|---|---|---|---|
| hero, every pixel | 9.8×10^-4 | 69% | 1.086 | 1.574 | 0.137 | 1.863 |
| hero, every 2nd pixel | 2.0×10^-3 | 76% | 1.198 | 1.576 | 0.144 | 1.856 |
| hero, every 4th pixel | 3.9×10^-3 | 84% | 1.325 | 1.584 | 0.140 | 1.860 |
| hero, every 2nd pixel, labels clipped to the upstream's 11 levels | 2.0×10^-3 | 76% | 1.139 | 1.498 | 0.147 | 1.853 |
| upstream image, 1000 x 1000 (metrics computed by us from its 11 colour levels) | 2.0×10^-3 | 76% | 1.141 | 1.496 | 0.148 | 1.852 |

The last two rows are the like-for-like comparison: the hero with its settle times clipped to the upstream image's 11 levels and sampled at the same pixel
pitch, against the upstream image analysed with the same code (`loopscape.metrics`, 5x5 boxes, eps = 1..5 px). They agree to about two decimals in every
metric. All metrics depend on the pixel pitch and on the labels, so each row says which.

### 3. Zoom: the fractal is "slim"

Zooming into the map does not reveal the same complexity again and again. Around the zoom target the share of 5x5 boxes that touch a basin boundary stays
high down to about ×32 (75% at ×8, 71% at ×32) and then falls: 45% at ×64 and 31% at ×128, with the basin entropy
dropping from 0.83 to 0.30. What is left at the finest layers is a flat domain next to fine speckle (the boundary dimension of that speckle edge is 1.93, close
to the 2 of pure noise). This is the scale dependence the paper calls a *slim* fractal (appendix E). The zoom video and the explorer therefore stop at ×128;
a trial window at ×5,120 found no coherent structure, only speckle next to a flat domain.

All rows use the same 11 labels (settle time clipped to 2..12, as in the upstream image). The keyframes are unbiased (centred on the zoom target);
the chase windows were chosen for their boundary richness.

| field | pixel pitch (plane units) | pixels | 5x5 boxes touching a boundary | S_b | boundary dim. |
|---|---|---|---|---|---|
| upstream image, every 8th pixel | 1.6×10^-2 | 125x125 | 96% | 1.55 | 1.897 |
| upstream image, every 4th pixel | 8.0×10^-3 | 250x250 | 90% | 1.41 | 1.873 |
| ours, 250x250 (asset grid, fp32) | 8.0×10^-3 | 250x250 | 91% | 1.42 | 1.873 |
| hero 2048x2048, every 8th pixel | 7.8×10^-3 | 256x256 | 90% | 1.41 | 1.871 |
| upstream image, every 2nd pixel | 4.0×10^-3 | 500x500 | 84% | 1.27 | 1.857 |
| hero 2048x2048, every 4th pixel | 3.9×10^-3 | 512x512 | 84% | 1.27 | 1.857 |
| upstream image, every pixel | 2.0×10^-3 | 1000x1000 | 76% | 1.14 | 1.852 |
| hero 2048x2048, every 2nd pixel | 2.0×10^-3 | 1024x1024 | 76% | 1.14 | 1.853 |
| hero 2048x2048, every pixel | 9.8×10^-4 | 2048x2048 | 69% | 1.02 | 1.860 |
| keyframe x8 (540x675, centred on the zoom target) | 4.6×10^-4 | 540x675 | 75% | 1.07 | 1.890 |
| keyframe x16 (540x675, centred on the zoom target) | 2.3×10^-4 | 540x675 | 70% | 0.96 | 1.938 |
| window x80 (160x160, chase2) | 1.6×10^-4 | 160x160 | 44% | 0.47 | 1.922 |
| keyframe x32 (540x675, centred on the zoom target) | 1.2×10^-4 | 540x675 | 71% | 0.83 | 1.932 |
| keyframe x64 (540x675, centred on the zoom target) | 5.8×10^-5 | 540x675 | 45% | 0.46 | 1.927 |
| keyframe x128 (540x675, centred on the zoom target) | 2.9×10^-5 | 540x675 | 31% | 0.30 | 1.934 |
| window x640 (160x160, chase2) | 2.0×10^-5 | 160x160 | 48% | 0.28 | 1.901 |
| window x5120 (160x160, chase2) | 2.5×10^-6 | 160x160 | 73% | 0.63 | 1.998 |

### 4. How far can a single pixel be trusted?

Two checks, both on random pixels of the hero (and of the chase windows), stratified by how rough the stored field is around the pixel
(the largest settle-time difference to its 4 neighbours).

*Same engine, rounding-sized nudges* (`alice/replay.py`, `results/replay/summary.json`): each pixel is re-run on the A100 in full (no early exit),
and 5 more times with the starting thought multiplied by `1 + 1e-6 * N(0, 1)`, a few dozen float32 ulps.
A plain re-run reproduces the stored settle time for 100.0% of the 600 pixels, so what the explorer and the video show is
what the stored pixel says (the exception is the very slowest pixels: only 6 of the 12 slowest candidates reproduce, because a pixel that is still changing in the last loops of a run also depends on how the GPU batch around it was composed; the explorer only features pixels that did reproduce). The nudged runs show how fragile a pixel is:

| pixel's surroundings (stored field) | pixels re-run | plain re-run reproduces the stored settle time | unchanged by all 5 nudged re-runs (relative 1e-06) | mean largest change from a nudge (loops) |
|---|---|---|---|---|
| all 4 neighbours equal (flat) | 150 | 100.0% | 97% | 0.1 |
| a neighbour differs by 1 loop | 150 | 100.0% | 55% | 1.9 |
| a neighbour differs by 2-4 loops | 150 | 100.0% | 17% | 4.2 |
| a neighbour differs by 5 or more | 150 | 100.0% | 19% | 4.6 |

*A second float32 implementation* (MLX on an Apple M4 against the stored A100 values, 160 pixels per field,
4 fields, `scripts/replay_agreement.py`):

| pixel's surroundings (stored field) | pixels checked | same settle time on a second float32 stack | ...of those stored as >= 8 loops |
|---|---|---|---|
| all 4 neighbours equal (flat) | 357 | 97% | - |
| a neighbour differs by 1 loop | 90 | 66% | - |
| a neighbour differs by 2-4 loops | 89 | 26% | 2 of 41 |
| a neighbour differs by 5 or more | 104 | 26% | 1 of 47 |

In flat regions every implementation gives the same number. In the speckled "dust" a different implementation, or a nudge below the
noise of a float32 matmul, changes the number, and slow dust pixels essentially never reproduce across implementations. The *statistics* of the
dust (histograms, basin entropy, boundary dimension) are stable; an individual dust pixel is one realisation of float32 rounding.

### 5. Native-scale starts: when does EqR really fail?

The hero slice stays within distance 1.41 of the all-zero state, while EqR's own random initialisation has norm about 223. At that scale the picture changes:

| scan | what was run | result |
|---|---|---|
| puzzle scan | 5,565 puzzles x 16 EqR-style random starts, 24 loops (job 5246311) | 90.1% end on the solution; 4,281 puzzles always, 115 never, 664 between 15% and 85% |
| ...the hero's puzzle | same scan | 16 of 16 starts solved (mean settle time 4.1) |
| long run | 400 starts that were unsolved at loop 24, re-run for 96 loops (job 5246477) | 46.8% reach the solution; 213 are still changing, 0 are frozen on a wrong grid |
| outcome maps | 15 windows of 48 x 48 around native-scale base points, 2 puzzles (job 5246477, cancelled at 15 of 32 windows) | the hero puzzle: 99.8%-100.0% solved, up to 5 distinct final grids; puzzle #72: 54.0%-60.5% solved, up to 1,060 distinct final grids, neighbouring pixels share their outcome 53% of the time (independent pixels would: 51%) |

![outcome maps at native scale](results/figures/oscan_sheet.png)

*Each tile is a 48 x 48 window of a random 2-D slice around a native-scale starting point, half-width h = 10, 30, 100, 300 (left to right, seeds 0 then 1). Top two
rows: the hero's puzzle, coloured by settle time: rings and spirals with speckle, all solved. Bottom row: puzzle #72 of the scan, where the model fails about
half the time: white = still changing at loop 24, blue-green = solved (darker = faster). Outcomes there look like noise.*

So EqR is far from perfect on Sudoku-Extreme from its own random starts, but the *wrong* endings are not stable wrong answers: they wander. That is why the
explorer calls a pixel that is still changing after 24 loops `UNSETTLED`, and reserves `STUCK` for a wrong grid that stopped changing.

### 6. Throughput and cost

Measured on this Apple M4 (shared with other jobs) and on the ALICE A100s; one rollout is 24 loops of one starting thought
(about 615.3 GFLOP).

| where | rollouts per second (24 loops each) | source |
|---|---|---|
| Apple M4 CPU, torch fp32 | 0.37 | results/throughput_m4.json |
| Apple M4 GPU, torch MPS fp32 | 1.04 | results/throughput_m4.json |
| Apple M4 GPU, MLX fp32 | 1.19 | results/throughput_m4.json |
| Apple M4 GPU, loopscape stock path (MPS) | 0.92 | results/throughput_m4.json |
| A100-80GB, fp32 eager, batch 1024 | 23.2 | results/alice/smoke_5246129_bench.json |
| A100-80GB, TF32 (not used: different picture) | 60.4 | same |
| A100-80GB, bfloat16 (not used) | 90.5 | same |
| A100 MIG 4g.40gb slice, fp32 | 12.3 | results/alice/eechk_5246309.json |
| A100 MIG 4g.40gb slice, fp32 + early exit | 33.0 | same |
| production hero tiles (mix of full A100s and MIG slices, early exit) | 39.2 | results/metrics.json (pixels / summed tile seconds) |

Time to render, from those rates (`scripts/cost_estimate.py`; early exit computes 8.8 loops per pixel instead of 24):

| render | Apple M4, MLX | A100, fp32 eager | A100, fp32 + early exit | A100 MIG 4g.40gb + early exit |
|---|---|---|---|---|
| 1000x1000 (the upstream asset's resolution): 1,000,000 px | 234.2 h | 12.0 h | 4.4 h | 8.3 h |
| 2048x2048 hero: 4,194,304 px | 982.4 h | 50.3 h | 18.5 h | 35.0 h |
| zoom: 5 keyframes of 540x675: 1,822,500 px | 426.9 h | 21.8 h | 8.0 h | 15.2 h |

The hero and keyframes actually cost 29.7 and 15.1 GPU-hours of tile time (summed over workers; full A100s and MIG slices mixed);
everything in this repository, scans and checks included, used 31.6 A100-equivalent hours (174 jobs, `results/alice/gpu_hours.json`).

## Honest limits

* **One puzzle, one model.** EqR's released Sudoku checkpoint on one rating-47 puzzle, the one behind loopscape's figure. Other puzzles give other maps.
* **One slice, tiny thoughts.** A random 2-D plane (seed 0) through the all-zero start. At EqR's own start scale the puzzle scan found failures on
  664 of 5,565 puzzles, but this puzzle was solved from all 16 of its native starts.
* **Almost every start ends right.** 99.92% of the hero's starts reach the true solution; the 3,441 others are mostly starts still
  changing at loop 24. The fractal is in the settle time.
* **Pixel-exact only where the map is smooth.** See section 4: a second float32 implementation agrees on 97% of pixels with flat surroundings
  and on 26% of speckle pixels; a 1e-6 nudge leaves 97% of flat-region pixels and 18% of speckle pixels unchanged. The reproduction of the upstream figure is statistical in the filaments for the same reason.
* **float32 only.** TF32 and bfloat16 give different pictures (21.9% and 13.6% pixel agreement with the upstream image).
* **Slim fractal.** The structure thins out under magnification (section 3); nothing was rendered beyond ×128.
* **Early exit** retires a start after 3 loops on the solution: identical settle times for 99.93% of 4,096 test pixels. Batch size matters (B=2048 is a
  different cuBLAS kernel regime and agrees on only 93.7%), so production used the validated batch of 1,024.
* **Metrics depend on the pixel pitch** (and on the labels: here settle time, as in the paper).
* **The upstream repository has no licence**, so none of its code or images is redistributed; see Credits.

## Credits and licences

* This repository's own code and our own renders (images, videos, explorer data) are released under the MIT licence
  (`LICENSE`).
* **loopscape** and the paper (Lai, Bao, Quinn, Gilpin), https://github.com/GilpinLab/loopscape. The repository carries
  no licence, so none of its code or images is redistributed here (`vendor/` is gitignored); the upstream image and the
  arrays derived pixel-by-pixel from it (frozen samples, probe arrays, the side-by-side figure) stay local and out of git.
  Aggregate statistics we computed from the upstream image are reported in the tables above.
* **History.** The git history was rewritten once before publication to remove those derived arrays, which the first
  local commits had briefly tracked. Commit hashes cited in `RUNS.md` name the original local commits;
  `docs/COMMIT-MAP.md` maps each to the commit in this repository.
* **EqR** (Huang, Geng, Kolter, arXiv:2605.21488), Apache-2.0 code and weights from locuslab
  (`locuslab/EqR`, pinned commit e9934826; `locuslab/EqR-model`).
* **Sudoku-Extreme** puzzles (HRM, Wang et al.; HF `sapientinc/sudoku-extreme`).
* FPRM (Movahedi et al., arXiv:2606.18206) is the other model loopscape wraps; it is not used here.
* Compute: ALICE, Leiden University (A100 GPUs, partition `gpu-short`).
