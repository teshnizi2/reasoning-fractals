# reasoning-fractals: the fractal "mind" of a reasoning model

Read `~/Random projects/visual-series-RULES.md` first. Those rules apply here.

## The one-line pitch
Give a recurrent reasoning model the same Sudoku a million times, changing only its initial latent
"thought". Colour each starting point by where the model ends up: solved, or which wrong answer it
got stuck in. The map is a fractal. A tiny nudge to the first thought flips the answer.

## Source
- **GilpinLab/loopscape** (created 28 Aug 2026): "Tools for probing the landscape of fractal basins
  in recurrent reasoning models". It wraps **EqR** (arXiv:2605.21488; weights at HF
  `locuslab/EqR-model`, code at `locuslab/EqR` pinned to commit e9934826) and **FPRM**
  (arXiv:2606.18206; HF `fixed-point-reasoners/fprm`), plus Parcae.
  - API: `get_solver("eqr").solve_batch(puzzle, z_H=z, z_L=z*0, max_steps=24, noise_scale=0.0,
    return_intermediates=True)`. For Sudoku, the latent shape is (97, 512).
  - `loopscape.metrics` covers basin entropy, uncertainty exponent, ordinal complexity and FLI.
    `loopscape.fractal` covers register-subspace slice construction.
  - `demos/basins.ipynb` makes basins through a random 2D slice of latent space.
  - `assets/basins_1000_0_eqr_sudoku_plane_rating47_7d68927f389fbdaa_rating47.png` is their 1000×1000
    EqR Sudoku basin. It is a gorgeous indigo/teal marbled fractal, and it is our target to
    reproduce first.
  - Note: the README's install line uses `williamgilpin/loopscape`. Check that it matches
    `GilpinLab/loopscape` and use the GilpinLab one.
- Find out whether there is a paper for loopscape itself (arXiv search: Gilpin, fractal basins,
  recurrent reasoning) and cite it if there is.

Clone the repo into `vendor/loopscape` and read the code and notebook before running anything.
Install it into the project `.venv` (uv, Python 3.12) in editable mode from `vendor/`. Use torch with
MPS where it works, and fall back to CPU per op if needed.

## Steps

### 1. Reproduce
- Run the quickstart.
- Then reproduce the asset's basin at low resolution (e.g. 200² to 300²) with the same slice, seed
  and puzzle the notebook uses.
- Show the low-res image next to the downscaled asset in `results/repro/`.
- If it doesn't match, find out why before going further.

### 2. Throughput and cost
- Measure rollouts/s on the M4 (MPS and CPU, a few batch sizes).
- Estimate the time for 1000², 2048², and a zoom sequence.
- Write `alice/` Slurm scripts for the heavy renders (A100, partition `gpu-short`,
  `--gres=gpu:a100:1`, 4 h limit, so split into tiles that resume). Do NOT submit them. Report what
  you'd run and the expected GPU-hours.

### 3. Render pipeline (`render/`)
- Per pixel, store:
  - an outcome id (each distinct final decoded grid gets an id, with 0 = solved, or keep "solved" as
    its own flag);
  - the settling time (the step at which the decoded grid stopped changing);
  - whether the final grid is a valid solution.
- Save as compact arrays (`.npy` or uint16 PNG) plus a JSON attractor dictionary (id → 81-char grid,
  count, valid?).
- Colour: a refined version of the loopscape look, i.e. deep indigo → teal → mint, with
  outcome-dependent hue and settling time as luminance. It must be beautiful at full resolution.
- Make a second colourway in the series palette (dark `#0e1117` with amber `#f59e0b` accents) and
  choose the better one.
- Write metrics to `results/metrics.json`: fraction solved, number of distinct attractors, basin
  entropy, uncertainty exponent / boundary dimension (using loopscape.metrics; state the method).

### 4. Deep zoom
- Pick a boundary point with rich structure.
- Compute keyframes at zoom 2^k (as deep as the structure stays interesting, or as local compute
  allows; deeper keyframes can come from the ALICE run later).
- Build a smooth zoom by scaling and crossfading between keyframes.
- Output a ~15–20 s `media/zoom-1080x1350.mp4` and a `media/hero.png`.
- With local compute only, deliver the best honest version (e.g. 512² keyframes, fewer levels), and
  make the pipeline re-runnable with ALICE outputs dropped in.

### 5. Explorer page (`site/`; invoke `frontend-design` first)
- Full-bleed basin image with smooth pan/zoom across the precomputed zoom levels.
- Hover/click a pixel to open a panel showing:
  - the 9×9 grid the model converged to from that starting thought;
  - givens styled differently from the filled cells;
  - wrong cells highlighted against the true solution;
  - a SOLVED / STUCK badge;
  - the settling steps.
- A tiny "what am I looking at" explainer: what a latent initial condition is, what a basin is, why
  fractal boundaries mean sensitivity.
- A credits line (loopscape, EqR/FPRM authors).
- Phone width: the image fits, and the panel becomes a bottom sheet.

### 6. Docs
`README.md` (repro evidence, metrics, cost, limits, credits), `STATE.md`, `POST.md`.

## Done means
- The asset's structure is reproduced (or the mismatch is explained).
- Metrics are computed.
- Hero image, zoom video and explorer work locally.
- ALICE scripts are ready for the high-res version, with a cost estimate.
