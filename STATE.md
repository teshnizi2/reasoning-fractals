# STATE: reasoning-fractals (handoff note)

Last updated: 2026-10-17 (session resumed after the previous process died; see "History").

## Goal
Fractal "mind" of a recurrent reasoning model: start EqR (Sudoku checkpoint) from a 2-D slice of
initial latent "thoughts", colour each start by where it ends up and how long it took. Deliver:
hero image, deep-zoom video, click-a-pixel explorer. Spec: `BRIEF.md`. Rules: `../visual-series-RULES.md`.
Final report is <= 300 words and must include ALICE job IDs, GPU-hours used, what each run produced.

## Permissions (read the rules file again if in doubt)
- ALICE is approved ONLY for the high-res fractal renders (and dream-engine, not mine). Reza, 17 Oct:
  "Yes, you can submit the long ALICE runs for the dream-engine training and the high-res fractal renders."
  Use `ssh alice2`, partition gpu-short, `--gres=gpu:a100:1`, folder `~/reasoning-fractals` on the cluster only,
  smoke job first, resumable tiles (4 h cap), rsync results back, one `RUNS.md` row per experiment
  (template `~/.claude/templates/RUNS.md`). No publishing, no `git push`, no AI attribution in commits.
- Local MPS runs: <= ~20 min each, never two heavy GPU jobs at once, background + log file.

## Upstream facts (verified this session)
- Repo: `GilpinLab/loopscape` (cloned to `vendor/loopscape`, commit fff8bf7). README install line says
  `williamgilpin/loopscape`: that repo returns 404, so use GilpinLab. **No licence file** (GitHub licence = null):
  do not redistribute their code or their asset image; the side-by-side in `results/repro/` is local evidence only.
- Paper for loopscape exists: **"Fractal basins trap latent reasoning"**, Jeffrey Lai, Anthony Bao, John Quinn,
  William Gilpin, arXiv:2609.04963 (4 Sep 2026, cs.LG). EqR = Huang, Geng, Kolter, arXiv:2605.21488
  ("Equilibrium Reasoners", ICML 2026). FPRM = Movahedi et al., arXiv:2606.18206.
- EqR source pinned at e9934826 (Apache-2.0) in `vendor/EqR`, read before running. Weights
  `locuslab/EqR-model/sudoku-extreme/eqr.pth` (40,254,748 B) in `data/raw/loopscape_cache/eqr_sudoku.pth`,
  sha256 258b1f0a... matches the HF LFS oid; pickle globals scanned (only torch/omegaconf/collections).
- Paper method (App. B): EqR noise off, 24 loops, deterministic map; plane z = a*u + b*v from QR of a Gaussian
  matrix; settle time = loops until decoded grid stops changing; basin entropy (box 5) and uncertainty
  exponent (eps = 1..5 px) on settle-time labels at 200x200. NOTE the paper text says z0 is a random Gaussian
  base point, but the notebook cell uses z0 = 0, z_L = 0, (a, b) in [-1, 1]^2 with unit-norm u, v.
  The paper's slices keep >= 90% of starts on the same final solution: the fractal is mostly in SETTLE TIME.
- Asset `assets/basins_1000_0_eqr_sudoku_plane_rating47_<hash>_rating47.png`: 1000x1000, exactly 11 unique
  colours = an 11-level settle-time field. Notebook puzzle `.....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2....`
  is the candidate (its rating and the asset hash 7d68927f389fbdaa are NOT confirmed; md5/sha1/sha256 of
  the puzzle string do not match the hash).

## Reproduction findings so far (17 Oct, local, MLX engine)
- The notebook's EqR puzzle is Sudoku-Extreme test puzzle rating **162** (source puzzles4_forum_hardest_1905),
  NOT the asset's rating-47 puzzle. Probe (notebook puzzle, plane seed 0, h=1, z_L=0, K=24 grid on the asset's
  grid, 576 px): 100% solved, 1 distinct final grid, settle_nb 1..11 (mean ~3.4), best pixel agreement with the
  asset only 21% (chance ~17%) => as expected, different puzzle.
- Asset puzzle IDENTIFIED: `blake2s(question, digest_size=8).hexdigest()` == 7d68927f389fbdaa for
  `2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3...` (rating 47, 422,786-row
  test.csv from HF sapientinc/sudoku-extreme, saved in data/raw/sudoku-extreme/).
- **Probe on the asset puzzle (notebook parametrisation: z_H = a*u + b*v, z_L = 0, (a,b) in [-1,1]^2, plane seed 0,
  24 loops, noise 0) on 196 pixels of the asset's own grid: orientation `xy_fy` (asset row 0 = y = +1, i.e.
  imshow origin='lower'), asset level = settle_nb - 2 (levels 0..10 = T 2..12, top level saturated), pixel
  agreement 67%, Spearman 0.92 (all other 7 orientations: agreement 13-24%, |rho| < 0.27).** Contingency: the
  smooth low-T regions match 100% (levels 0-1: 42/42; level 2: 47/51), mismatches sit in the speckled
  high-T fractal dust, where outcomes are sensitive to floating-point details (MLX vs the authors' stack).
  Files: results/repro/probe_r47_k14.{json,raw.npz,npz}. (Notebook puzzle probe, 576 px: results/repro/probe_k24.*)
- ALICE: env built (torch 2.14.1+cu130, ~/reasoning-fractals/env), code pushed, smoke job 5246129 submitted.

## Compute facts (measured)
- One 24-step rollout = 21 L-level calls/step x 2 layers; ~0.615 TFLOP fp32 per rollout.
- Stock loopscape path: ~1 rollout/s on MPS (B=16..128), 0.53/s on CPU (10 cores). GPU is shared: fp32 matmul
  4096^3 gives only ~0.55 TFLOPS on MPS, ~0.86 on MLX (M4 peak ~3.7), device utilisation spikes to 94% from
  other builders. MLX matmuls for the model's shapes: 0.74-0.98 TFLOPS (torch-MPS 0.30-0.92).
- So local budget is ~1-2 rollouts/s: 20 min ~ 1.2-2.4k pixels. Heavy renders go to ALICE (A100).
- fp32 limits zoom depth: neighbouring pixel latents collapse below ~2^-20 relative spacing.

## Code (TDD; tests in `tests/`, run `.venv/bin/python -m pytest`)
- `render/sudoku.py` (solver, validity, packing): done, 9 tests pass.
- `render/slices.py` (notebook-exact plane/mesh/latents, zoom windows): done, 11 tests pass (bitwise vs notebook).
- `render/outcomes.py` (settle times, outcome flags, attractor dictionary): done, 12 tests pass.
- `render/engine.py` (MLX + torch rollouts): written; `tests/test_engine.py` run was KILLED (exit 137) when the
  previous process died. Re-run it in the background with a log before trusting the engine.

## Decisions so far
- Pixel-level reproduction test: sample the asset's own grid (`torch.linspace(-1, 1, 1000)`) at a random
  subset and a stride grid, compare settle time via colour rank (monotone map), plus metrics on identical points.
- Keep both settle-time definitions (`settle_nb` = notebook, `settle_last`). Outcome id 0 = the true solution.
- Wrong-answer pixels get amber/rose accents in the series palette; most pixels may still end solved.

## Open for Reza
1. Upstream has no licence: the repro side-by-side shows their asset (downscaled). Remove it or ask the authors
   before any public use. Nothing from their repo is copied into `media/` or `site/`.
2. If the asset slice turns out to be ~100% "solved", the post must say the fractal is in HOW LONG it thinks
   (and in which answers appear at larger initial-thought scales), not in whether it solves.

## Next steps
1. Re-run engine tests (background, log) and fix; commit.
2. Reproduction: sparse pixel check + aligned low-res grid + metrics vs the asset -> `results/repro/`.
3. Throughput benchmark -> `results/throughput.json`; ALICE scripts in `alice/`; smoke job; tiled renders; RUNS.md rows.
4. Colourways, hero, zoom keyframes (ALICE), video, explorer (`frontend-design` first), README/POST, verification.

## History
- 11:09-11:36 first session: setup, upstream reading, helpers + tests, engine written.
- 13:5x resumed after the process was killed mid-test; coordinator reported ALICE approval; rules file re-read
  (mtime 13:54) and confirms it.
