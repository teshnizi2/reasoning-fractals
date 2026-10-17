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

## ALICE results so far (A100 80GB PCIe, torch 2.14.1+cu130; job ids in RUNS.md)
- Smoke job 5246129 (RUNS row 1): eager fp32 23.0 rollouts/s, tf32 ~60, bf16 ~90 (B 256-1024; compile tf32 83).
- **fp32 only**: on 4096 random pixels of the asset grid, fp32 agrees with the upstream asset on 65.0% of pixels
  (99.9% in the 33% of pixels where the asset is locally flat, 48.1% in the filaments; Spearman 0.90) and
  reproduces its level histogram (e.g. 971 vs 971 pixels at level 1); tf32 gives 21.9% (interior 42%), bf16 13.6%
  (rho ~ 0): reduced precision is a different realisation even in flat regions. All deliverable renders use fp32.
- **Outcome is almost never wrong**: 48x48 scale scan (h = 1, 4, 16, 64, 256; tf32 so indicative only) gives >= 99.7%
  solved everywhere, 1-2 distinct wrong grids; the few unsolved pixels are the ones still changing at loop 24
  (censored), not stable wrong attractors. So the fractal is in SETTLE TIME, as in the paper. The "which wrong
  answer" colouring needs a puzzle where the model is genuinely torn: scan job 5246238 looks for one (5565 puzzles
  x 16 native random starts).
- **Early exit validated on a 1024-pixel pilot**: retire a pixel once its decoded grid equals the true solution for 3
  loops in a row; 0 mispredictions, mean 8.8 loops instead of 24 (63% saved). (Latent residuals converge too slowly
  to be useful.) Implemented as Engine.rollout_early (compaction); to be validated at scale vs full runs.
- Jobs in flight: 5246208 (repro250 array 0-3, fp32, asset grid stride 4), 5246207 (stock loopscape path on the same
  pixels), 5246238 (puzzle scan).

## Reproduction verdict (RUNS row 2, job 5246208)
Asset puzzle + notebook slice (z_H = a*u + b*v, z_L = 0, plane seed 0, h = 1), fp32, on the asset's own 250x250 grid
(every 4th pixel, orientation y-up): 64.4% of 62,500 pixels identical; **99.94% identical where the upstream image is
locally flat (33% of pixels)**, 48.1% in the filaments; Spearman 0.889; basin entropy 1.416 vs 1.412, S_bb 1.557 vs
1.552, boundary dimension 1.873 vs 1.872; level histogram within ~1%. Verdict: structure and statistics reproduced;
pixel-exact only where outcomes are numerically stable (the filaments are float32-order sensitive, see stock check).
Figure: results/repro/repro250_side_by_side.png.

## ALICE capacity note
Full A100s were saturated (dream-engine training + other users: scheduler start estimates 40-60 min apart), so
production jobs after the first bursts request A100 **MIG slices** (`--gres=gpu:4g.40gb:1`, A100 silicon, gpu-short)
instead of `gpu:a100:1`. Same architecture and fp32 semantics; slower per slice (~4/7 of an A100). Mentioned in the report.

## Production plan and capacity (17 Oct ~15:45)
- Early exit validated (RUNS row 4): 99.93% identical settle times at batch 1024, 2.7x faster. Production batch = 1024
  (B=2048 is a different cuBLAS kernel regime: still valid, but not the validated setting).
- QOS cap: 12 GPUs per user on gpu-short (shared with the dream-engine builder, who holds ~5): I run <= ~7 slots.
  Per-slot speed: full A100 62 rollouts/s with early exit, A100 MIG 4g.40gb 33, so full A100s are preferred.
- Hero 2048^2 (spec alice/specs/hero.json, notebook mode, 256 tiles of 128^2): jobs 5246384 (workers 0-23, a100)
  and 5246385 (workers 24-47, 4g.40gb), WORKERS=48. Keyframes (k=3..12, 540x675, 20 tiles each) wait for the zoom
  chase (job 5246357, started from a hand-picked point on the core of the top-right 'eye', c = (0.87882, 0.86937)).

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

## More decisions / open items (added 17 Oct afternoon)
- Explorer built (site/index.html, style.css, app.js) against a throwaway mock pack (site/data-mock, gitignored);
  real pack comes from `render/buildpack.py` once the hero/keyframes exist. Featured starts with loop-by-loop replay
  traces are planned (the UI is done; traces will be computed locally with MLX and kept only if they reproduce the
  stored settle time).
- Early exit (solution plateau, m=3) is the production default if job 5246309 (RUNS pending) shows it changes the
  field no more than a batch-size change does; otherwise production shrinks (hero 1536^2, fewer keyframes).
- Series-palette colourway retuned (slate domains, amber/gold dust, `render/colour.py`); the indigo-teal-mint
  colourway is closer to the upstream look. Choice of the better one is made on the full-res hero.

## Finding: the fractal is scale dependent (17 Oct 15:50, chase windows of the asset slice, loopscape colours)
Zoom 10 (window around the top-right "eye"): boundary-pixel fraction 0.72, 21 settle levels. Zoom 80 (core edge):
0.34, 20 levels; a big flat domain on one side, salt-and-pepper dust on the other, no new coherent filaments.
This looks like a "slim fractal" (paper App. E): complexity strongest near the 1e-3 pitch of the hero and thinning
when zoomed. Zoom 640 and 5120 windows come from job 5246357; keyframes (k >= 7) are held back until they show
whether anything coherent survives. Fallback video: hero-only zoom to ~x64 (one pixel = one thought), no extra cost.

## Keyframes (submitted 15:55)
Because the field thins out when zoomed (see above), keyframes cover only zoom 8..128 (k = 3..7), 540x675 each
(4:5, square pixels), centred at c = (0.87882, 0.86937) (hand-picked left tip of the dark core of the top-right
"eye"; the first automatic chase picked a flat corner because its score saturated, hence the smooth scores + anchor
in render/chase.py). Jobs 5246430-5246439: per level an array of 10 tasks on full A100s and 10 on 4g.40gb slices,
WORKERS=20, one 135x135 tile per task. More levels (k = 8, 9) only if chase windows at x640 still show structure.

## Findings added 17 Oct ~16:30 (read before writing README / POST)
- **Cross-engine reproducibility depends on local roughness** (local MLX replay of random chase-window pixels vs the stored A100
  fp32 values; throwaway study, repo script `scripts/replay_agreement.py` re-creates it -> results/replay_agreement.json):
  roughness 0 (all 4 neighbours equal): 98-100% equal; roughness 1: 48-67%; roughness 2-4: 23-40%; roughness >= 5: 14%;
  stored settle >= 8 in dust: 0 of 13 reproduced. Nudging the latent by relative 1e-6 (a few dozen ulps) also moves dust
  pixels' settle times by several loops. So the speckled "dust" is rounding-sensitive (chaotic); flat domains and borders
  between big flat domains are not. Statistics of the dust reproduce, single dust pixels do not.
- **Native-scale starts (puzzle scan, job 5246311, RUNS row pending):** 5565 puzzles x 16 EqR-style random starts (norm ~223):
  90.07% solved, 4281 puzzles always solved, 664 puzzles with 15-85% solved (multistable). Long-run check (job 5246477,
  400 unsolved starts, 96 loops): 46.75% reach the solution by loop 96, the remaining 213 are all STILL CHANGING (0 frozen):
  the "wrong" outcomes are wandering transients, not fixed wrong grids -> the badge word STUCK is wrong for them (UNSETTLED).
  The asset puzzle itself is >= 99.8% solved at native-scale base points (oscan rows for asset: 8 windows).
- Chase #2 (job 5246357): x80 boundary 0.34, x640 0.30, x5120 0.48 but at x5120 it is salt-and-pepper next to a flat domain;
  nothing coherent below ~x128 => keyframes stop at k=7 (x128). No k=8,9.
- Zoom compositor had a half-sample offset for magnified layers (fixed 17 Oct, tests); blocks are drawn hard-edged once a
  sample is >3-8 screen px ("one pixel = one thought").
- **Video finale** = two neighbouring starts of the deepest keyframe as big blocks with their Sudoku replays beneath
  (`render/finale.py`, `scripts/make_video.py --pair`). Pair must come from the CUDA engine's own replay because MLX does not
  reproduce dust pixels: pipeline = `scripts/plan_replay.py` -> rsync `results/replay/replay_in.json` to the cluster ->
  `sbatch alice/replay.slurm` -> `alice/sync.sh pull replay` -> `scripts/apply_replay.py` (featured starts into the pack
  manifest + results/video/pair.json). Pair choice: both pixels reproduced by the plain replay, monotone traces, slow/fast
  ordering with gap >= 3 loops surviving every nudged re-run.
- Explorer drawer text no longer claims "new filaments at every scale"; inspector says flat vs speckled (rounding-sensitive).

## Next steps
1. Re-run engine tests (background, log) and fix; commit.
2. Reproduction: sparse pixel check + aligned low-res grid + metrics vs the asset -> `results/repro/`.
3. Throughput benchmark -> `results/throughput.json`; ALICE scripts in `alice/`; smoke job; tiled renders; RUNS.md rows.
4. Colourways, hero, zoom keyframes (ALICE), video, explorer (`frontend-design` first), README/POST, verification.

## History
- 11:09-11:36 first session: setup, upstream reading, helpers + tests, engine written.
- 13:5x resumed after the process was killed mid-test; coordinator reported ALICE approval; rules file re-read
  (mtime 13:54) and confirms it.
