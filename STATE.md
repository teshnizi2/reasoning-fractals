# STATE: reasoning-fractals (handoff note)

Last updated: 2026-10-17 ~17:00 (second session; the first process died at ~13:50, see History).

## Goal
Fractal "mind" of a recurrent reasoning model: start EqR (Sudoku checkpoint) from a 2-D slice of initial latent "thoughts",
colour each start by where it ends up and how long it took. Deliver: hero image, deep-zoom video, click-a-pixel explorer,
metrics, docs. Spec `BRIEF.md`, rules `../visual-series-RULES.md`. Final report <= 300 words with ALICE job ids, GPU-hours,
what each run produced.

## Permissions (read the rules file again if in doubt)
* ALICE approved by Reza (17 Oct) for exactly: dream-engine training (not mine) and the high-res fractal renders.
  `ssh alice2`, partition gpu-short, folder `~/reasoning-fractals` on the cluster only, resumable tiles, rsync back,
  one `RUNS.md` row per experiment (append-only). Production uses full A100s and MIG `4g.40gb` slices (QOS cap 12 GPUs
  per user, shared with dream-engine).
* No publishing, no `git push` (no remote is configured), no posting, no AI attribution in commits. POST.md is never posted.
* Local GPU runs <= ~20 min, never two heavy jobs at once.

## Where things stand (17 Oct ~17:00)
* **Reproduction DONE** (RUNS rows 1-4): the upstream asset puzzle was identified (blake2s hash of the question), our fp32
  A100 render agrees pixel-by-pixel on 64.4% (99.94% where smooth, 48% in filaments), statistics match; authors' own code
  agrees no better. Details in README section 1.
* **Hero** (2048^2, notebook mode, `alice/specs/hero.json`): jobs 5246384 (workers 0-23, A100) + 5246385 (workers 24-47,
  MIG 4g), WORKERS=48, 256 tiles; ~90/256 done at 16:52, ETA ~19:00. Pull: `scripts/finish.sh 1`.
* **Keyframes k=3..7** (zoom 8..128, 540x675, centre c = (0.87882, 0.86937) = left tip of the dark core of the top-right
  vortex): jobs 5246430-5246439 (array of 20 tasks per level; 10 A100 + 10 MIG), queued behind the hero.
* Local pipeline is built and tested (177 fast tests + an end-to-end dry run on a stub render): merge -> metrics -> pack ->
  replay plan -> cluster replay -> featured + finale pair -> hero images -> video -> README.
  **Run it with `scripts/finish.sh 1..7`** (stages documented in that file).

## What is left (in this order)
1. Wait for hero + keyframes (monitor tile counts; do not resubmit running arrays; never mix WORKERS values).
2. `scripts/finish.sh 1` (pull + merge), `2` (metrics, scale dependence, scans, gpu hours, cost, replay agreement),
   look at the full hero in both colourways (series = dark slate + amber looked like gold-leaf marbling on the partial
   render; loopscape = indigo/teal/mint), choose, `3` (pack + explorer copy), `4` (replay plan, push `replay_in.json`,
   `sbatch alice/replay.slurm`, pull, `4b` apply -> featured starts + `results/video/pair.json`), `5` (hero images),
   `6` (video, check stills at full res first with `--stills`), run `tools/verify_site.mjs` on the real pack,
   `7` (README), write POST.md (humanizer skill; <= 120 words + 3 hooks), RUNS rows for hero/keyframes/replay,
   `scripts/verify_deliverables.py`, then `verification-before-completion` and the final report.
2b. Remove throwaway packs when done: `site/data-mock`, `site/data-preview`, `site/data-big`, `site/data-proto` (all gitignored).
3. Delete `data/cluster/chase2_l2` etc. if disk matters (gitignored).

## Decisions and facts worth keeping
* **Asset identification:** puzzle `2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3...`
  (rating 47); orientation `xy_fy`; level = settle_nb - 2. The notebook's own puzzle is a different one (rating 162).
* **fp32 only** (TF32 22%, bf16 14% agreement). Early exit (3 loops on the solution) is validated at batch 1024
  (99.93% identical settle times, 2.7x faster); batch 2048 is a different cuBLAS regime, never used.
* **The fractal is slim and rounding-sensitive** (see README sections 3-4): complexity thins under magnification, nothing
  coherent below ~x128 (chase job 5246357 to x5120), speckle pixels are not reproducible across float32 stacks (MLX replay
  of random chase/hero pixels: flat surroundings 98-100%, speckle 14%, slow speckle 0 of 13). Hence the video/explorer
  traces come from the cluster's own engine (`alice/replay.py`), not from MLX.
* **Native-scale starts** (EqR's own init, norm ~223): 90.1% solved over 5565 puzzles x 16; 664 multistable puzzles; the
  wrong endings are wandering transients (213 of 213 still changing after 96 loops), outcome maps are noise-like; so there
  is NO outcome-coloured second map. The explorer calls them UNSETTLED; STUCK only for a settled wrong grid.
* Compositor facts: half-sample offset bug fixed; finer layers feather out over 20 samples; blocks (hard-edged pixels)
  from 4 to 10 on-screen px per sample; finale = pair of neighbours with Sudoku replays (`render/finale.py`).
* Colourways in `render/colour.py`: `loopscape` (indigo-teal-mint, close to upstream) and `series` (slate + amber).

## Open for Reza (decisions I could not ask)
1. **Upstream licence**: loopscape has no licence file. Nothing of theirs is redistributed; files derived from their image
   (`data/asset_samples.npz`, `results/repro/probe_*.npz`, the side-by-side figure) are untracked and gitignored, **but
   earlier local commits still contain them** (before 16:26 on 17 Oct). Before any publication, rewrite or squash the git
   history, or ask the authors. No remote exists, nothing was pushed.
2. **Wrong-answer story**: at the hero's scale >99% of starts solve, so the post should say the fractal is in HOW LONG the
   model thinks (and that wrong endings only appear at EqR's native start scale, as wandering, not as stuck answers).
3. **MIG slices** were used next to full A100s (queue cap); same silicon, fp32 semantics identical, only slower.
4. **Zoom depth** capped at x128 on purpose (slim fractal); the video counter goes further only because the last block
   magnification shows single pixels.
5. If a cleaner finale pair is wanted, `scripts/plan_replay.py --finale-per-layer N` and `scripts/apply_replay.py --min-robust-gap`
   control the search.

## ALICE jobs (details and GPU-hours: RUNS.md, results/alice/gpu_hours.json)
smoke 5246129; repro250 5246208_0..3; stock 5246207; early-exit check 5246309; puzzle scan 5246311; chase 5246310/5246316/5246357;
longrun+oscan 5246477; replay smoke test 5246629; first hero try 5246369/70/80 (cancelled); hero 5246384/5246385;
keyframes 5246430-5246439.

## History
* 11:09-11:36 first session: setup, upstream reading, helpers + tests, engine written. Process died ~13:50.
* 13:55 resumed (this note started), ALICE approved, reproduction + production plumbing, hero/keyframes submitted 15:40-15:55.
* 16:00-17:00: scans, replay pipeline, video finale, README generator, dry-run test.
