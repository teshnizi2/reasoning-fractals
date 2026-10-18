# STATE: reasoning-fractals (handoff note)

Last updated: 2026-10-18 (deployed, licensed, history rewritten, published; the build notes date from 2026-10-17, the first process died at ~13:50, see History).

## Goal
Fractal "mind" of a recurrent reasoning model: start EqR (Sudoku checkpoint) from a 2-D slice of initial latent "thoughts",
colour each start by how long it takes to settle, and show it: hero image, deep-zoom video, click-a-pixel explorer, metrics,
docs. Spec `BRIEF.md`, rules `../visual-series-RULES.md`. Final report <= 300 words with job IDs, GPU-hours, what each run produced.

## Permissions (read the rules file again if in doubt)
* ALICE approved by Reza (17 Oct) for exactly: dream-engine training (not mine) and the high-res fractal renders.
  `ssh alice2`, partition gpu-short, folder `~/reasoning-fractals` on the cluster only, resumable tiles, rsync back,
  one `RUNS.md` row per experiment (append-only). Full A100s and MIG `4g.40gb` slices were used (QOS cap 12 GPUs per user,
  shared with dream-engine; full-GPU tasks starved in the queue, see RUNS rows 11 and 13).
* Build rules at the time: no publishing, no `git push`. Reza approved publishing on 2026-10-18 (MIT licence, Vercel + teshnizi.dev subdomain, public repo); see "Deployed and published". Still no posting and no AI attribution in commits. POST.md is never posted.

## Status: DONE (all deliverables exist; only the verification pass and the report remain)
* **Reproduction** (RUNS 1-4): the upstream asset puzzle was identified (blake2s of the question); our fp32 A100 render agrees
  pixel-by-pixel on 64.4% (99.94% where smooth, 48% in filaments); the authors' own code agrees no better; statistics match.
* **Hero** (RUNS 11): 2048^2, jobs 5246384/5246385/5246863; `media/hero.png` (series colourway), `hero-1080x1350.png`,
  `hero-1080x1080.png`; the indigo/teal alternative is in `media/loopscape/`. Like-for-like with the upstream image (same 11
  labels, same pitch): boundary dimension 1.853 vs 1.852, basin entropy 1.139 vs 1.141.
* **Keyframes** (RUNS 13): five 540x675 layers, zoom x8..x128, centre (0.87882, 0.86937); the fractal is slim (31% of boxes
  touch a boundary at x128 vs 69% on the full map).
* **Replay** (RUNS 10, 12, 14): the cluster re-runs chosen pixels on the render engine; featured starts of the explorer and the
  video's finale pair (A: 5 loops, B: 21 loops, B in the speckle, 6-21 under 1e-6 nudges) come from there.
* **Explorer** (`site/`): 6-layer pack in `site/data` (built by `scripts/finish.sh 3`), verified with `tools/verify_site.mjs`
  (desktop, phone, featured starts, replay). **Video**: `media/zoom-1080x1350.mp4` and `zoom-1080x1080.mp4` (20 s, H.264,
  yuv420p, no audio; gitignored, rebuild with `scripts/finish.sh 6`). **Docs**: README is generated (see below), POST.md draft
  (116 words + 3 hooks), RUNS.md rows 1-14.
* Tests: `pytest` runs 211 tests, all passing (the `slow` ones run the model on the Mac), plus an end-to-end dry run on a stub render.

## How to rebuild everything (scripts/finish.sh has the stages)
1 pull + merge -> 2 metrics, scale dependence, scans, GPU hours, cost -> 3 pack + explorer copy -> 4h/4k replay plans (cluster)
-> 4b apply -> 5 hero images -> 6 video -> 7 README (strict: every number must exist in `results/`).
README.md is generated from `docs/README.template.md` by `scripts/build_readme.py`; edit the template, never README.md.

## Decisions and facts worth keeping
* **Asset identification:** puzzle `2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3...`
  (rating 47); orientation `xy_fy`; level = settle_nb - 2. The notebook's own puzzle is a different one (rating 162).
* **fp32 only** (TF32 22%, bf16 14% agreement). Early exit (3 loops on the solution) validated at batch 1024 (99.93% identical
  settle times, 2.7x faster); batch 2048 is a different cuBLAS regime, never used.
* **The fractal is slim and rounding-sensitive:** speckle pixels are not reproducible across float32 stacks (MLX vs A100: flat
  97%, speckle 26%) and a 1e-6 nudge moves them even on the same GPU (97% of flat pixels unchanged, 17-19% of speckle). That is why
  traces for the explorer and the video come from the cluster's engine (`alice/replay.py`), never from MLX, and why the video
  footnote and the explorer say so.
* **Native-scale starts** (EqR's own init, norm ~223): 90.1% solved over 5565 puzzles x 16; 664 multistable puzzles; wrong endings
  are wandering transients (213 of 213 unsolved starts still changing after 96 loops); outcome maps are noise-like, so there is
  no outcome-coloured second map. The explorer says UNSETTLED for those; STUCK is reserved for a settled wrong grid (none in the hero).
* Compositor: finer layers feather out over 20 samples and are clipped to the map; blocks (hard-edged pixels) from 4 to 10
  on-screen px per sample; the camera has a knee (layers in ~70% of the move, magnified pixels after); the finale shows a pair of
  neighbours with Sudoku replays (`render/finale.py`).
* Colourways in `render/colour.py`: `series` (slate + amber, default everywhere) and `loopscape` (indigo/teal/mint).

## Open for Reza (decisions I could not ask)
1. **Upstream licence**: RESOLVED 2026-10-18. loopscape has no licence file and nothing of theirs is redistributed. The history was rewritten
   once (`git filter-branch`, 94 commits kept) so that no file derived from their image (`data/asset_samples.npz`,
   `results/repro/probe_*.npz`, the side-by-side figure) is in any commit; `scripts/verify_deliverables.py` checks this on every run.
   Old hashes in `RUNS.md` map to the new ones in `docs/COMMIT-MAP.md`.
2. **Wrong-answer story**: at the hero's scale >99.9% of starts solve, so the post says the fractal is in HOW LONG the model
   thinks; wrong endings only appear at EqR's native start scale, as wandering, not as stuck answers.
3. **MIG slices** were used next to full A100s (queue cap); same silicon and fp32 semantics, only slower. Hero tiles therefore
   come from both; replays on MIG reproduce tiles from either (RUNS 10, 12).
4. **Zoom depth** is capped at x128 on purpose (slim fractal; a trial window at x5120 was pure speckle). The video's counter goes
   further (x3,577) only because the last seconds magnify single pixels (hard-edged blocks).
5. **Finale pair honesty**: the slow pixel B is in the speckle; its 21 loops is the A100 computation, 1e-6 nudges give 6-21.
   The video says so on screen. A stricter pair (order kept in every nudged run) exists only at x8; use
   `scripts/apply_replay.py --criterion strict --rank robust` and re-run `scripts/finish.sh 6` if you prefer that.
6. **Videos are 57 MB / 48 MB** at CRF 16 (the speckle is hard to compress); `scripts/make_video.py --crf 20` makes them about half
   that size.

## Deployed and published (2026-10-18, approved by Reza)
- Live: https://fractals.teshnizi.dev (also https://reasoning-fractals.vercel.app). Vercel project `reasoning-fractals` (team hmkhd2-1413s-projects, `prj_rCntqpBPmpy463Su6cwf1EHq0edM`), static deploy of `site/`, no build step.
- Code: https://github.com/teshnizi2/reasoning-fractals (public, MIT licence, chosen by Reza). Our renders in the loopscape colourway (`media/loopscape/`, `site/data/layer*_loopscape.webp`) are our own output, not their image.
- Redeploy after changing `site/`: `cd site && vercel deploy --prod --yes --scope hmkhd2-1413s-projects` (`site/.vercel/` is gitignored), then `git push`.
- History rewrite: done in a fresh clone, then its `.git` replaced the old one. 94 commits kept (dates and messages unchanged), the tree of the last rewritten commit is identical to the original, and the six derived blobs are gone from the object store. Backups outside the repo: `~/Random projects/_publish-backups-2026-10-18/` (bundle of the old history and the old `.git` folder; they still contain the derived files, so never publish them).
- Checked live: all 24 served files are byte-identical to `site/` (SHA-256); in the in-app browser all 6 layers loaded, `data-ready=1`, tapping a pixel gave a=+0.014167 b=-0.077675, solved, 4 loops (the same as the builder's phone screenshot), "Fly in" ran to the deepest layer, no console errors, 375 px layout without horizontal scroll.
- Share card: `site/og.jpg` (1200x630 crop of `media/hero.png`) plus Open Graph, Twitter and canonical tags in `site/index.html`.
- Hosting limits: Vercel Hobby includes 100 GB/month of Fast Data Transfer for the whole team (it also serves www.studypuff.com); sustained overage can lead to paused deployments (Vercel warns first; fix = upgrade to Pro or cut traffic). A first visit transfers about 3.6 MB here, so the allowance is not a concern for this one. If a post takes off, move the heavy files to a host with free egress or upgrade the team to Pro.

## ALICE jobs (details and GPU-hours: RUNS.md, results/alice/gpu_hours.json)
smoke 5246129; repro250 5246208_0..3; stock 5246207; early-exit check 5246309; puzzle scan 5246311; chase 5246310/5246316/5246357;
longrun+oscan 5246477; replay smoke test 5246629; first hero try 5246369/70/80 (cancelled); hero 5246384/5246385/5246863;
keyframes 5246431, 5246864-5246868, 5247171; hero replay 5247124; keyframe replay 5247635.

## History
* 11:09-11:36 first session: setup, upstream reading, helpers + tests, engine written. Process died ~13:50.
* 13:55 resumed (this note started), ALICE approved, reproduction + production plumbing, hero/keyframes submitted 15:40-15:55.
* 16:00-19:00: scans, replay pipeline, video finale, README generator, dry-run test; hero finished 19:00.
* 19:00-20:40: hero images, metrics, keyframes finished 20:14, replay, pack, video, docs.
