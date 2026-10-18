# Commit map: original local history to published history

The history of this repository was rewritten once before publication (2026-10-18) to remove six files derived from
the upstream loopscape image, which has no licence (`data/asset_samples.npz`, `results/repro/probe_*.npz`,
`results/repro/repro250_side_by_side.png`). Nothing else changed: every commit keeps its author, date, message and
all other files, and the tree of the last rewritten commit (`fd81dc5`) is identical to the original final tree.

`RUNS.md` cites commits by their original short hash. Find the original hash in the left column to get the commit in
this repository (right column).

| original | published | subject |
|---|---|---|
| `7d5427b` | `7d5427b` | Scaffold reasoning-fractals: sudoku/slices/outcomes helpers with tests, engine draft, state note |
| `5264ba1` | `f0d68e7` | Reproduction helpers, runner, tiled resumable renderer, torchfn backend, cluster scripts; asset puzzle identified |
| `a99f763` | `2cea81b` | Asset puzzle probe: orientation and level offset identified; RUNS.md; asset sample freezer |
| `1bf64fa` | `e6f0f0d` | Early-exit rollout, tiled renderer with rectangular windows, metrics, colourways, pack builder, explorer skeleton, ALICE smoke results |
| `55cfc7b` | `33f6aaa` | Puzzle multistability scan, mixed-puzzle batches, STATE update |
| `ed74b8c` | `99396c3` | Pack builder glue, chase scoring, early exit through the runner, explorer mobile peek |
| `f97d43f` | `5ef6ec9` | Reproduction on the asset's own grid: 250x250 fp32 render, report, figure, RUNS row |
| `3feaebc` | `5f04493` | STATE: reproduction verdict, MIG note; series colourway retune |
| `2d5564b` | `d64e02d` | Outcome scan, base-point slices, notebook-mode hero spec, tests |
| `6b54782` | `b36c3cb` | GPU-hours accounting, README draft, featured replay UI, STATE |
| `d397e7a` | `8f2f186` | Stock-path check results and RUNS row |
| `a1d9211` | `da4d1f8` | Keep raw cluster pulls out of git; keep the reproduction field under results/repro |
| `e95cc92` | `8b220e1` | Featured-start selection, metrics script, status helper |
| `507af10` | `e80f31a` | Chase: smooth non-saturating scores with anchor; hand-picked restart |
| `9c7626f` | `dfd317c` | site_text generator; keyframes start at k=3 (k=2 would only match the hero's pitch) |
| `afdd661` | `5de20b1` | Early-exit validation (RUNS row 4); production plan |
| `ce3200f` | `dcda790` | RUNS correction row |
| `7572cb3` | `6e00918` | Adjacent-gap statistic, smooth-pixel toggle |
| `3542a2e` | `323fa03` | Fix adjacent-gap test arithmetic |
| `8284d85` | `40ebf5a` | How-it-works diagram, preview tool, quickstart script |
| `19cd6b9` | `178107b` | How-it-works diagram, preview tool |
| `5644558` | `06299dc` | Local M4 throughput results and cost estimates |
| `43060f8` | `5230586` | alice/README capacity notes |
| `693c6bc` | `38f0672` | STATE: scale dependence finding |
| `a9244ff` | `02cb2d3` | oscan contact-sheet viewer |
| `46289ed` | `3ce0b84` | README: how the picture is made |
| `491eeb0` | `224bd3b` | Keyframes k=3..7 submitted; STATE |
| `916432d` | `dde204c` | Browser verification (puppeteer-core + Edge), favicon |
| `c214a8a` | `ce84b90` | Respect the hidden attribute everywhere |
| `f6f69f2` | `5c6b5b3` | Explorer copy: scale-dependence sentence |
| `f29f066` | `564b366` | results/README index |
| `44e5cd7` | `a37d483` | Sudoku inset renderer for the video |
| `235c048` | `36190fa` | Zoom compositor: fix half-sample offset of magnified layers, hard-edged blocks at large zoom, screen projection helpers |
| `971c170` | `5adfa16` | Alignment test for stacked layers |
| `78595d4` | `039f178` | Video finale (pair of neighbouring starts + Sudoku replays), caption pills, pair picking, pack builder CLI |
| `b9046c4` | `3cc012d` | Replay pipeline (library, planner, cluster job), explorer: rounding-sensitivity hint, honest fractal wording |
| `9c4a098` | `9fe9084` | Replay planner/apply scripts |
| `d3f65bf` | `c14989d` | STATE: findings (replay agreement, native-scale scan, chase), video finale plan |
| `9693ebf` | `4bf51ee` | Stop tracking files derived from the upstream image (no licence); keep them locally |
| `630d788` | `5760680` | README generator: numbers and tables from results/, template, scan summaries |
| `8efa698` | `4ace757` | README template with results sections and honest limits |
| `a2c1de3` | `74170cb` | Scan summaries (pscan, longrun, oscan), oscan contact sheet |
| `dee61be` | `057f7d1` | README: oscan row wording |
| `1a2c50a` | `76019f1` | RUNS rows 5-9 |
| `3b6aea4` | `4ee3eb9` | Feather finer layers at their footprint edge |
| `a259e95` | `34fa9b2` | End-to-end dry run of the finishing pipeline on a stub render; drop the MLX-only featured builder |
| `d4f4634` | `40ec7bd` | finish.sh: the finishing sequence |
| `8341aa9` | `eb73fb1` | cost estimate: 5 keyframes |
| `bd9eff2` | `46d4e45` | Explorer copy: unsettled vs stuck counts, trial-window depth, reproducibility meta |
| `d55588b` | `f8db716` | plan_replay defaults: 60 pairs per layer, 5 nudged runs |
| `c9ecfd3` | `6a00f51` | Finale: bigger Sudoku cards, loop label in the card header, outlined connectors |
| `83979bf` | `ca502f0` | site_text tests follow the unsettled/stuck wording |
| `7b4eb8a` | `68f6b9b` | alice/README: extras and replay jobs; blocks start at 4 px per sample |
| `7815fd8` | `6820466` | Deliverables verification script |
| `abb55a8` | `e75f97c` | README template: test-pixel count from results |
| `a00fa82` | `2bc6bc3` | STATE rewritten for handoff |
| `41ec713` | `a2c3a52` | results/README index additions |
| `4e4f92d` | `02e7213` | verify_site: featured starts and replay |
| `4b0dc7a` | `d28a235` | Replay: slow cap, roughness strata and nudge statistics into summary, README section 4 |
| `fc5d28a` | `35e7913` | Replay smoke test summary and RUNS row 10 |
| `c666b40` | `42cd656` | Video: kicker line, captions lowered |
| `811ff49` | `8419767` | Explorer default colourway: series; neutral legend wording |
| `0583b9a` | `c4d6318` | README template: explorer section |
| `27341a4` | `d3d3ef5` | Remove the MLX pair builder (superseded by the cluster replay) |
| `0951abd` | `66c0a58` | make_video: show the pair's own layer as the finest |
| `1c4bf33` | `0f75a70` | Fallback captions for a video without the pair finale |
| `cd03981` | `57bd0f0` | Replay in two runs (hero as soon as it is done, keyframes later); dry run follows |
| `26e2087` | `2934daa` | finish.sh: stage 1 for a subset, metrics skip unmerged keyframes |
| `5d78d40` | `c36a5dd` | Hero merged: metrics (also on the upstream's 11 labels), scale dependence with like-for-like labels, hero images (series + loopscape) |
| `183314a` | `41e9a87` | gpu_hours: query sacct by explicit job ids; lossless webp layers |
| `606fb80` | `d54392e` | Explorer: inspector below the readout; unsettled note |
| `7a77f8e` | `47561cf` | README tables: pitch per stride, ordinals, like-for-like wording |
| `373afd9` | `7878694` | tilerun --backend override, demo16 spec, README quickstart and colour wording |
| `eddb86a` | `53faa7e` | README: native norm, slow-pixel reproducibility sentence |
| `c87d9f2` | `0d69b78` | replay agreement results; data/local ignored |
| `4760ba4` | `fe8e827` | RUNS rows 11-12 (hero, hero replay) |
| `55145de` | `642ef75` | POST.md draft |
| `bdfd48e` | `6192600` | Finale pick: median fallback criterion; POST.md trimmed to 116 words |
| `ce8179c` | `c1e8bb8` | Video finale footnote: the unstable pixel's nudged range (4:5 cut) |
| `92d2cd6` | `a4778f9` | Keyframe metrics, scale dependence rows and README section 3 |
| `9118830` | `fd210e4` | Explorer copy: measured numbers for the slice, speckle and zoom sentences |
| `9a5dc03` | `fe782b4` | Sudoku card: given cells are opaque (they were see-through); camera knee; kicker pill |
| `002ffec` | `d3739b7` | RUNS rows 13-14 (keyframes, keyframe replay); explorer legend/credit backdrops |
| `1ed8fa5` | `2f6e9ef` | Video: earlier pan, map edge hairline and clipping, camera knee |
| `4b711db` | `b7caf3e` | README cost wording |
| `3afc334` | `fcb06f9` | Explorer: single-row chips on phones; screenshots for the README |
| `f01121a` | `cf5292a` | Untrack the videos (rebuilt by scripts/finish.sh 6) |
| `137e5ac` | `fae8149` | Video: hard layer edges in the block regime, replay speed fitted to the finale |
| `6ba53d4` | `5336627` | README: video section, repro_report inputs |
| `d46bf75` | `c56e48d` | README built from results (strict); throwaway packs removed |
| `8f8da9f` | `415b2af` | RUNS row 13b: job id per keyframe level |
| `5155df5` | `7b72cac` | STATE: final numbers |
| `cb89458` | `83d590a` | Explorer: show a dash instead of NaN when the viewport has no size yet |
| `dc76091` | `fd81dc5` | POST: tighter draft, source for the 5-vs-21 pair |
