# results/

Every number quoted in `../README.md`, `../POST.md` and the explorer comes from one of these files. Each row names the
script that wrote it and the `../RUNS.md` row (ALICE jobs) or local run behind it.

| file | what | script | run |
|---|---|---|---|
| `repro/repro250_summary.json`, `repro/repro250_side_by_side.png` | 250x250 fp32 render on the upstream image's own grid vs the upstream image: pixel agreement, contingency table, basin entropy, uncertainty exponent, histograms | `scripts/repro_report.py` | RUNS 2 (job 5246208) |
| `repro/repro250_field.npz`, `_dictionary.json`, `_spec.json` | the rendered field itself (settle time, flags, outcome ids) | `alice/specs/repro250.json`, `render/tilerun.py` | RUNS 2 |
| `repro/probe_*.json` | first local probes on 196 / 576 pixels of the asset grid (MLX backend, Apple M4): orientation and level offset were identified here; the notebook's own puzzle (rating 162) does not match, the asset's (rating 47) does | `scripts/repro_probe.py` | local |
| `alice/smoke_5246129_bench.json` | A100 throughput (fp32/tf32/bf16), agreement of each precision with the upstream image, scale scan (tf32, indicative only), pipeline smoke test | `alice/bench_cuda.py` | RUNS 1 |
| `alice/stock_5246207.json` | loopscape's own stock solver on the same 4096 pixels, three numeric configurations | `alice/stock_check.py` | RUNS 3 |
| `alice/eechk_5246309.json` | early exit vs full runs, batch-size invariance | `alice/earlyexit_check.py` | RUNS 4 |
| `alice/gpu_hours.json` | GPU-hours of every `rf-*` job from `sacct` (MIG slices counted as 4/7 or 3/7 of an A100) | `scripts/gpu_hours.py` | all |
| `throughput_m4.json` | rollouts/s on this Apple M4 (MLX, torch MPS, torch CPU, loopscape stock) | `scripts/bench_local.py` | local |
| `cost_estimate.json` | time to render 1000x1000, 2048x2048 and a zoom sequence on each backend, from the measured rates | `scripts/cost_estimate.py` | derived |
| `quickstart.json` | the loopscape README quickstart (64 random latents, CPU) | `scripts/quickstart.py` | local |
| `metrics.json` | settle-time metrics of the hero and keyframes | `scripts/compute_metrics.py` | hero / keyframe jobs |
| `scale_dependence.json`, `figures/scale_dependence.png` | basin entropy / boundary dimension versus pixel pitch (upstream image, ours, hero, zoom windows) | `scripts/scale_dependence.py` | derived |
| `scans.json`, `figures/oscan_sheet.png` | native-scale scans: 5565 puzzles x 16 EqR-style starts (RUNS 5), long-run check of unsolved starts (RUNS 7), outcome maps around native-scale base points (RUNS 8, cancelled at 15 of 32 windows) | `scripts/summarize_scans.py`, `scripts/view_oscan.py` | RUNS 5, 7, 8 |
| `replay_agreement.json` | how well a second float32 stack (MLX on the Mac) reproduces stored settle times, by local roughness | `scripts/replay_agreement.py` | local |
| `replay/summary.json`, `replay/plan.json`, `replay/replay_in.json` | cluster replay of featured starts and finale pairs on the render engine: plain replay vs stored values, 1e-6 nudges | `scripts/plan_replay.py`, `alice/replay.py`, `scripts/apply_replay.py` | RUNS (replay) |
| `video/pair.json` | the two neighbouring starts shown in the video finale, with their loop-by-loop replays | `scripts/apply_replay.py` | RUNS (replay) |
