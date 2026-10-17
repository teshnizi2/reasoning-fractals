# RUNS.md

Append-only. One row per experiment: one group of Slurm jobs that answers one question. Never edit or reorder old rows; a correction is a new row that cites the old row number, so row numbers stay stable ids. Record smoke tests and cancelled jobs too, marked as such, and keep them out of the results. State the gap in the paper's own units.

Verdict words: MATCH | MISMATCH | UNDECIDED | FAILED | SMOKE | CANCELLED

| # | Date (start) | Job id(s) | Cluster / partition / hardware | Commit or version | Config | Metric (ours vs reference) | Verdict | Wall / GPU-h | Notes file |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-17 14:44 | 5246129 | ALICE gpu-short, node876, A100 80GB PCIe, torch 2.14.1+cu130 | 1bf64fa | alice/bench_cuda.py (smoke): torchfn engine, asset puzzle, 4096 frozen asset pixels (xy_fy), fp32/tf32/bf16, scale scan | throughput fp32 23.0 / tf32 59-60 / bf16 ~90 rollouts/s (B=256..1024); pixel agreement with upstream asset fp32 65.0% (99.9% where asset locally flat, 48.1% in filaments), tf32 21.9%, bf16 13.6% | SMOKE | 12 min / 0.2 GPU-h | results/alice/smoke_5246129_bench.json |
