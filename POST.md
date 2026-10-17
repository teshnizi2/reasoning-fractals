# POST.md (draft, never posted)

Every number below comes from `results/` (the sources are listed at the end). Nothing here was sent anywhere.

## LinkedIn draft

I gave a small Sudoku-solving model the same puzzle 4.2 million times and changed only its first "thought".

Each pixel is one starting thought; colour is how many loops the model needs before its answer stops changing. The map is a fractal: 12% of neighbouring pixels differ by five loops or more.

99.9% still reach the right grid. The fractal is in how long it thinks, not whether it gets there.

I rebuilt the loopscape figure (Lai, Bao, Quinn, Gilpin) on A100s: boundary dimension 1.853 against their 1.852. One caveat: nudge a start by one part in a million and a speckle pixel's loop count moves. Smooth regions don't care.

Zoom video and click-a-pixel explorer attached.

## Alternative hooks

1. Same Sudoku, same weights, 4.2 million different first thoughts. Neighbouring starts differ by up to 21 loops of thinking.
2. 99.9% of 4.2 million starting points solve the puzzle. The picture is still a fractal, because it measures how long they take.
3. I coloured in how long a tiny reasoning model thinks, one pixel per starting thought. It looks like gold-leaf marbling, and the speckle in it depends on float32 rounding.

## Where the numbers come from

- 4.2 million starting thoughts, 99.9% solved, 12% of neighbour pairs >= 5 loops, largest gap 21 loops: `results/metrics.json` (hero), `scripts/compute_metrics.py`.
- Boundary dimension 1.853 vs 1.852: `results/metrics.json` (hero, labels clipped to the upstream's 11 levels, every 2nd pixel) and `results/repro/repro250_summary.json` (upstream image analysed with the same code), `results/scale_dependence.json`.
- "One part in a million" nudges, smooth vs speckle pixels: `results/replay/summary.json` (97% of smooth-region pixels unchanged, 17-19% of speckle pixels), `alice/replay.py`.
