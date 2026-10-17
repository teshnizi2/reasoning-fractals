#!/usr/bin/env python
"""Sample a rating-stratified set of Sudoku-Extreme test puzzles for the multistability scan."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import init as I  # noqa: E402
from render import puzzles as P  # noqa: E402

rows = []
with open(ROOT / "data/raw/sudoku-extreme/test.csv", newline="") as f:
    r = csv.reader(f)
    next(r)
    for src, q, a, rating in r:
        rows.append((src, q, a, int(rating)))
# easy-to-medium puzzles: a stratified sample; every puzzle rated >= 100 (the tail where a model is most likely torn)
pick = I.stratified(rows, edges=(0, 10, 30, 60, 100), per_bin=300, seed=0)
pick += I.stratified(rows, edges=(100, 10_000), per_bin=100_000, seed=0)
have = {p[1] for p in pick}
for p in (P.ASSET, P.NOTEBOOK):  # always include the two puzzles we know
    if p.question not in have:
        pick.append((p.source, p.question, p.answer, p.rating))
out = [dict(source=s, q=q, a=a, rating=rt) for s, q, a, rt in pick]
(ROOT / "data/puzzles_scan.json").write_text(json.dumps(out))
import collections
print(len(out), "puzzles; per rating bin:", dict(sorted(collections.Counter(min(x["rating"] // 50, 7) for x in out).items())))
