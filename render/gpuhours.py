"""GPU-hour accounting from ``sacct`` rows.  A100 MIG slices (``4g.40gb``, ``3g.40gb``) count as 4/7 and 3/7 of an A100."""
from __future__ import annotations

import re

_FRACTION = {"a100": 1.0, "4g.40gb": 4 / 7, "3g.40gb": 3 / 7}


def parse_alloc_tres(tres: str) -> dict[str, int]:
    """'cpu=4,gres/gpu=1,gres/gpu:4g.40gb=1' -> {'4g.40gb': 1} (typed GPU entries only)."""
    out: dict[str, int] = {}
    for m in re.finditer(r"gres/gpu:([^=,]+)=(\d+)", tres or ""):
        out[m.group(1)] = out.get(m.group(1), 0) + int(m.group(2))
    return out


def a100_equiv(counts: dict[str, int]) -> float:
    return sum(_FRACTION.get(k, 1.0) * n for k, n in counts.items())


def summarize(rows: list[dict]) -> dict:
    by: dict[str, dict] = {}
    for r in rows:
        hours = float(r["ElapsedRaw"]) / 3600.0
        c = parse_alloc_tres(r["AllocTRES"])
        n_gpu = sum(c.values())
        e = by.setdefault(r["JobName"], dict(jobs=0, gpu_hours_raw=0.0, a100_hours=0.0))
        e["jobs"] += 1
        e["gpu_hours_raw"] += hours * n_gpu
        e["a100_hours"] += hours * a100_equiv(c)
    total = dict(jobs=sum(v["jobs"] for v in by.values()), gpu_hours_raw=sum(v["gpu_hours_raw"] for v in by.values()),
                 a100_hours=sum(v["a100_hours"] for v in by.values()))
    return dict(by_name=by, total=total)
