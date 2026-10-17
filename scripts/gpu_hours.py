#!/usr/bin/env python
"""Sum this project's ALICE usage from sacct (jobs named rf-*) into results/alice/gpu_hours.json."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import gpuhours as G  # noqa: E402

cmd = ("sacct -u $USER -S 2026-10-17 -X -P --noheader "
       "--format=JobID,JobName,State,Elapsed,ElapsedRaw,AllocTRES,Start,End")
out = subprocess.run(["ssh", "alice2", cmd], capture_output=True, text=True, check=True).stdout
rows = []
for line in out.splitlines():
    f = line.split("|")
    if len(f) < 8 or not f[1].startswith("rf-") or f[4] in ("", "0"):
        continue
    rows.append(dict(JobID=f[0], JobName=f[1], State=f[2], Elapsed=f[3], ElapsedRaw=f[4], AllocTRES=f[5], Start=f[6], End=f[7]))
res = G.summarize(rows)
res["jobs"] = rows
(ROOT / "results/alice").mkdir(parents=True, exist_ok=True)
(ROOT / "results/alice/gpu_hours.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("by_name", "total")}, indent=1))
