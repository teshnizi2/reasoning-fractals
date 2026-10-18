#!/usr/bin/env python
"""Final checks of the deliverables (run before reporting): files, formats, honesty rules.

    .venv/bin/python scripts/verify_deliverables.py

Prints PASS/FAIL per check and exits non-zero if any fail.  Nothing here modifies the project.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout


def probe(path: Path) -> dict:
    out = sh("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,duration:format=duration",
             "-of", "json", str(path))
    d = json.loads(out)
    s = d["streams"][0]
    s["duration"] = float(s.get("duration") or d["format"]["duration"])
    streams = json.loads(sh("ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(path)))["streams"]
    s["audio"] = any(x["codec_type"] == "audio" for x in streams)
    return s


def main() -> None:
    # --- media
    hero = ROOT / "media/hero.png"
    if hero.exists():
        w, h = Image.open(hero).size
        check("hero.png is at least 2048x2048", w >= 2048 and h >= 2048, f"{w}x{h}")
    else:
        check("hero.png exists", False)
    for name, size in (("media/hero-1080x1350.png", (1080, 1350)), ("media/hero-1080x1080.png", (1080, 1080))):
        p = ROOT / name
        check(f"{name} has the right size", p.exists() and Image.open(p).size == size)
    for name, size in (("media/zoom-1080x1350.mp4", (1080, 1350)), ("media/zoom-1080x1080.mp4", (1080, 1080))):
        p = ROOT / name
        if not p.exists():
            check(f"{name} exists", False)
            continue
        s = probe(p)
        check(f"{name}: H.264, yuv420p, {size[0]}x{size[1]}, no audio", s["codec_name"] == "h264" and s["pix_fmt"] == "yuv420p"
              and (s["width"], s["height"]) == size and not s["audio"], f"{s['codec_name']} {s['pix_fmt']} {s['width']}x{s['height']} audio={s['audio']}")
        check(f"{name}: 15-22 s", 15 <= s["duration"] <= 22, f"{s['duration']:.1f} s")
    # --- explorer
    man_p = ROOT / "site/data/manifest.json"
    check("explorer pack exists", man_p.exists())
    if man_p.exists():
        man = json.loads(man_p.read_text())
        check("pack has the hero and zoom layers", len(man["layers"]) >= 4, f"{len(man['layers'])} layers")
        feats = man["meta"].get("featured") or []
        check("featured starts carry 24-loop traces", len(feats) >= 2 and all(len(f["trace"]) == 24 for f in feats), f"{[f['id'] for f in feats]}")
        check("pack is not a mock/preview", not man["meta"].get("mock") and not man["meta"].get("preview") and not man["meta"].get("proto"))
        check("explorer copy exists", bool(man["meta"].get("limits")) and bool(man["meta"].get("how")))
    # --- documents
    for name in ("README.md", "STATE.md", "POST.md", "RUNS.md"):
        check(f"{name} exists", (ROOT / name).exists())
    readme = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
    check("README has no unresolved placeholders", "[pending" not in readme and "{{" not in readme)
    post = (ROOT / "POST.md").read_text() if (ROOT / "POST.md").exists() else ""
    m = re.search(r"^## LinkedIn draft\s*\n(.*?)(?=^## )", post, re.S | re.M)
    words = len(m.group(1).split()) if m else -1
    check("POST.md draft is at most 120 words", 0 < words <= 120, f"{words} words")
    check("POST.md has 3 alternative hooks", len(re.findall(r"^\d\.\s", post.split("## Alternative hooks")[-1], re.M)) >= 3 if "## Alternative hooks" in post else False)
    # --- RUNS.md covers every job in gpu_hours.json
    gh = ROOT / "results/alice/gpu_hours.json"
    runs = (ROOT / "RUNS.md").read_text()
    if gh.exists():
        ids = {j["JobID"].split("_")[0] for j in json.loads(gh.read_text())["jobs"]}
        missing = sorted(i for i in ids if i not in runs)
        check("every rf-* job id appears in RUNS.md", not missing, f"missing {missing}" if missing else f"{len(ids)} jobs")
    # --- honesty / safety rules
    log = sh("git", "log", "--format=%B")
    check("no AI attribution in commit messages", not re.search(r"co-authored-by|generated with|claude", log, re.I))
    tracked = sh("git", "ls-files").splitlines()
    check("no .env / key files tracked", not [t for t in tracked if re.search(r"(^|/)\.env|\.pem$|id_rsa|secret", t, re.I)])
    big = [t for t in tracked if (ROOT / t).exists() and (ROOT / t).stat().st_size > 30_000_000]
    check("no file over 30 MB is tracked", not big, str(big))
    for t in ("data/asset_samples.npz", "results/repro/repro250_side_by_side.png"):
        check(f"{t} (derived from the unlicensed upstream image) is not tracked", t not in tracked)
    ever = set(sh("git", "log", "--all", "--name-only", "--format=").split())
    derived_ever = sorted(t for t in ever if re.search(r"(^|/)asset_samples\.npz$|results/repro/probe_[^/]*\.npz$|repro250_side_by_side\.png$", t))
    check("no file derived from the upstream image is in any commit", not derived_ever, str(derived_ever))
    urls = [ln.split()[1] for ln in sh("git", "remote", "-v").splitlines()]
    check("the only remote (if any) is the published repository",
          all(u.removesuffix(".git").endswith("github.com/teshnizi2/reasoning-fractals") for u in urls), str(urls))
    secrets = sh("git", "grep", "-nI", "-E", r"(api[_-]?key|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}")
    check("no secret-looking strings in tracked files", not secrets.strip(), secrets.strip()[:200])
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
