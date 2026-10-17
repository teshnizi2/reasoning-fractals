#!/usr/bin/env python
"""Build README.md from docs/README.template.md and the JSON files under results/ (see render/readme.py).

    .venv/bin/python scripts/build_readme.py            # strict: every placeholder must resolve
    .venv/bin/python scripts/build_readme.py --partial  # unresolved placeholders become '[pending: key]' (drafting only)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import readme as RM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--template", default="docs/README.template.md")
    ap.add_argument("--out", default="README.md")
    a = ap.parse_args()
    tpl = (ROOT / a.template).read_text()
    N = RM.load_numbers(ROOT)
    T = RM.tables(ROOT, N)
    if a.partial:
        for m in RM.PLACEHOLDER.finditer(tpl):
            key = m.group(1)
            if key.startswith("table:"):
                T.setdefault(key[6:], f"*[pending: {key}]*")
            else:
                N.setdefault(key, f"[pending: {key}]")
        # numbers that are pending are strings: format them as plain text
        tpl = re.sub(r"\{\{\s*([^{}|]+?)\s*\|\s*[a-z0-9]+\s*\}\}", lambda m: "{{%s}}" % m.group(1) if isinstance(N.get(m.group(1).strip()), str) else m.group(0), tpl)
    out = RM.substitute(tpl, N, T)
    (ROOT / a.out).write_text(out)
    print(f"wrote {a.out} ({len(out.splitlines())} lines, {len(RM.PLACEHOLDER.findall(tpl))} placeholders)")


if __name__ == "__main__":
    main()
