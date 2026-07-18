#!/usr/bin/env python3
"""Watch paper.md (and figures/, refs.bib) and recompile PDF on save.

  cd paper
  python watch.py

Then edit paper.md in any editor. On each save the PDF rebuilds.
Requires: typst on PATH. Stdlib only (no watchdog package).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import compile_pdf  # noqa: E402

WATCH = [
    HERE / "paper.md",
    HERE / "refs.bib",
    HERE / "figures",
]
POLL_S = 0.6


def snapshot() -> dict[str, float]:
    out: dict[str, float] = {}
    for p in WATCH:
        if p.is_file():
            out[str(p)] = p.stat().st_mtime
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    out[str(f)] = f.stat().st_mtime
    return out


def main() -> None:
    print("Watching paper.md / figures / refs.bib — Ctrl-C to stop")
    print("Initial compile…")
    try:
        compile_pdf()
    except SystemExit as e:
        if e.code not in (0, None):
            print("Initial build failed; still watching for fixes…", file=sys.stderr)

    prev = snapshot()
    while True:
        time.sleep(POLL_S)
        cur = snapshot()
        if cur != prev:
            changed = [k for k in cur if cur.get(k) != prev.get(k)]
            changed += [k for k in prev if k not in cur]
            names = ", ".join(Path(c).name for c in changed[:5])
            print(f"\n[{time.strftime('%H:%M:%S')}] change: {names}")
            try:
                compile_pdf()
            except SystemExit:
                print("build failed — fix paper.md and save again", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"build error: {exc}", file=sys.stderr)
            prev = snapshot()


if __name__ == "__main__":
    main()
