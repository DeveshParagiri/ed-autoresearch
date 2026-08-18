#!/usr/bin/env python3
"""Plan or install the data links declared by this workspace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autoresearch.data_install import install, source_root  # noqa: E402


FAILURES = {"conflict", "invalid-source", "missing-repository-file", "missing-source"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "install"))
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()

    root = source_root(PROJECT_ROOT, args.source_root)
    events = install(PROJECT_ROOT, root, dry_run=args.command == "plan")
    failures = 0
    for event in events:
        print(f"{event.status.upper():24} {event.item_id}: {event.message}")
        failures += event.status in FAILURES
    print(f"source_root={root} events={len(events)} failures={failures}")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
