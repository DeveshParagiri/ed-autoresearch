#!/usr/bin/env python3
"""Create and inspect the project data links declared by the data manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autoresearch.data_install import (  # noqa: E402
    default_source_root,
    install_links,
    load_sources,
    recovery_instructions,
)


FAILURE_STATUSES = {
    "conflict",
    "invalid-source",
    "missing-repository-file",
    "missing-source",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install every cataloged dataset path from one canonical data store."
    )
    parser.add_argument(
        "command",
        choices=("plan", "install"),
        help="plan makes no changes; install creates missing symlinks",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    manifest, _ = load_sources(project_root)
    source_root = args.source_root or default_source_root(project_root, manifest)
    events = install_links(
        project_root,
        source_root,
        dry_run=args.command == "plan",
    )

    failures = []
    optional_missing = []
    for event in events:
        print(f"{event.status.upper():24} {event.item_id}: {event.message}")
        if event.status in FAILURE_STATUSES:
            failures.append(event)
        if event.status == "optional-missing":
            optional_missing.append(event)

    print()
    print(
        f"source_root={source_root.expanduser().resolve()} "
        f"entries={len(events)} failures={len(failures)} "
        f"optional_missing={len(optional_missing)}"
    )
    if failures:
        print()
        print("Recovery instructions")
        for instruction in recovery_instructions(
            project_root, {event.item_id for event in failures}
        ):
            print(instruction.replace("SOURCE_ROOT", str(source_root.expanduser().resolve())))
        return 2
    if optional_missing:
        print()
        print("Optional acquisition instructions")
        for instruction in recovery_instructions(
            project_root, {event.item_id for event in optional_missing}
        ):
            print(instruction.replace("SOURCE_ROOT", str(source_root.expanduser().resolve())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
