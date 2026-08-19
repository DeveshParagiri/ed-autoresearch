#!/usr/bin/env python3
"""Reproduce models A-I and generate their comparable figure suites in one run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_root() -> Path:
    value = os.environ.get("AUTORESEARCH_RUN_ROOT")
    if not value:
        raise RuntimeError("AUTORESEARCH_RUN_ROOT is required")
    root = Path(value).resolve()
    expected = ROOT / "research" / "experiments"
    try:
        root.relative_to(expected)
    except ValueError as error:
        raise RuntimeError("the run root is outside research/experiments") from error
    return root


def invoke(argv: list[str]) -> None:
    print("$ " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)


def main() -> int:
    replay = run_root() / "work" / "models-a-i"
    invoke(
        [
            sys.executable,
            "scripts/reproduce_models.py",
            "--models",
            "all",
            "--output",
            str(replay),
            "--evaluate",
        ]
    )
    invoke(
        [
            sys.executable,
            "scripts/figure_models.py",
            "--models",
            "all",
            "--replay-root",
            str(replay),
            "--output",
            str(replay / "figures"),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
