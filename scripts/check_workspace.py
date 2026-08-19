#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autoresearch.workspace import status  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate this autoresearch workspace.")
    parser.add_argument("--json", action="store_true", help="print the complete JSON report")
    args = parser.parse_args()
    report = status(PROJECT_ROOT)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for issue in report["issues"]:
            print(
                f"{issue['severity'].upper():7} | {issue['code']:30} | "
                f"{issue['subject']} | {issue['message']}"
            )
        print(
            f"status={report['status']} datasets={report['datasets']} "
            f"framings={report['framings']} experiments={report['experiments']} "
            f"runs={report['runs']} "
            f"runnable={len(report['runnable_experiments'])} "
            f"errors={report['errors']} warnings={report['warnings']}"
        )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
