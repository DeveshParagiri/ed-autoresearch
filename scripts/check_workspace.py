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
    parser = argparse.ArgumentParser(description="Validate the simplified ED-Fire workspace.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = status(PROJECT_ROOT)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for item in report["issues"]:
            print(
                f"{item['severity'].upper():7} | {item['code']:30} | "
                f"{item['subject']} | {item['message']}"
            )
        print(
            f"status={report['status']} datasets={report['datasets']} "
            f"models={report['models']} contracts={report['contracts']} "
            f"errors={report['errors']} warnings={report['warnings']}"
        )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
