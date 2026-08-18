#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autoresearch.runner import RunError, run_experiment  # noqa: E402
from autoresearch.workspace import experiments, final_check, input_ids, status  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one declared experiment.")
    parser.add_argument("experiment", help="experiment ID under research/experiments")
    parser.add_argument("--json", action="store_true", help="print the final run record")
    args = parser.parse_args()

    report = status(PROJECT_ROOT)
    if report["status"] != "ok":
        print("ERROR workspace validation failed; run scripts/check_workspace.py", file=sys.stderr)
        return 2
    records = experiments(PROJECT_ROOT)
    experiment = records.get(args.experiment)
    if experiment is None:
        print(f"ERROR unknown experiment: {args.experiment}", file=sys.stderr)
        return 2
    execution = experiment.get("execution", {})
    command = [
        {
            "{python}": sys.executable,
            "{project_root}": str(PROJECT_ROOT),
        }.get(value, value)
        for value in execution.get("argv", [])
    ]
    try:
        record = run_experiment(
            PROJECT_ROOT,
            experiment_id=args.experiment,
            experiment=experiment,
            known_inputs=input_ids(PROJECT_ROOT),
            command=command,
            final_check=lambda: final_check(PROJECT_ROOT),
        )
    except (KeyError, OSError, RunError, ValueError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        location = PROJECT_ROOT / "research" / "experiments" / args.experiment / "runs" / record["id"]
        print(f"run={record['id']} status={record['status']} path={location.relative_to(PROJECT_ROOT)}")
        if record.get("failure"):
            print(f"failure={record['failure']}")
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
