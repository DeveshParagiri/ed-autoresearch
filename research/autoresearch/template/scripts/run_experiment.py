#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autoresearch.runner import RunError, run_experiment  # noqa: E402
from autoresearch.validation import dataset_ids, load_experiments, validate_workspace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Record one declared experiment.")
    parser.add_argument("experiment")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = validate_workspace(PROJECT_ROOT)
    if not report.ok:
        print("ERROR workspace validation failed; run scripts/check_workspace.py", file=sys.stderr)
        return 2
    loaded, _ = load_experiments(PROJECT_ROOT)
    record = loaded.get(args.experiment)
    if record is None:
        print(f"ERROR unknown experiment: {args.experiment}", file=sys.stderr)
        return 2
    experiment = record.metadata
    replacements = {"{python}": sys.executable, "{project_root}": str(PROJECT_ROOT)}
    command = [replacements.get(value, value) for value in experiment["execution"].get("argv", [])]
    try:
        run = run_experiment(
            PROJECT_ROOT,
            experiment_id=args.experiment,
            experiment=experiment,
            known_inputs=dataset_ids(PROJECT_ROOT),
            command=command,
            final_check=lambda: _final_check(PROJECT_ROOT),
        )
    except (KeyError, OSError, RunError, ValueError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(run, indent=2, sort_keys=True))
    else:
        path = Path("research") / "experiments" / args.experiment / "runs" / run["id"]
        print(f"run={run['id']} status={run['status']} path={path}")
        if run.get("failure"):
            print(f"failure={run['failure']}")
    return 0 if run["status"] == "completed" else 1


def _final_check(project_root: Path) -> tuple[bool, str]:
    report = validate_workspace(project_root)
    errors = [issue for issue in report.issues if issue.severity == "error"]
    return not errors, "; ".join(f"{issue.code}: {issue.message}" for issue in errors[:5])


if __name__ == "__main__":
    raise SystemExit(main())
