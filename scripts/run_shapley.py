#!/usr/bin/env python3
"""Run exact grouped Shapley attribution for a small set of mechanisms."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

from autoresearch.run_tools import (
    build_command,
    numeric_metrics,
    read_json,
    run,
    shapley_values,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "evaluate_candidate.py"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate every mechanism subset, then compute exact Shapley values."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    mechanisms = config.get("mechanisms")
    if not isinstance(mechanisms, list) or not mechanisms:
        raise ValueError("mechanisms must be a nonempty list")
    if any(not isinstance(name, str) or not name for name in mechanisms):
        raise ValueError("mechanism names must be nonempty strings")
    if len(set(mechanisms)) != len(mechanisms):
        raise ValueError("mechanism names must be unique")
    if 1 << len(mechanisms) > 64:
        raise ValueError("exact Shapley is limited to 64 subset evaluations")
    template = config.get("build_command")
    if not isinstance(template, list) or not all(isinstance(item, str) for item in template):
        raise ValueError("build_command must be a JSON array of command arguments")
    joined = " ".join(template)
    if "{mechanisms}" not in joined or "{candidate}" not in joined:
        raise ValueError("build_command must contain {mechanisms} and {candidate}")
    return mechanisms, template


def output_path(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output == ROOT or ROOT not in output.parents:
        raise ValueError("output must be inside this repository")
    return output


def run_subset(
    *,
    number: int,
    enabled: tuple[str, ...],
    template: list[str],
    output: Path,
) -> dict[str, Any]:
    subset = output / "subsets" / f"{number:03d}"
    settings = subset / "mechanisms.json"
    candidate = subset / "candidate.nc"
    evaluation = subset / "evaluation"
    metrics = evaluation / "metrics.json"
    if metrics.is_file():
        if read_json(settings).get("mechanisms") != list(enabled):
            raise ValueError(f"cached subset does not match {subset}")
        return read_json(metrics)

    write_json(settings, {"mechanisms": list(enabled)})
    command = build_command(
        template,
        values={"mechanisms": settings.resolve(), "candidate": candidate.resolve()},
    )
    run(command, cwd=ROOT, logs=subset / "logs" / "build")
    if not candidate.is_file():
        raise FileNotFoundError(f"builder did not write {candidate}")
    run(
        [sys.executable, str(EVALUATOR), str(candidate), "--output", str(evaluation)],
        cwd=ROOT,
        logs=subset / "logs" / "evaluate",
    )
    return read_json(metrics)


def report(
    mechanisms: list[str],
    results: dict[frozenset[str], dict[str, Any]],
) -> dict[str, Any]:
    flattened = {
        subset: numeric_metrics(
            {
                "candidate": values["candidate"],
                "benchmark_sensitivity": values["benchmark_sensitivity"],
            }
        )
        for subset, values in results.items()
    }
    names = set.intersection(*(set(values) for values in flattened.values()))
    empty = frozenset()
    full = frozenset(mechanisms)
    metrics: dict[str, Any] = {}
    for name in sorted(names):
        values = {subset: fields[name] for subset, fields in flattened.items()}
        attribution = shapley_values(mechanisms, values)
        total = values[full] - values[empty]
        metrics[name] = {
            "empty": values[empty],
            "full": values[full],
            "change": total,
            "shapley": attribution,
            "drop_one": {
                mechanism: values[full] - values[full - {mechanism}]
                for mechanism in mechanisms
            },
            "reconstruction_error": sum(attribution.values()) - total,
        }
    return {
        "schema": "ed-fire-shapley/v1",
        "mechanisms": mechanisms,
        "subset_count": len(results),
        "metrics": metrics,
    }


def main() -> int:
    args = arguments()
    config = read_json(args.config.expanduser().resolve())
    mechanisms, template = validate(config)
    output = output_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "config.json"
    if frozen.exists() and read_json(frozen) != config:
        raise ValueError("output already contains a different config")
    write_json(frozen, config)

    combinations = [
        subset
        for size in range(len(mechanisms) + 1)
        for subset in itertools.combinations(mechanisms, size)
    ]
    results: dict[frozenset[str], dict[str, Any]] = {}
    for number, enabled in enumerate(combinations):
        print(f"subset {number + 1}/{len(combinations)}: {', '.join(enabled) or 'none'}")
        results[frozenset(enabled)] = run_subset(
            number=number,
            enabled=enabled,
            template=template,
            output=output,
        )
    write_json(output / "shapley.json", report(mechanisms, results))
    print(f"result={output / 'shapley.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        raise
