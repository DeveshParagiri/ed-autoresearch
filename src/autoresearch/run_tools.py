"""Small shared helpers for search and attribution scripts."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_command(
    template: object,
    *,
    values: Mapping[str, Path],
) -> list[str]:
    if not isinstance(template, list) or not template:
        raise ValueError("build_command must be a nonempty JSON array")
    command: list[str] = []
    for item in template:
        if not isinstance(item, str) or not item:
            raise ValueError("build_command entries must be nonempty strings")
        for name, path in values.items():
            item = item.replace("{" + name + "}", str(path))
        command.append(item)
    return command


def run(command: Sequence[str], *, cwd: Path, logs: Path) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    (logs / "stdout.log").write_text(completed.stdout)
    (logs / "stderr.log").write_text(completed.stderr)
    write_json(logs / "command.json", {"argv": list(command), "exit_code": completed.returncode})
    if completed.returncode != 0:
        raise RuntimeError(f"command failed; see {logs}")


def metric(metrics: Mapping[str, Any], path: str) -> float:
    value: object = metrics
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"metric is missing: {path}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric is not numeric: {path}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metric is not finite: {path}")
    return number


def numeric_metrics(value: object, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for name, item in value.items():
            key = f"{prefix}.{name}" if prefix else str(name)
            result.update(numeric_metrics(item, key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            result[prefix] = number
    return result


def shapley_values(
    mechanisms: Sequence[str],
    values: Mapping[frozenset[str], float],
) -> dict[str, float]:
    count = len(mechanisms)
    if len(values) != 1 << count:
        raise ValueError("subset values are incomplete")
    denominator = math.factorial(count)
    result: dict[str, float] = {}
    for mechanism in mechanisms:
        others = [name for name in mechanisms if name != mechanism]
        contribution = 0.0
        for mask in range(1 << len(others)):
            subset = frozenset(
                name for index, name in enumerate(others) if mask & (1 << index)
            )
            size = len(subset)
            weight = math.factorial(size) * math.factorial(count - size - 1) / denominator
            contribution += weight * (values[subset | {mechanism}] - values[subset])
        result[mechanism] = contribution
    return result
