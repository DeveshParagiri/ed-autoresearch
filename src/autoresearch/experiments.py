from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .catalog import ValidationIssue


EXPERIMENT_SCHEMA = "autoresearch-experiment/v1"
EXPERIMENT_STATUSES = {"proposed", "running", "completed", "paused", "closed", "invalid"}
EXECUTION_MODES = {"mechanistic", "simulation", "hybrid"}
REQUIRED_EXPERIMENT_FIELDS = {
    "schema",
    "id",
    "title",
    "kind",
    "status",
    "created_at",
    "parents",
    "inputs",
    "contract",
    "execution",
}
REQUIRED_BODY_SECTIONS = (
    "# Question",
    "# Change",
    "# Prediction",
    "# Plan",
    "# Result",
    "# Decision",
    "# Revisit when",
)


@dataclass(frozen=True)
class Experiment:
    path: Path
    metadata: dict[str, Any]
    body: str


@dataclass(frozen=True)
class ExperimentValidation:
    experiment_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "error",
            "experiments": self.experiment_count,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def safe_relative_path(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not Path(value).is_absolute()
        and ".." not in Path(value).parts
    )


def parse_experiment(path: Path) -> Experiment:
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening YAML front-matter delimiter")
    try:
        closing = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError("missing closing YAML front-matter delimiter") from exc
    metadata = yaml.safe_load("\n".join(lines[1:closing]))
    if not isinstance(metadata, dict):
        raise ValueError("front matter must be a YAML mapping")
    created_at = metadata.get("created_at")
    if not isinstance(created_at, str) and hasattr(created_at, "isoformat"):
        metadata["created_at"] = created_at.isoformat()
    return Experiment(path, metadata, "\n".join(lines[closing + 1 :]).strip())


def experiment_directories(project_root: Path) -> tuple[Path, ...]:
    root = project_root / "research" / "experiments"
    if not root.is_dir():
        return ()
    return tuple(sorted(path for path in root.iterdir() if path.is_dir()))


def load_experiments(project_root: Path) -> tuple[dict[str, Experiment], list[ValidationIssue]]:
    root = project_root / "research" / "experiments"
    if not root.is_dir():
        return {}, [
            ValidationIssue("error", "missing-experiments", "research/experiments", str(root))
        ]

    experiments: dict[str, Experiment] = {}
    issues: list[ValidationIssue] = []
    for directory in experiment_directories(project_root):
        path = directory / "experiment.md"
        if not path.is_file():
            issues.append(
                ValidationIssue("error", "missing-experiment-file", directory.name, str(path))
            )
            continue
        try:
            experiment = parse_experiment(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            issues.append(ValidationIssue("error", "invalid-experiment", directory.name, str(exc)))
            continue
        metadata = experiment.metadata
        experiment_id = metadata.get("id")
        missing = sorted(REQUIRED_EXPERIMENT_FIELDS - metadata.keys())
        if missing or not isinstance(experiment_id, str) or not experiment_id:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid-experiment-fields",
                    directory.name,
                    ", ".join(missing) or repr(experiment_id),
                )
            )
            continue
        if experiment_id != directory.name:
            issues.append(
                ValidationIssue(
                    "error",
                    "experiment-directory-mismatch",
                    experiment_id,
                    directory.name,
                )
            )
        if experiment_id in experiments:
            issues.append(
                ValidationIssue("error", "duplicate-experiment-id", experiment_id, str(path))
            )
            continue
        experiments[experiment_id] = experiment

    for experiment_id, experiment in experiments.items():
        metadata = experiment.metadata
        if metadata.get("schema") != EXPERIMENT_SCHEMA:
            issues.append(
                ValidationIssue(
                    "error", "invalid-experiment-schema", experiment_id, repr(metadata.get("schema"))
                )
            )
        for field in ("title", "kind", "created_at"):
            if not isinstance(metadata.get(field), str) or not metadata[field].strip():
                issues.append(ValidationIssue("error", "invalid-experiment-field", experiment_id, field))
        if metadata.get("status") not in EXPERIMENT_STATUSES:
            issues.append(
                ValidationIssue(
                    "error", "invalid-experiment-status", experiment_id, repr(metadata.get("status"))
                )
            )

        parents = metadata.get("parents")
        if not isinstance(parents, list) or not all(isinstance(value, str) for value in parents):
            issues.append(ValidationIssue("error", "invalid-experiment-parents", experiment_id, repr(parents)))
        else:
            for parent in parents:
                if parent == experiment_id or parent not in experiments:
                    issues.append(ValidationIssue("error", "unknown-experiment-parent", experiment_id, parent))

        inputs = metadata.get("inputs")
        if not isinstance(inputs, list) or not all(isinstance(value, str) for value in inputs):
            issues.append(ValidationIssue("error", "invalid-experiment-inputs", experiment_id, repr(inputs)))
        if not safe_relative_path(metadata.get("contract")):
            issues.append(
                ValidationIssue("error", "invalid-experiment-contract", experiment_id, repr(metadata.get("contract")))
            )

        execution = metadata.get("execution")
        if not isinstance(execution, dict):
            issues.append(ValidationIssue("error", "invalid-execution", experiment_id, repr(execution)))
        else:
            mode = execution.get("mode")
            tool = execution.get("tool")
            adapter = execution.get("adapter")
            argv = execution.get("argv")
            if mode not in EXECUTION_MODES:
                issues.append(ValidationIssue("error", "invalid-execution-mode", experiment_id, repr(mode)))
            if not isinstance(tool, str) or not tool.strip():
                issues.append(ValidationIssue("error", "invalid-execution-tool", experiment_id, repr(tool)))
            if adapter is None:
                issues.append(ValidationIssue("warning", "missing-execution-adapter", experiment_id, "not runnable"))
            elif not safe_relative_path(adapter) or not (project_root / adapter).is_file():
                issues.append(ValidationIssue("error", "invalid-execution-adapter", experiment_id, repr(adapter)))
            elif (
                not isinstance(argv, list)
                or not argv
                or not all(isinstance(value, str) and value for value in argv)
                or adapter not in argv
            ):
                issues.append(ValidationIssue("error", "invalid-execution-argv", experiment_id, repr(argv)))
            if tool == "optuna":
                issues.extend(_validate_optuna_search(project_root, experiment_id, metadata.get("search")))
            elif metadata.get("search") is not None:
                issues.append(
                    ValidationIssue(
                        "error", "unexpected-search-config", experiment_id, "execution.tool is not optuna"
                    )
                )

        for heading in REQUIRED_BODY_SECTIONS:
            if heading not in experiment.body:
                issues.append(ValidationIssue("error", "missing-experiment-section", experiment_id, heading))

    issues.extend(_cycle_issues(experiments))
    return experiments, issues


def _validate_optuna_search(
    project_root: Path, experiment_id: str, search: Any
) -> list[ValidationIssue]:
    if not isinstance(search, dict) or search.get("engine") != "optuna":
        return [ValidationIssue("error", "invalid-optuna-search", experiment_id, repr(search))]
    issues: list[ValidationIssue] = []
    objectives = search.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        issues.append(ValidationIssue("error", "missing-optuna-objectives", experiment_id, repr(objectives)))
    else:
        for objective in objectives:
            if (
                not isinstance(objective, dict)
                or not isinstance(objective.get("metric"), str)
                or objective.get("direction") not in {"minimize", "maximize"}
            ):
                issues.append(ValidationIssue("error", "invalid-optuna-objective", experiment_id, repr(objective)))
    for field in ("study_name", "sampler", "pruner", "selection_rule"):
        if not isinstance(search.get(field), str) or not search[field].strip():
            issues.append(ValidationIssue("error", "invalid-optuna-field", experiment_id, field))
    if not isinstance(search.get("seed"), int):
        issues.append(ValidationIssue("error", "invalid-optuna-seed", experiment_id, repr(search.get("seed"))))
    budget = search.get("budget")
    if not isinstance(budget, dict) or not any(
        isinstance(budget.get(field), int) and budget[field] > 0
        for field in ("trials", "timeout_seconds")
    ):
        issues.append(ValidationIssue("error", "invalid-optuna-budget", experiment_id, repr(budget)))
    parameter_space = search.get("parameter_space")
    if not safe_relative_path(parameter_space) or not (project_root / parameter_space).is_file():
        issues.append(
            ValidationIssue("error", "invalid-optuna-parameter-space", experiment_id, repr(parameter_space))
        )
    if isinstance(objectives, list) and len(objectives) > 1 and search.get("pruner") != "NopPruner":
        issues.append(
            ValidationIssue(
                "error",
                "multiobjective-optuna-pruning",
                experiment_id,
                "Optuna pruning is only supported for single-objective studies; use NopPruner",
            )
        )
    return issues


def _cycle_issues(experiments: dict[str, Experiment]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(experiment_id: str) -> None:
        if experiment_id in visited:
            return
        if experiment_id in visiting:
            issues.append(ValidationIssue("error", "experiment-cycle", experiment_id, "parent cycle"))
            return
        visiting.add(experiment_id)
        parents = experiments[experiment_id].metadata.get("parents", [])
        if isinstance(parents, list):
            for parent in parents:
                if parent in experiments:
                    visit(parent)
        visiting.remove(experiment_id)
        visited.add(experiment_id)

    for experiment_id in experiments:
        visit(experiment_id)
    return issues


def validate_experiments(project_root: Path) -> ExperimentValidation:
    project_root = project_root.resolve()
    issues: list[ValidationIssue] = []
    if not (project_root / "research.md").is_file():
        issues.append(
            ValidationIssue("error", "missing-charter", "research.md", str(project_root / "research.md"))
        )
    experiments, experiment_issues = load_experiments(project_root)
    issues.extend(experiment_issues)
    return ExperimentValidation(len(experiments), tuple(issues))


def validate_event_log(path: Path) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if not path.is_file():
        return (ValidationIssue("error", "missing-event-log", path.parent.name, str(path)),)
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(ValidationIssue("error", "invalid-event", f"{path}:{number}", str(exc)))
            continue
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("timestamp"), str)
            or not isinstance(event.get("event"), str)
            or not isinstance(event.get("metadata"), dict)
        ):
            issues.append(ValidationIssue("error", "invalid-event", f"{path}:{number}", repr(event)))
    return tuple(issues)
