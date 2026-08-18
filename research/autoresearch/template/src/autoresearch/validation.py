from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .runner import runnable_experiments, sha256_file, validate_contracts, validate_runs


EXPERIMENT_FIELDS = {
    "schema", "id", "kind", "title", "status", "created_at", "parents",
    "inputs", "contract", "execution",
}
EXPERIMENT_STATUSES = {"proposed", "running", "completed", "paused", "closed", "invalid"}
EXPERIMENT_SECTIONS = (
    "# Question", "# Change", "# Prediction", "# Plan", "# Result", "# Decision", "# Revisit when",
)
SOURCE_FIELDS = {
    "id", "source_path", "source_url", "acquisition", "source_name", "version",
    "retrieval", "time_coverage", "spatial_coverage", "units", "preprocessing",
    "limitations", "license", "integrity", "experiments",
}


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    subject: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return self.__dict__


@dataclass(frozen=True)
class Experiment:
    path: Path
    metadata: dict[str, Any]
    body: str


@dataclass(frozen=True)
class ValidationReport:
    dataset_count: int
    experiment_count: int
    run_count: int
    issues: tuple[Issue, ...]
    runnable_experiments: tuple[str, ...] = ()

    @property
    def errors(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warnings(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def ok(self) -> bool:
        return self.errors == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "error",
            "datasets": self.dataset_count,
            "experiments": self.experiment_count,
            "runs": self.run_count,
            "runnable_experiments": list(self.runnable_experiments),
            "errors": self.errors,
            "warnings": self.warnings,
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
        raise ValueError("missing opening YAML delimiter")
    try:
        closing = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("missing closing YAML delimiter") from exc
    metadata = yaml.safe_load("\n".join(lines[1:closing]))
    if not isinstance(metadata, dict):
        raise ValueError("front matter must be a mapping")
    created_at = metadata.get("created_at")
    if not isinstance(created_at, str) and hasattr(created_at, "isoformat"):
        metadata["created_at"] = created_at.isoformat()
    return Experiment(path, metadata, "\n".join(lines[closing + 1 :]).strip())


def load_experiments(project_root: Path) -> tuple[dict[str, Experiment], list[Issue]]:
    experiments_root = project_root / "research" / "experiments"
    if not experiments_root.is_dir():
        return {}, [Issue("error", "missing-experiments", "research/experiments", str(experiments_root))]
    experiments: dict[str, Experiment] = {}
    issues: list[Issue] = []
    for directory in sorted(path for path in experiments_root.iterdir() if path.is_dir()):
        path = directory / "experiment.md"
        if not path.is_file():
            issues.append(Issue("error", "missing-experiment-file", directory.name, str(path)))
            continue
        try:
            experiment = parse_experiment(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            issues.append(Issue("error", "invalid-experiment", directory.name, str(exc)))
            continue
        missing = sorted(EXPERIMENT_FIELDS - experiment.metadata.keys())
        experiment_id = experiment.metadata.get("id")
        if missing or not isinstance(experiment_id, str) or not experiment_id:
            issues.append(
                Issue("error", "invalid-experiment-fields", directory.name, ", ".join(missing) or repr(experiment_id))
            )
            continue
        if experiment_id != directory.name:
            issues.append(Issue("error", "experiment-directory-mismatch", experiment_id, directory.name))
        if experiment_id in experiments:
            issues.append(Issue("error", "duplicate-experiment-id", experiment_id, path.name))
            continue
        experiments[experiment_id] = experiment

    for experiment_id, experiment in experiments.items():
        metadata = experiment.metadata
        if metadata.get("schema") != "autoresearch-experiment/v1":
            issues.append(Issue("error", "invalid-experiment-schema", experiment_id, repr(metadata.get("schema"))))
        for field in ("kind", "title", "created_at"):
            if not isinstance(metadata.get(field), str) or not metadata[field].strip():
                issues.append(Issue("error", "invalid-experiment-field", experiment_id, field))
        if metadata.get("status") not in EXPERIMENT_STATUSES:
            issues.append(Issue("error", "invalid-experiment-status", experiment_id, repr(metadata.get("status"))))
        parents = metadata.get("parents")
        if not isinstance(parents, list) or not all(isinstance(value, str) for value in parents):
            issues.append(Issue("error", "invalid-experiment-parents", experiment_id, repr(parents)))
        else:
            for parent in parents:
                if parent == experiment_id or parent not in experiments:
                    issues.append(Issue("error", "unknown-experiment-parent", experiment_id, parent))
        inputs = metadata.get("inputs")
        if not isinstance(inputs, list) or not all(isinstance(value, str) for value in inputs):
            issues.append(Issue("error", "invalid-experiment-inputs", experiment_id, repr(inputs)))
        if not safe_relative_path(metadata.get("contract")):
            issues.append(Issue("error", "invalid-experiment-contract", experiment_id, repr(metadata.get("contract"))))
        execution = metadata.get("execution")
        if not isinstance(execution, dict):
            issues.append(Issue("error", "invalid-execution", experiment_id, repr(execution)))
        else:
            mode, tool, adapter, argv = (
                execution.get("mode"), execution.get("tool"), execution.get("adapter"), execution.get("argv")
            )
            if mode not in {"mechanistic", "simulation", "hybrid"}:
                issues.append(Issue("error", "invalid-execution-mode", experiment_id, repr(mode)))
            if not isinstance(tool, str) or not tool:
                issues.append(Issue("error", "invalid-execution-tool", experiment_id, repr(tool)))
            if adapter is None:
                issues.append(Issue("warning", "missing-execution-adapter", experiment_id, "not runnable"))
            elif not safe_relative_path(adapter) or not (project_root / adapter).is_file():
                issues.append(Issue("error", "invalid-execution-adapter", experiment_id, repr(adapter)))
            elif not isinstance(argv, list) or not argv or adapter not in argv:
                issues.append(Issue("error", "invalid-execution-argv", experiment_id, repr(argv)))
            if tool == "optuna":
                issues.extend(_validate_optuna(project_root, experiment_id, metadata.get("search")))
            elif metadata.get("search") is not None:
                issues.append(Issue("error", "unexpected-search-config", experiment_id, repr(metadata.get("search"))))
        for heading in EXPERIMENT_SECTIONS:
            if heading not in experiment.body:
                issues.append(Issue("error", "missing-experiment-section", experiment_id, heading))
    issues.extend(_cycle_issues(experiments))
    return experiments, issues


def _validate_optuna(project_root: Path, experiment_id: str, search: Any) -> list[Issue]:
    if not isinstance(search, dict) or search.get("engine") != "optuna":
        return [Issue("error", "invalid-optuna-search", experiment_id, repr(search))]
    issues: list[Issue] = []
    objectives = search.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        issues.append(Issue("error", "missing-optuna-objectives", experiment_id, repr(objectives)))
    else:
        for objective in objectives:
            if not isinstance(objective, dict) or not isinstance(objective.get("metric"), str) or objective.get("direction") not in {"minimize", "maximize"}:
                issues.append(Issue("error", "invalid-optuna-objective", experiment_id, repr(objective)))
    for field in ("study_name", "sampler", "pruner", "selection_rule"):
        if not isinstance(search.get(field), str) or not search[field]:
            issues.append(Issue("error", "invalid-optuna-field", experiment_id, field))
    if not isinstance(search.get("seed"), int):
        issues.append(Issue("error", "invalid-optuna-seed", experiment_id, repr(search.get("seed"))))
    budget = search.get("budget")
    if not isinstance(budget, dict) or not any(isinstance(budget.get(field), int) and budget[field] > 0 for field in ("trials", "timeout_seconds")):
        issues.append(Issue("error", "invalid-optuna-budget", experiment_id, repr(budget)))
    parameter_space = search.get("parameter_space")
    if not safe_relative_path(parameter_space) or not (project_root / parameter_space).is_file():
        issues.append(Issue("error", "invalid-optuna-parameter-space", experiment_id, repr(parameter_space)))
    if isinstance(objectives, list) and len(objectives) > 1 and search.get("pruner") != "NopPruner":
        issues.append(Issue("error", "multiobjective-optuna-pruning", experiment_id, "use NopPruner"))
    return issues


def _cycle_issues(experiments: dict[str, Experiment]) -> list[Issue]:
    issues: list[Issue] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(experiment_id: str) -> None:
        if experiment_id in visited:
            return
        if experiment_id in visiting:
            issues.append(Issue("error", "experiment-cycle", experiment_id, "parent cycle"))
            return
        visiting.add(experiment_id)
        for parent in experiments[experiment_id].metadata.get("parents", []):
            if parent in experiments:
                visit(parent)
        visiting.remove(experiment_id)
        visited.add(experiment_id)

    for experiment_id in experiments:
        visit(experiment_id)
    return issues


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def dataset_ids(project_root: Path) -> set[str]:
    try:
        records = _toml(project_root / "data" / "catalog.toml").get("datasets", [])
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    return {
        record["id"]
        for record in records
        if isinstance(record, dict) and isinstance(record.get("id"), str)
    }


def _check_dataset(project_root: Path, dataset: dict[str, Any]) -> list[Issue]:
    item_id = dataset.get("id")
    issues: list[Issue] = []
    required_fields = {"id", "category", "role", "required_for", "path", "kind", "required", "description"}
    missing = sorted(required_fields - dataset.keys())
    if missing or not isinstance(item_id, str) or not item_id:
        return [Issue("error", "invalid-dataset", str(item_id), ", ".join(missing))]
    relative = dataset.get("path")
    kind = dataset.get("kind")
    if not safe_relative_path(relative) or kind not in {"file", "directory"}:
        return [Issue("error", "invalid-dataset-location", item_id, repr(relative))]
    if not isinstance(dataset.get("required_for"), list) or not dataset["required_for"]:
        issues.append(Issue("error", "invalid-required-for", item_id, repr(dataset.get("required_for"))))
    if not isinstance(dataset.get("required"), bool):
        issues.append(Issue("error", "invalid-required-flag", item_id, repr(dataset.get("required"))))
    location = project_root / relative
    if not location.exists():
        severity = "error" if dataset.get("required") else "warning"
        issues.append(Issue(severity, "missing-dataset", item_id, relative))
        return issues
    if (kind == "file" and not location.is_file()) or (kind == "directory" and not location.is_dir()):
        issues.append(Issue("error", "wrong-dataset-kind", item_id, relative))
        return issues
    for check in dataset.get("checks", []):
        if not isinstance(check, dict):
            issues.append(Issue("error", "invalid-dataset-check", item_id, repr(check)))
            continue
        check_type = check.get("type")
        if check_type == "glob_count":
            pattern, expected = check.get("pattern"), check.get("expected")
            actual = sum(path.is_file() for path in location.glob(pattern)) if isinstance(pattern, str) else -1
            valid = isinstance(expected, int) and actual == expected
        elif check_type in {"file_size", "file_sha256"}:
            checked = location / str(check.get("path", ""))
            if check_type == "file_size":
                valid = checked.is_file() and checked.stat().st_size == check.get("bytes")
            else:
                valid = checked.is_file() and sha256_file(checked) == check.get("sha256")
        else:
            valid = False
        if not valid:
            issues.append(Issue("error", "dataset-check-failed", item_id, repr(check)))
    return issues


def validate_data(project_root: Path) -> tuple[int, list[Issue]]:
    try:
        catalog = _toml(project_root / "data" / "catalog.toml")
        sources = _toml(project_root / "data" / "sources.toml")
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return 0, [Issue("error", "invalid-data-manifest", "data", str(exc))]
    issues: list[Issue] = []
    if catalog.get("schema") != "data-catalog/v1" or sources.get("schema") != "data-sources/v1":
        issues.append(Issue("error", "invalid-data-schema", "data", "unsupported manifest schema"))
    datasets = catalog.get("datasets", [])
    source_records = sources.get("sources", [])
    if not isinstance(datasets, list) or not isinstance(source_records, list):
        return 0, [Issue("error", "invalid-data-records", "data", "records must be lists")]

    catalog_ids: set[str] = set()
    for dataset in datasets:
        if not isinstance(dataset, dict):
            issues.append(Issue("error", "invalid-dataset", "data/catalog.toml", repr(dataset)))
            continue
        item_id = dataset.get("id")
        if isinstance(item_id, str) and item_id in catalog_ids:
            issues.append(Issue("error", "duplicate-dataset-id", item_id, "duplicate ID"))
        elif isinstance(item_id, str):
            catalog_ids.add(item_id)
        issues.extend(_check_dataset(project_root, dataset))

    source_ids: set[str] = set()
    for source in source_records:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            issues.append(Issue("error", "invalid-source", "data/sources.toml", repr(source)))
            continue
        item_id = source["id"]
        if item_id in source_ids:
            issues.append(Issue("error", "duplicate-source-id", item_id, "duplicate ID"))
        source_ids.add(item_id)
        missing = sorted(SOURCE_FIELDS - source.keys())
        if missing:
            issues.append(Issue("error", "incomplete-source", item_id, ", ".join(missing)))
        for field in SOURCE_FIELDS - {"source_path", "source_url", "experiments"}:
            if field in source and (not isinstance(source[field], str) or not source[field].strip()):
                issues.append(Issue("error", "invalid-source-field", item_id, field))
        if source.get("source_path") is not None and not safe_relative_path(source["source_path"]):
            issues.append(Issue("error", "invalid-source-path", item_id, source["source_path"]))
        if not isinstance(source.get("experiments"), list):
            issues.append(Issue("error", "invalid-source-experiments", item_id, repr(source.get("experiments"))))
    if catalog_ids != source_ids:
        issues.append(
            Issue(
                "error",
                "data-source-coverage",
                "data",
                f"missing={sorted(catalog_ids - source_ids)} extra={sorted(source_ids - catalog_ids)}",
            )
        )
    return len(datasets), issues


def validate_memory(project_root: Path) -> list[Issue]:
    memory = project_root / "memory.md"
    if memory.is_file() and memory.read_text().strip():
        return []
    return [Issue("error", "missing-memory", "memory.md", str(memory))]


def validate_workspace(project_root: Path) -> ValidationReport:
    project_root = project_root.resolve()
    issues: list[Issue] = []
    if not (project_root / "research.md").is_file():
        issues.append(Issue("error", "missing-charter", "research.md", str(project_root / "research.md")))
    experiments, experiment_issues = load_experiments(project_root)
    dataset_count, data_issues = validate_data(project_root)
    metadata = {
        experiment_id: experiment.metadata
        for experiment_id, experiment in experiments.items()
    }
    contract_issues = validate_contracts(project_root, metadata, dataset_ids(project_root))
    run_count, run_issues = validate_runs(project_root, metadata, dataset_ids(project_root))
    issues.extend(experiment_issues)
    issues.extend(data_issues)
    issues.extend(Issue(**issue.as_dict()) for issue in contract_issues)
    issues.extend(Issue(**issue.as_dict()) for issue in run_issues)
    issues.extend(validate_memory(project_root))
    return ValidationReport(
        dataset_count,
        len(experiments),
        run_count,
        tuple(issues),
        runnable_experiments(project_root, metadata, dataset_ids(project_root)),
    )
