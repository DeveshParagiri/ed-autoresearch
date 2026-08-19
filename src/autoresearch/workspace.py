from __future__ import annotations

from pathlib import Path
from typing import Any

from .catalog import load_catalog, validate_catalog, validate_sources
from .experiments import load_experiments, validate_experiments
from .runner import runnable_experiments, validate_contracts, validate_runs


def experiments(project_root: Path) -> dict[str, dict[str, Any]]:
    loaded, _ = load_experiments(project_root)
    return {experiment_id: experiment.metadata for experiment_id, experiment in loaded.items()}


def input_ids(project_root: Path) -> set[str]:
    return {
        record["id"]
        for record in load_catalog(project_root).get("datasets", [])
        if isinstance(record, dict) and isinstance(record.get("id"), str)
    }


def _memory_issues(project_root: Path) -> list[dict[str, str]]:
    memory = project_root / "memory.md"
    if memory.is_file() and memory.read_text().strip():
        return []
    return [
        {
            "severity": "error",
            "code": "missing-memory",
            "subject": "memory.md",
            "message": str(memory),
        }
    ]


def status(project_root: Path) -> dict[str, Any]:
    data = validate_catalog(project_root)
    source_issues = validate_sources(project_root)
    research = validate_experiments(project_root)
    experiment_records = experiments(project_root)
    inputs = input_ids(project_root)
    contract_issues = validate_contracts(project_root, experiment_records, inputs)
    run_count, run_issues = validate_runs(project_root, experiment_records, inputs)
    issues = [
        {
            "severity": issue.severity,
            "code": issue.code,
            "subject": issue.item_id,
            "message": issue.message,
        }
        for issue in (*data.issues, *source_issues, *research.issues)
    ]
    issues.extend(issue.as_dict() for issue in (*contract_issues, *run_issues))
    issues.extend(_memory_issues(project_root))
    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "status": "ok" if errors == 0 else "error",
        "datasets": data.dataset_count,
        "framings": research.framing_count,
        "experiments": research.experiment_count,
        "runs": run_count,
        "runnable_experiments": list(
            runnable_experiments(project_root, experiment_records, inputs)
        ),
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
    }


def final_check(project_root: Path) -> tuple[bool, str]:
    report = status(project_root)
    errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
    detail = "; ".join(f"{issue['code']}: {issue['message']}" for issue in errors[:5])
    return not errors, detail
