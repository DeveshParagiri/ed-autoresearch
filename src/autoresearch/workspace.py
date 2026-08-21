from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .catalog import load_catalog, validate_catalog, validate_sources


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def issue(severity: str, code: str, subject: str, message: str) -> dict[str, str]:
    return {
        "severity": severity,
        "code": code,
        "subject": subject,
        "message": message,
    }


def status(project_root: Path) -> dict[str, Any]:
    data = validate_catalog(project_root)
    source_issues = validate_sources(project_root)
    issues = [
        issue(item.severity, item.code, item.item_id, item.message)
        for item in (*data.issues, *source_issues)
    ]

    research = project_root / "research.md"
    if not research.is_file() or not research.read_text().strip():
        issues.append(issue("error", "missing-research-plan", "research.md", str(research)))

    registry_path = project_root / "model" / "registry.toml"
    models: list[dict[str, Any]] = []
    if not registry_path.is_file():
        issues.append(issue("error", "missing-model-registry", "model/registry.toml", str(registry_path)))
    else:
        registry = tomllib.loads(registry_path.read_text())
        models = registry.get("models", [])
        for model in models:
            for value in model.get("parameter_files", []):
                path = project_root / value
                if not path.is_file():
                    issues.append(issue("error", "missing-model-parameters", model.get("id", "unknown"), value))

    contracts = sorted((project_root / "evals").glob("*.json"))
    if len(contracts) != 1:
        issues.append(issue("error", "evaluation-contract-count", "evals", f"expected 1 JSON contract, found {len(contracts)}"))
    elif contracts:
        contract = json.loads(contracts[0].read_text())
        evaluator = project_root / contract["evaluator"]["path"]
        if not evaluator.is_file():
            issues.append(issue("error", "missing-evaluator", "evaluation", str(evaluator)))
        elif sha256(evaluator) != contract["evaluator"]["sha256"]:
            issues.append(issue("error", "evaluator-hash", "evaluation", str(evaluator)))
        for record in contract.get("protected_files", []):
            path = project_root / record["path"]
            if not path.is_file():
                issues.append(issue("error", "missing-evaluation-input", record["path"], str(path)))
            elif sha256(path) != record["sha256"]:
                issues.append(issue("error", "evaluation-input-hash", record["path"], str(path)))

    errors = sum(item["severity"] == "error" for item in issues)
    warnings = sum(item["severity"] == "warning" for item in issues)
    return {
        "status": "ok" if errors == 0 else "error",
        "datasets": data.dataset_count,
        "models": len(models),
        "contracts": len(contracts),
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
    }
