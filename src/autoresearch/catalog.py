from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility for the existing ED environment.
    import tomli as tomllib


VALID_DATASET_ROLES = {
    "baseline-input",
    "benchmark-configuration",
    "candidate-input",
    "comparison-context",
    "coupled-context",
    "coupled-input",
    "derived-candidate",
    "historical-benchmark",
    "native-model-baseline",
    "primary-benchmark",
    "reconstruction-source",
    "reference-tool",
    "supporting-source",
}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    item_id: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "item_id": self.item_id,
            "message": self.message,
        }


@dataclass(frozen=True)
class CatalogValidation:
    dataset_count: int
    present_count: int
    optional_missing_count: int
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
            "datasets": self.dataset_count,
            "present": self.present_count,
            "optional_missing": self.optional_missing_count,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def _sha256(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(project_root: Path) -> dict[str, Any]:
    catalog_file = project_root / "data" / "catalog.toml"
    with catalog_file.open("rb") as handle:
        catalog = tomllib.load(handle)
    if catalog.get("schema") != "data-catalog/v1":
        raise ValueError(f"unsupported data catalog schema in {catalog_file}")
    return catalog


def _missing_issue(item_id: str, required: bool, location: Path) -> ValidationIssue:
    return ValidationIssue(
        severity="error" if required else "warning",
        code="missing-required" if required else "missing-optional",
        item_id=item_id,
        message=f"path does not resolve: {location}",
    )


def validate_catalog(project_root: Path) -> CatalogValidation:
    project_root = project_root.resolve()
    catalog = load_catalog(project_root)
    datasets = catalog.get("datasets", [])
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    present_count = 0
    optional_missing_count = 0

    for dataset in datasets:
        item_id = dataset.get("id")
        if not isinstance(item_id, str) or not item_id:
            issues.append(
                ValidationIssue("error", "invalid-id", "<unknown>", "dataset ID is missing")
            )
            continue
        if item_id in seen_ids:
            issues.append(
                ValidationIssue("error", "duplicate-id", item_id, "dataset ID is duplicated")
            )
            continue
        seen_ids.add(item_id)

        role = dataset.get("role")
        if role not in VALID_DATASET_ROLES:
            issues.append(
                ValidationIssue(
                    "error", "invalid-role", item_id, f"unsupported dataset role: {role!r}"
                )
            )
        required_for = dataset.get("required_for")
        if (
            not isinstance(required_for, list)
            or not required_for
            or not all(isinstance(profile, str) and profile for profile in required_for)
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid-required-for",
                    item_id,
                    "required_for must be a nonempty list of profile names",
                )
            )

        relative_location = dataset.get("path")
        required = bool(dataset.get("required", False))
        if not isinstance(relative_location, str) or not relative_location:
            issues.append(
                ValidationIssue("error", "invalid-path", item_id, "dataset path is missing")
            )
            continue
        location = project_root / relative_location

        if not location.exists():
            if location.is_symlink():
                issues.append(
                    ValidationIssue(
                        "error" if required else "warning",
                        "broken-link",
                        item_id,
                        f"local link target does not resolve: {location}",
                    )
                )
            else:
                issues.append(_missing_issue(item_id, required, location))
            if not required:
                optional_missing_count += 1
            continue

        present_count += 1
        expected_kind = dataset.get("kind")
        if expected_kind == "file" and not location.is_file():
            issues.append(
                ValidationIssue("error", "wrong-kind", item_id, f"expected a file: {location}")
            )
            continue
        if expected_kind == "directory" and not location.is_dir():
            issues.append(
                ValidationIssue("error", "wrong-kind", item_id, f"expected a directory: {location}")
            )
            continue
        if expected_kind not in {"file", "directory"}:
            issues.append(
                ValidationIssue(
                    "error", "invalid-kind", item_id, f"unsupported kind: {expected_kind!r}"
                )
            )
            continue

        minimum_bytes = dataset.get("min_bytes")
        if expected_kind == "file" and isinstance(minimum_bytes, int):
            actual_bytes = location.stat().st_size
            if actual_bytes < minimum_bytes:
                issues.append(
                    ValidationIssue(
                        "error",
                        "file-too-small",
                        item_id,
                        f"expected at least {minimum_bytes} bytes, found {actual_bytes}: {location}",
                    )
                )

        for check in dataset.get("checks", []):
            check_type = check.get("type")
            if check_type == "glob_count":
                pattern = check.get("pattern")
                expected = check.get("expected")
                if not isinstance(pattern, str) or not isinstance(expected, int):
                    issues.append(
                        ValidationIssue(
                            "error", "invalid-check", item_id, "glob_count check is malformed"
                        )
                    )
                    continue
                actual = sum(1 for match in location.glob(pattern) if match.is_file())
                if actual != expected:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "count-mismatch",
                            item_id,
                            f"pattern {pattern!r} expected {expected} files, found {actual}",
                        )
                    )
            elif check_type == "file_sha256":
                relative_file = check.get("path")
                expected_hash = check.get("sha256")
                if not isinstance(relative_file, str) or not isinstance(expected_hash, str):
                    issues.append(
                        ValidationIssue(
                            "error", "invalid-check", item_id, "file_sha256 check is malformed"
                        )
                    )
                    continue
                checked_file = location / relative_file
                if not checked_file.is_file():
                    issues.append(
                        ValidationIssue(
                            "error",
                            "missing-check-file",
                            item_id,
                            f"checksum target is missing: {checked_file}",
                        )
                    )
                    continue
                actual_hash = _sha256(checked_file)
                if actual_hash != expected_hash:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "checksum-mismatch",
                            item_id,
                            f"checksum mismatch for {checked_file}",
                        )
                    )
            elif check_type == "file_size":
                relative_file = check.get("path")
                expected_bytes = check.get("bytes")
                if not isinstance(relative_file, str) or not isinstance(expected_bytes, int):
                    issues.append(
                        ValidationIssue(
                            "error", "invalid-check", item_id, "file_size check is malformed"
                        )
                    )
                    continue
                checked_file = location / relative_file
                if not checked_file.is_file():
                    issues.append(
                        ValidationIssue(
                            "error",
                            "missing-check-file",
                            item_id,
                            f"size target is missing: {checked_file}",
                        )
                    )
                    continue
                actual_bytes = checked_file.stat().st_size
                if actual_bytes != expected_bytes:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "size-mismatch",
                            item_id,
                            f"expected {expected_bytes} bytes, found {actual_bytes}: {checked_file}",
                        )
                    )
            else:
                issues.append(
                    ValidationIssue(
                        "error", "unknown-check", item_id, f"unsupported check type: {check_type!r}"
                    )
                )

    return CatalogValidation(
        dataset_count=len(datasets),
        present_count=present_count,
        optional_missing_count=optional_missing_count,
        issues=tuple(issues),
    )
