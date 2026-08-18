from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CONTRACT_SCHEMA = "autoresearch-contract/v1"
RUN_SCHEMA = "research-run/v1"
RUN_STATUSES = {"running", "completed", "failed", "interrupted", "invalid"}


@dataclass(frozen=True)
class KernelIssue:
    severity: str
    code: str
    subject: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return self.__dict__


class RunError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and not Path(value).is_absolute() and ".." not in Path(value).parts


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _issue(severity: str, code: str, subject: str, message: Any) -> KernelIssue:
    return KernelIssue(severity, code, subject, str(message))


def read_contract(
    project_root: Path,
    relative: str,
    *,
    require_active: bool = False,
) -> tuple[dict[str, Any], str, tuple[KernelIssue, ...]]:
    if not _safe(relative):
        return {}, "", (_issue("error", "invalid-contract-path", relative, "must be project-relative"),)
    path = project_root.resolve() / relative
    try:
        contract = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, "", (_issue("error", "invalid-contract", relative, exc),)
    if not isinstance(contract, dict):
        return {}, "", (_issue("error", "invalid-contract", relative, "expected an object"),)

    issues: list[KernelIssue] = []
    if contract.get("schema") != CONTRACT_SCHEMA:
        issues.append(_issue("error", "invalid-contract-schema", relative, contract.get("schema")))
    if not isinstance(contract.get("id"), str) or not contract["id"]:
        issues.append(_issue("error", "invalid-contract-id", relative, contract.get("id")))
    status = contract.get("status")
    if status not in {"draft", "active", "retired"}:
        issues.append(_issue("error", "invalid-contract-status", relative, status))
    elif status == "draft":
        issues.append(_issue("warning", "draft-contract", relative, "not approved for model runs"))
    if require_active and status != "active":
        issues.append(_issue("error", "inactive-contract", relative, status))
    if status != "active":
        return contract, sha256_file(path), tuple(issues)

    for field in ("owner", "approved_by", "approved_at"):
        if not isinstance(contract.get(field), str) or not contract[field].strip():
            issues.append(_issue("error", "incomplete-approval", relative, field))

    locked_paths: set[str] = set()
    protected = contract.get("protected_files")
    if not isinstance(protected, list) or not protected:
        issues.append(_issue("error", "missing-protected-files", relative, protected))
    else:
        for lock in protected:
            if not isinstance(lock, dict) or not _safe(lock.get("path")):
                issues.append(_issue("error", "invalid-protected-file", relative, lock))
                continue
            locked_paths.add(lock["path"])
            locked = project_root / lock["path"]
            if not locked.is_file() or sha256_file(locked) != lock.get("sha256"):
                issues.append(_issue("error", "protected-file-drift", relative, lock["path"]))

    evaluator = contract.get("evaluator")
    evaluator_path = evaluator.get("path") if isinstance(evaluator, dict) else None
    evaluator_argv = evaluator.get("argv") if isinstance(evaluator, dict) else None
    if (
        not _safe(evaluator_path)
        or not isinstance(evaluator_argv, list)
        or not evaluator_argv
        or not all(isinstance(value, str) and value for value in evaluator_argv)
        or evaluator_path not in evaluator_argv
    ):
        issues.append(_issue("error", "invalid-evaluator", relative, evaluator))
    else:
        evaluator_file = project_root / evaluator_path
        if not evaluator_file.is_file() or sha256_file(evaluator_file) != evaluator.get("sha256"):
            issues.append(_issue("error", "evaluator-drift", relative, evaluator_path))

    metrics = contract.get("metrics")
    if not isinstance(metrics, list) or not metrics or not all(
        isinstance(metric, dict)
        and isinstance(metric.get("id"), str)
        and bool(metric["id"])
        and isinstance(metric.get("direction"), str)
        and bool(metric["direction"])
        for metric in metrics
    ):
        issues.append(_issue("error", "invalid-metrics", relative, metrics))
    if not isinstance(contract.get("comparison_baselines"), list) or not contract["comparison_baselines"]:
        issues.append(_issue("error", "missing-comparison-baselines", relative, contract.get("comparison_baselines")))
    for field in ("development_evaluation", "promotion_evaluation"):
        if not isinstance(contract.get(field), dict) or not contract[field]:
            issues.append(_issue("error", "invalid-evaluation-scope", relative, field))

    figures = contract.get("figures")
    required_figure_fields = {
        "filename", "description", "generator", "format", "dimensions", "panels",
        "scope", "scale", "labels", "units", "required",
    }
    if not isinstance(figures, list) or not figures:
        issues.append(_issue("error", "missing-canonical-figures", relative, figures))
    else:
        seen: set[str] = set()
        for figure in figures:
            if not isinstance(figure, dict) or not required_figure_fields.issubset(figure):
                issues.append(_issue("error", "invalid-canonical-figure", relative, figure))
                continue
            filename = figure["filename"]
            dimensions = figure["dimensions"]
            if not _safe(filename) or filename in seen:
                issues.append(_issue("error", "invalid-figure-name", relative, filename))
            seen.add(filename)
            text_fields = ("description", "generator", "format", "scope", "scale", "units")
            if (
                not all(isinstance(figure[field], str) and figure[field] for field in text_fields)
                or not isinstance(figure["panels"], list)
                or not isinstance(figure["labels"], list)
                or not isinstance(figure["required"], bool)
            ):
                issues.append(_issue("error", "invalid-figure-fields", relative, filename))
            if not isinstance(dimensions, dict) or not all(
                isinstance(dimensions.get(axis), int) and dimensions[axis] > 0
                for axis in ("width", "height")
            ):
                issues.append(_issue("error", "invalid-figure-dimensions", relative, filename))
            if figure["generator"] != evaluator_path and figure["generator"] not in locked_paths:
                issues.append(_issue("error", "unprotected-figure-generator", relative, figure["generator"]))
    return contract, sha256_file(path), tuple(issues)


def validate_contracts(
    project_root: Path,
    experiments: Mapping[str, Mapping[str, Any]],
    known_inputs: set[str],
) -> tuple[KernelIssue, ...]:
    contracts_root = project_root / "evals" / "contracts"
    if not contracts_root.is_dir():
        return (_issue("error", "missing-contracts", "evals/contracts", contracts_root),)
    issues: list[KernelIssue] = []
    available: set[str] = set()
    ids: set[str] = set()
    for path in sorted(contracts_root.glob("*.json")):
        relative = str(path.relative_to(project_root))
        contract, _, found = read_contract(project_root, relative)
        issues.extend(found)
        available.add(relative)
        contract_id = contract.get("id")
        if isinstance(contract_id, str) and contract_id in ids:
            issues.append(_issue("error", "duplicate-contract-id", relative, contract_id))
        if isinstance(contract_id, str):
            ids.add(contract_id)
    if not available:
        issues.append(_issue("error", "missing-contract", "evals/contracts", "no JSON contract found"))
    if experiments and not _git(project_root, "rev-parse", "HEAD", text=True):
        issues.append(_issue("warning", "missing-git-commit", "git", "model runs require a commit"))
    elif experiments:
        untracked = _untracked(project_root)
        if untracked:
            issues.append(
                _issue("warning", "untracked-project-files", "git", f"{len(untracked)} file(s)")
            )
    for experiment_id, experiment in experiments.items():
        contract_path = experiment.get("contract")
        inputs = experiment.get("inputs")
        if contract_path not in available:
            issues.append(_issue("error", "unknown-experiment-contract", experiment_id, contract_path))
        if not isinstance(inputs, list) or not all(isinstance(value, str) for value in inputs):
            issues.append(_issue("error", "invalid-experiment-inputs", experiment_id, inputs))
        elif set(inputs) - known_inputs:
            issues.append(_issue("error", "unknown-experiment-inputs", experiment_id, sorted(set(inputs) - known_inputs)))
    return tuple(issues)


def runnable_experiments(
    project_root: Path,
    experiments: Mapping[str, Mapping[str, Any]],
    known_inputs: set[str],
) -> tuple[str, ...]:
    untracked = _untracked(project_root)
    if not _git(project_root, "rev-parse", "HEAD", text=True) or untracked is None or untracked:
        return ()
    runnable: list[str] = []
    for experiment_id, experiment in experiments.items():
        inputs = experiment.get("inputs")
        contract_path = experiment.get("contract")
        execution = experiment.get("execution")
        if (
            not isinstance(inputs, list)
            or not all(isinstance(value, str) for value in inputs)
            or set(inputs) - known_inputs
            or not isinstance(execution, Mapping)
            or not _safe(execution.get("adapter"))
            or not (project_root / execution["adapter"]).is_file()
            or not isinstance(execution.get("argv"), list)
            or not execution["argv"]
        ):
            continue
        _, _, issues = read_contract(project_root, contract_path, require_active=True)
        if not any(issue.severity == "error" for issue in issues):
            runnable.append(experiment_id)
    return tuple(sorted(runnable))


def _metric_exists(metrics: Mapping[str, Any], dotted: str) -> bool:
    value: Any = metrics
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False
        value = value[part]
    return value is not None


def _png_size(path: Path) -> tuple[int, int] | None:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", header[16:24])


def _collect_outputs(
    project_root: Path,
    run_root: Path,
    experiment_id: str,
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    try:
        metrics = _read_json(run_root / "metrics.json")
    except (OSError, json.JSONDecodeError) as exc:
        metrics = None
        failures.append(f"invalid metrics.json: {exc}")
    if not isinstance(metrics, dict) or not metrics:
        failures.append("trusted evaluator did not write nonempty metrics.json")
    else:
        for metric in contract["metrics"]:
            if metric.get("required", True) and not _metric_exists(metrics, metric["id"]):
                failures.append(f"required metric is missing: {metric['id']}")

    artifacts: list[dict[str, Any]] = []
    for directory, kind in ((run_root / "figures", "figure"), (run_root / "artifacts", "artifact")):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                failures.append(f"artifact may not be a symlink: {path.relative_to(run_root)}")
            elif path.is_file():
                artifacts.append(
                    {
                        "name": str(path.relative_to(directory)),
                        "kind": kind,
                        "path": str(path.relative_to(project_root)),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                        "producing_run": run_root.name,
                        "related_experiment": experiment_id,
                    }
                )
    registered = {artifact["path"] for artifact in artifacts}
    for figure in contract["figures"]:
        if not figure.get("required", True):
            continue
        path = run_root / "figures" / figure["filename"]
        if not path.is_file() or path.is_symlink():
            failures.append(f"required canonical figure is missing: {figure['filename']}")
            continue
        expected_format = figure["format"].lower().lstrip(".")
        if path.suffix.lower().lstrip(".") != expected_format:
            failures.append(f"canonical figure has the wrong format: {figure['filename']}")
        if expected_format == "png":
            expected_size = (figure["dimensions"]["width"], figure["dimensions"]["height"])
            if _png_size(path) != expected_size:
                failures.append(f"canonical figure has the wrong dimensions: {figure['filename']}")
        if str(path.relative_to(project_root)) not in registered:
            failures.append(f"canonical figure was not registered: {figure['filename']}")
    return artifacts, failures


def validate_runs(
    project_root: Path,
    experiments: Mapping[str, Mapping[str, Any]],
    known_inputs: set[str],
) -> tuple[int, tuple[KernelIssue, ...]]:
    issues: list[KernelIssue] = []
    directories: list[Path] = []
    for experiment_id in sorted(experiments):
        runs_root = project_root / "research" / "experiments" / experiment_id / "runs"
        if not runs_root.is_dir():
            continue
        directories.extend(sorted(path for path in runs_root.iterdir() if path.is_dir()))
    seen_run_ids: set[str] = set()
    for directory in directories:
        try:
            run = _read_json(directory / "run.json")
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(_issue("error", "invalid-run", directory.name, exc))
            continue
        run_id = run.get("id") if isinstance(run, dict) else None
        if (
            not isinstance(run, dict)
            or run.get("schema") != RUN_SCHEMA
            or run_id != directory.name
            or run.get("status") not in RUN_STATUSES
            or run.get("experiment") not in experiments
            or directory.parents[1].name != run.get("experiment")
        ):
            issues.append(_issue("error", "invalid-run-record", directory.name, run))
            continue
        if run_id in seen_run_ids:
            issues.append(_issue("error", "duplicate-run-id", run_id, directory))
        seen_run_ids.add(run_id)
        inputs = run.get("inputs")
        if not isinstance(inputs, list) or not all(isinstance(value, str) for value in inputs) or set(inputs) - known_inputs:
            issues.append(_issue("error", "invalid-run-inputs", run_id, inputs))
        commands = run.get("commands")
        if run["status"] != "running" and (not isinstance(commands, list) or not commands):
            issues.append(_issue("error", "missing-run-command", run_id, commands))
        for command in commands or []:
            for stream in ("stdout", "stderr"):
                relative = command.get(stream) if isinstance(command, dict) else None
                if not _safe(relative) or not (project_root / relative).is_file():
                    issues.append(_issue("error", "missing-run-log", run_id, relative))
        events = directory / "events.jsonl"
        if not events.is_file():
            issues.append(_issue("error", "missing-run-events", run_id, events))
        else:
            for number, line in enumerate(events.read_text().splitlines(), start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    issues.append(_issue("error", "invalid-run-event", run_id, f"line {number}: {exc}"))
                    continue
                if (
                    not isinstance(event, dict)
                    or not isinstance(event.get("timestamp"), str)
                    or not isinstance(event.get("event"), str)
                    or not isinstance(event.get("metadata"), dict)
                ):
                    issues.append(_issue("error", "invalid-run-event", run_id, f"line {number}"))

        contract_ref = run.get("contract")
        snapshot = directory / "contract.json"
        expected_contract = contract_ref.get("sha256") if isinstance(contract_ref, dict) else None
        live_relative = contract_ref.get("path") if isinstance(contract_ref, dict) else None
        live = project_root / live_relative if _safe(live_relative) else None
        if (
            not snapshot.is_file()
            or not isinstance(expected_contract, str)
            or sha256_file(snapshot) != expected_contract
            or live is None
            or not live.is_file()
            or sha256_file(live) != expected_contract
        ):
            issues.append(_issue("error", "run-contract-drift", run_id, live_relative))
        for lock in run.get("protected_files", []):
            path = project_root / lock.get("path", "") if isinstance(lock, dict) and _safe(lock.get("path")) else None
            if path is None or not path.is_file() or sha256_file(path) != lock.get("sha256"):
                issues.append(_issue("error", "run-lock-drift", run_id, lock))

        if run["status"] != "completed":
            continue
        metrics_path = directory / "metrics.json"
        if not metrics_path.is_file() or sha256_file(metrics_path) != run.get("official_metrics_sha256"):
            issues.append(_issue("error", "official-metrics-drift", run_id, metrics_path))
        try:
            manifest = _read_json(directory / "artifacts.json")
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(_issue("error", "invalid-artifacts", run_id, exc))
            continue
        if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), list):
            issues.append(_issue("error", "invalid-artifacts", run_id, manifest))
            continue
        for artifact in manifest["artifacts"]:
            path = project_root / artifact.get("path", "") if isinstance(artifact, dict) and _safe(artifact.get("path")) else None
            if path is None or not path.is_file() or path.is_symlink() or sha256_file(path) != artifact.get("sha256"):
                issues.append(_issue("error", "artifact-drift", run_id, artifact))
    return len(directories), tuple(issues)


def _git(project_root: Path, *args: str, text: bool = False) -> bytes | str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode(errors="replace").strip() if text else result.stdout


def _untracked(project_root: Path) -> tuple[str, ...] | None:
    output = _git(project_root, "ls-files", "--others", "--exclude-standard", text=True)
    if not isinstance(output, str):
        return None
    return tuple(
        path
        for path in output.splitlines()
        if path
        and not (path.startswith("research/experiments/") and "/runs/" in path)
    )


def _invoke(
    project_root: Path,
    run_root: Path,
    argv: Sequence[str],
    role: str,
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], bool]:
    stdout_path = run_root / "logs" / f"{role}.stdout.log"
    stderr_path = run_root / "logs" / f"{role}.stderr.log"
    started = _now()
    interrupted = False
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            exit_code = subprocess.run(
                list(argv), cwd=project_root, env=dict(environment), stdout=stdout, stderr=stderr, check=False
            ).returncode
    except KeyboardInterrupt:
        exit_code, interrupted = 130, True
        stdout_path.touch(exist_ok=True)
        stderr_path.write_text("Interrupted by user.\n")
    except OSError as exc:
        exit_code = 127
        stdout_path.touch(exist_ok=True)
        stderr_path.write_text(str(exc) + "\n")
    return {
        "role": role,
        "argv": list(argv),
        "started_at": started,
        "completed_at": _now(),
        "exit_code": exit_code,
        "stdout": str(stdout_path.relative_to(project_root)),
        "stderr": str(stderr_path.relative_to(project_root)),
    }, interrupted


def _locked(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    records = [dict(record) for record in contract["protected_files"]]
    evaluator = {
        "path": contract["evaluator"]["path"],
        "sha256": contract["evaluator"]["sha256"],
    }
    if not any(record["path"] == evaluator["path"] for record in records):
        records.append(evaluator)
    return records


def _unchanged(project_root: Path, contract_path: Path, contract_hash: str, locks: Sequence[Mapping[str, str]]) -> bool:
    return contract_path.is_file() and sha256_file(contract_path) == contract_hash and all(
        _safe(lock.get("path"))
        and (project_root / lock["path"]).is_file()
        and sha256_file(project_root / lock["path"]) == lock.get("sha256")
        for lock in locks
    )


def _record_event(run_root: Path, event_name: str, **metadata: Any) -> None:
    path = run_root / "events.jsonl"
    event = {
        "timestamp": _now(),
        "event": event_name,
        "metadata": metadata,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def run_experiment(
    project_root: Path,
    *,
    experiment_id: str,
    experiment: Mapping[str, Any],
    known_inputs: set[str],
    command: Sequence[str],
    final_check: Callable[[], tuple[bool, str]] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    inputs = experiment.get("inputs")
    execution = experiment.get("execution")
    if experiment.get("schema") != "autoresearch-experiment/v1" or experiment.get("id") != experiment_id:
        raise RunError(f"invalid experiment record: {experiment_id}")
    if not command:
        raise RunError("candidate command cannot be empty")
    if not isinstance(inputs, list) or not all(isinstance(value, str) for value in inputs) or set(inputs) - known_inputs:
        raise RunError(f"experiment {experiment_id} has invalid inputs")
    if not isinstance(execution, Mapping) or execution.get("mode") not in {"mechanistic", "simulation", "hybrid"}:
        raise RunError(f"experiment {experiment_id} has invalid execution metadata")
    adapter = execution.get("adapter")
    if not _safe(adapter) or not (project_root / adapter).is_file():
        raise RunError(f"experiment {experiment_id} has no executable adapter")
    contract_relative = experiment.get("contract")
    contract, contract_hash, issues = read_contract(project_root, contract_relative, require_active=True)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise RunError("experiment contract is not runnable: " + "; ".join(issue.code for issue in errors))

    revision = _git(project_root, "rev-parse", "HEAD", text=True)
    untracked_before = _untracked(project_root)
    diff_before = _git(project_root, "diff", "HEAD", "--binary")
    if not isinstance(revision, str) or not revision:
        raise RunError("model runs require a Git commit")
    if untracked_before:
        raise RunError("model runs require project files to be tracked or ignored: " + ", ".join(untracked_before))
    if untracked_before is None or not isinstance(diff_before, bytes):
        raise RunError("cannot capture Git state")

    run_id = f"run.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.{uuid.uuid4().hex[:8]}"
    run_root = project_root / "research" / "experiments" / experiment_id / "runs" / run_id
    run_root.mkdir(parents=True)
    for name in ("artifacts", "figures", "logs", "work"):
        (run_root / name).mkdir()
    contract_path = project_root / contract_relative
    (run_root / "contract.json").write_bytes(contract_path.read_bytes())
    (run_root / "logs" / "worktree-before.diff").write_bytes(diff_before)
    (run_root / "logs" / "git-status-before.txt").write_text(str(_git(project_root, "status", "--short", text=True) or "") + "\n")
    lockfiles = {
        name: sha256_file(project_root / name)
        for name in ("uv.lock", "pyproject.toml", "requirements.txt", "poetry.lock", "environment.yml")
        if (project_root / name).is_file()
    }
    environment_record = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "lockfiles": lockfiles,
    }
    _write_json(run_root / "logs" / "environment.json", environment_record)
    _write_json(run_root / "metrics.json", {})
    _write_json(run_root / "artifacts.json", {"schema": "research-artifacts/v1", "artifacts": []})
    (run_root / "events.jsonl").touch()

    locks = _locked(contract)
    run: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "id": run_id,
        "experiment": experiment_id,
        "experiment_kind": experiment.get("kind"),
        "execution": dict(execution),
        "search": experiment.get("search"),
        "status": "running",
        "code_revision": revision,
        "dirty_worktree": bool(diff_before),
        "changed_files": str(_git(project_root, "diff", "HEAD", "--name-only", text=True) or "").splitlines(),
        "working_directory": str(project_root),
        "commands": [],
        "inputs": list(inputs),
        "contract": {"id": contract["id"], "path": contract_relative, "sha256": contract_hash},
        "protected_files": locks,
        "official_metrics_sha256": None,
        "started_at": _now(),
        "completed_at": None,
        "failure": None,
    }
    _write_json(run_root / "run.json", run)
    _record_event(run_root, "run-started", run_id=run_id, experiment_id=experiment_id, execution_mode=execution.get("mode"), tool=execution.get("tool"))
    environment = os.environ.copy()
    environment.update(
        AUTORESEARCH_RUN_ID=run_id,
        AUTORESEARCH_RUN_ROOT=str(run_root),
        AUTORESEARCH_PROJECT_ROOT=str(project_root),
    )

    _record_event(run_root, "candidate-started", argv=list(command))
    candidate, interrupted = _invoke(project_root, run_root, command, "candidate", environment)
    run["commands"].append(candidate)
    _record_event(run_root, "candidate-finished", exit_code=candidate["exit_code"], interrupted=interrupted)
    changed = _git(project_root, "diff", "HEAD", "--binary") != diff_before or _untracked(project_root) != untracked_before
    if interrupted:
        run["status"], run["failure"] = "interrupted", "candidate command was interrupted"
    elif candidate["exit_code"] != 0:
        run["status"], run["failure"] = "failed", f"candidate command exited with {candidate['exit_code']}"
    elif changed:
        run["status"], run["failure"] = "invalid", "candidate changed project files during execution"
    elif not _unchanged(project_root, contract_path, contract_hash, locks):
        run["status"], run["failure"] = "invalid", "candidate changed locked evaluation files"
    else:
        if execution.get("tool") == "optuna":
            required_search_artifacts = (
                run_root / "artifacts" / "optuna-study.json",
                run_root / "artifacts" / "optuna-trials.jsonl",
                run_root / "artifacts" / "selected-parameters.json",
            )
            missing = [path.name for path in required_search_artifacts if not path.is_file()]
            if missing:
                run["status"], run["failure"] = "invalid", "Optuna execution did not preserve required study artifacts: " + ", ".join(missing)
                _record_event(run_root, "search-artifacts-missing", files=missing)

    if run["status"] == "running":
        _write_json(run_root / "metrics.json", {})
        shutil.rmtree(run_root / "figures")
        (run_root / "figures").mkdir()
        replacements = {
            "{python}": sys.executable,
            "{project_root}": str(project_root),
            "{run_root}": str(run_root),
            "{run_id}": run_id,
        }
        evaluator_argv = [replacements.get(value, value) for value in contract["evaluator"]["argv"]]
        _record_event(run_root, "evaluation-started", argv=evaluator_argv)
        evaluator, interrupted = _invoke(project_root, run_root, evaluator_argv, "evaluator", environment)
        run["commands"].append(evaluator)
        _record_event(run_root, "evaluation-finished", exit_code=evaluator["exit_code"], interrupted=interrupted)
        changed = _git(project_root, "diff", "HEAD", "--binary") != diff_before or _untracked(project_root) != untracked_before
        if interrupted:
            run["status"], run["failure"] = "interrupted", "trusted evaluator was interrupted"
        elif evaluator["exit_code"] != 0:
            run["status"], run["failure"] = "failed", f"trusted evaluator exited with {evaluator['exit_code']}"
        elif changed or not _unchanged(project_root, contract_path, contract_hash, locks):
            run["status"], run["failure"] = "invalid", "evaluation changed project or locked files"
        else:
            artifacts, failures = _collect_outputs(project_root, run_root, experiment_id, contract)
            _write_json(run_root / "artifacts.json", {"schema": "research-artifacts/v1", "artifacts": artifacts})
            if failures:
                run["status"], run["failure"] = "invalid", "; ".join(failures)
            else:
                run["official_metrics_sha256"] = sha256_file(run_root / "metrics.json")
                run["status"] = "completed"

    (run_root / "logs" / "worktree-after.diff").write_bytes(_git(project_root, "diff", "HEAD", "--binary") or b"")
    (run_root / "logs" / "git-status-after.txt").write_text(str(_git(project_root, "status", "--short", text=True) or "") + "\n")
    run["completed_at"] = _now()
    _write_json(run_root / "run.json", run)
    if run["status"] == "completed" and final_check is not None:
        ok, detail = final_check()
        if not ok:
            run["status"], run["failure"] = "invalid", f"workspace validation failed after execution: {detail}"
            _write_json(run_root / "run.json", run)
    _record_event(run_root, "run-finished", status=run["status"], failure=run.get("failure"))
    return run
