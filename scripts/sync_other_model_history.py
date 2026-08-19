#!/usr/bin/env python3
"""Materialize every committed ED-Fire model artifact from the history bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model" / "other-models" / "upstream" / "ed-autoresearch.bundle"
OUTPUT = ROOT / "model" / "other-models" / "commit-artifacts"

AUDITED_REFS = {
    "main": (
        "refs/remotes/origin/main",
        "222b8569268a65566de0073a5f84dcbb2028da12",
    ),
    "coupled-refit-gfed5": (
        "refs/remotes/origin/coupled-refit-gfed5",
        "11ee71418e597e977a4d49f6fda166e20c098e9f",
    ),
    "modelD-paper-params": (
        "refs/remotes/origin/modelD-paper-params",
        "32d283ff595d6653dfd84c076f418a82074b0d26",
    ),
}

MANIFEST_FIELDS = [
    "source_path",
    "git_blob",
    "archive_path",
    "sha256",
    "bytes",
    "git_mode",
    "artifact_kind",
    "model_family",
    "stage",
    "declared_model",
    "parameter_count",
    "record_count",
    "first_seen_commit",
    "first_seen_date",
    "last_seen_commit",
    "last_seen_date",
    "commit_count",
    "history_refs",
    "tip_refs",
]

ATTEMPT_FIELDS = [
    "attempt_id",
    "model_family",
    "stage",
    "artifact_kind",
    "source_path",
    "git_blob",
    "archive_path",
    "declared_model",
    "parameter_count",
    "record_count",
    "first_seen_commit",
    "first_seen_date",
    "history_refs",
]


@dataclass(frozen=True)
class TreeEntry:
    mode: str
    blob: str


@dataclass
class ArtifactVersion:
    source_path: str
    mode: str
    blob: str
    commits: list[str]


def run_git(repository: Path, *args: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        capture_output=True,
        check=False,
        text=text,
    )
    if completed.returncode != 0:
        stderr = completed.stderr if text else completed.stderr.decode(errors="replace")
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return completed.stdout


def selected_model_artifact(path: str) -> bool:
    return path.startswith(
        (
            "models/",
            "configs/",
            "data_human/",
            "patches/",
            "HPC_AFRICA_HANDOFF/reference/",
        )
    )


def parse_tree(raw: bytes) -> dict[str, TreeEntry]:
    tree: dict[str, TreeEntry] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode, object_type, blob = metadata.decode().split()
        path = encoded_path.decode(errors="surrogateescape")
        if object_type == "blob" and selected_model_artifact(path):
            tree[path] = TreeEntry(mode=mode, blob=blob)
    return tree


def artifact_kind(path: str) -> str:
    source = PurePosixPath(path)
    name = source.name.lower()
    if path.startswith("patches/") or source.suffix.lower() in {".cc", ".h"}:
        return "implementation"
    if source.suffix.lower() == ".md":
        return "documentation"
    if "combustion" in path or "betas" in name:
        return "combustion-parameters"
    if path.startswith("models/paper/"):
        return "paper-model"
    if name.startswith("topk.") or "summary" in name or "shapley" in name:
        return "optimization-evidence"
    if path == "configs/optimized_params.json":
        return "optimized-configuration"
    if path.startswith("configs/"):
        return "model-configuration"
    if path.startswith("data_human/"):
        if source.suffix.lower() == ".json":
            return "human-driver-parameters"
        return "human-driver-data"
    if path.startswith("HPC_AFRICA_HANDOFF/"):
        if source.suffix.lower() == ".json":
            return "coupled-reference"
        return "coupled-runtime-support"
    return "model-parameters"


def model_family(path: str) -> str:
    lower = path.lower()
    name = PurePosixPath(path).name.lower()
    if path.startswith("models/A/") or "modela" in name:
        return "A"
    if path.startswith("models/B/") or "modelb" in name:
        return "B"
    if "gtrop" in lower:
        return "I"
    if re.search(r"(?:params|topk)\.h(?:\.|$)", name):
        return "H"
    if re.search(r"(?:params|topk)\.g_", name):
        return "G"
    if "paperd" in lower or path.startswith("models/paper/D"):
        return "D"
    if path.startswith("models/paper/E"):
        return "E"
    if "couplede_gdp" in lower or "gdp_regional" in lower:
        return "F"
    if "couplede_cure" in lower or "regional_cure" in lower:
        return "curing"
    if "combustion" in lower or "betas" in name:
        return "combustion"
    if "couplede" in lower or "coupledfw" in lower or "hurtt-betas" in lower:
        return "coupled-E"
    if path.startswith("models/paper/C") or path.startswith("models/C/"):
        return "C-development"
    if "shapley" in lower:
        return "A-B-C-attribution"
    if path.startswith("HPC_AFRICA_HANDOFF/"):
        return "coupled-handoff"
    return "initial-fire-model"


def artifact_stage(path: str, kind: str) -> str:
    name = PurePosixPath(path).name.lower()
    if name.startswith("topk."):
        return "ranked-candidates"
    if "summary" in name:
        return "optimizer-summary"
    if "shapley" in name:
        return "attribution"
    if re.search(r"\.k\d+\.json$", name):
        return "optimizer-checkpoint"
    if "pre-" in name:
        return "predecessor"
    if "cell" in name:
        return "held-out-cells"
    if re.search(r"(?:^|[._])ho(?:[._]|$)", name):
        return "held-out-years"
    if "assembly" in name:
        return "assembly"
    if kind in {"implementation", "documentation"}:
        return "definition"
    if kind == "optimized-configuration":
        return "optimization-result"
    if kind == "model-configuration":
        return "configuration"
    if kind == "human-driver-data":
        return "input-evidence"
    if kind == "coupled-runtime-support":
        return "runtime-support"
    return "committed-parameter-set"


def json_metadata(data: bytes) -> tuple[str, str, str]:
    try:
        payload: Any = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "", "", ""

    declared_model = ""
    parameter_count: int | str = ""
    record_count: int | str = ""
    if isinstance(payload, dict):
        if isinstance(payload.get("model"), str):
            declared_model = payload["model"]
        if isinstance(payload.get("n_params"), int):
            parameter_count = payload["n_params"]
        elif isinstance(payload.get("params"), dict):
            parameter_count = len(payload["params"])
        elif isinstance(payload.get("best_params"), dict):
            parameter_count = len(payload["best_params"])
        record_count = len(payload)
    elif isinstance(payload, list):
        record_count = len(payload)
    return declared_model, str(parameter_count), str(record_count)


def archive_path(source_path: str, blob: str) -> str:
    source = PurePosixPath(source_path)
    suffix = source.suffix
    base = source.name[: -len(suffix)] if suffix else source.name
    filename = f"{base}.{blob}{suffix}"
    relative = PurePosixPath("model/other-models/commit-artifacts/by-path")
    return str(relative / source.parent / filename)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def csv_bytes(fields: list[str], rows: list[dict[str, Any]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def collect_history(repository: Path) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, bytes], dict[str, Any]
]:
    refs = [ref for ref, _tip in AUDITED_REFS.values()]
    actual_tips = {
        name: str(run_git(repository, "rev-parse", ref)).strip()
        for name, (ref, _tip) in AUDITED_REFS.items()
    }
    expected_tips = {name: tip for name, (_ref, tip) in AUDITED_REFS.items()}
    if actual_tips != expected_tips:
        raise RuntimeError(
            "history bundle refs do not match the pinned tips: "
            f"expected {expected_tips}, found {actual_tips}"
        )

    commit_order = str(
        run_git(repository, "rev-list", "--reverse", "--topo-order", *refs)
    ).splitlines()
    commit_dates = {
        commit: str(run_git(repository, "show", "-s", "--format=%aI", commit)).strip()
        for commit in commit_order
    }
    ref_commits = {
        name: set(str(run_git(repository, "rev-list", ref)).splitlines())
        for name, (ref, _tip) in AUDITED_REFS.items()
    }

    versions: dict[tuple[str, str], ArtifactVersion] = {}
    tip_trees: dict[str, dict[str, TreeEntry]] = {}
    for name, (ref, _tip) in AUDITED_REFS.items():
        tip_trees[name] = parse_tree(
            run_git(repository, "ls-tree", "-r", "-z", ref, text=False)
        )

    for commit in commit_order:
        tree = parse_tree(
            run_git(repository, "ls-tree", "-r", "-z", commit, text=False)
        )
        for source_path, entry in tree.items():
            key = (source_path, entry.blob)
            if key not in versions:
                versions[key] = ArtifactVersion(
                    source_path=source_path,
                    mode=entry.mode,
                    blob=entry.blob,
                    commits=[],
                )
            versions[key].commits.append(commit)

    blobs: dict[str, bytes] = {}
    manifest: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for key in sorted(versions):
        version = versions[key]
        if version.blob not in blobs:
            blobs[version.blob] = run_git(
                repository, "cat-file", "blob", version.blob, text=False
            )
        data = blobs[version.blob]
        if git_blob_sha1(data) != version.blob:
            raise RuntimeError(f"Git blob verification failed for {version.blob}")

        kind = artifact_kind(version.source_path)
        family = model_family(version.source_path)
        stage = artifact_stage(version.source_path, kind)
        declared_model, parameter_count, record_count = json_metadata(data)
        first_commit = version.commits[0]
        last_commit = version.commits[-1]
        history_refs = [
            name
            for name in AUDITED_REFS
            if any(commit in ref_commits[name] for commit in version.commits)
        ]
        tip_refs = [
            name
            for name in AUDITED_REFS
            if tip_trees[name].get(version.source_path)
            == TreeEntry(mode=version.mode, blob=version.blob)
        ]
        relative_archive = archive_path(version.source_path, version.blob)
        row = {
            "source_path": version.source_path,
            "git_blob": version.blob,
            "archive_path": relative_archive,
            "sha256": sha256(data),
            "bytes": len(data),
            "git_mode": version.mode,
            "artifact_kind": kind,
            "model_family": family,
            "stage": stage,
            "declared_model": declared_model,
            "parameter_count": parameter_count,
            "record_count": record_count,
            "first_seen_commit": first_commit,
            "first_seen_date": commit_dates[first_commit],
            "last_seen_commit": last_commit,
            "last_seen_date": commit_dates[last_commit],
            "commit_count": len(version.commits),
            "history_refs": ";".join(history_refs),
            "tip_refs": ";".join(tip_refs),
        }
        manifest.append(row)

        if PurePosixPath(version.source_path).suffix.lower() == ".json":
            attempts.append(
                {
                    "attempt_id": f"{family}:{version.blob[:12]}:{version.source_path}",
                    "model_family": family,
                    "stage": stage,
                    "artifact_kind": kind,
                    "source_path": version.source_path,
                    "git_blob": version.blob,
                    "archive_path": relative_archive,
                    "declared_model": declared_model,
                    "parameter_count": parameter_count,
                    "record_count": record_count,
                    "first_seen_commit": first_commit,
                    "first_seen_date": commit_dates[first_commit],
                    "history_refs": ";".join(history_refs),
                }
            )

    artifact_files = {
        row["archive_path"]: blobs[row["git_blob"]] for row in manifest
    }
    coverage = {
        "schema_version": 1,
        "source_repository": "https://github.com/DeveshParagiri/ed-autoresearch",
        "history_bundle": "model/other-models/upstream/ed-autoresearch.bundle",
        "history_bundle_sha256": sha256(BUNDLE.read_bytes()),
        "audited_refs": {
            name: {"ref": ref, "tip": tip}
            for name, (ref, tip) in AUDITED_REFS.items()
        },
        "commit_count": len(commit_order),
        "source_path_count": len({row["source_path"] for row in manifest}),
        "path_blob_version_count": len(manifest),
        "unique_blob_count": len(blobs),
        "materialized_file_count": len(artifact_files),
        "json_attempt_record_count": len(attempts),
        "selector": [
            "models/**",
            "configs/**",
            "data_human/**",
            "patches/**",
            "HPC_AFRICA_HANDOFF/reference/**",
        ],
        "boundary": (
            "The archive covers every committed file in the model-bearing directories "
            "above at every distinct path/blob version. Full source outside those "
            "directories, scripts, logs, and commit metadata remain in the complete Git "
            "bundle. Uncommitted outputs cannot be recovered from Git, and optimizer "
            "trials are not mislabeled as separate named models."
        ),
        "coverage_complete": True,
    }
    return manifest, attempts, artifact_files, coverage


def prepare_repository(temporary_root: Path) -> Path:
    repository = temporary_root / "history.git"
    subprocess.run(
        ["git", "init", "--bare", "--quiet", str(repository)], check=True
    )
    refspecs = [
        f"+{ref}:{ref}" for ref, _tip in AUDITED_REFS.values()
    ]
    completed = subprocess.run(
        ["git", "-C", str(repository), "fetch", "--quiet", str(BUNDLE), *refspecs],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read history bundle: {completed.stderr.strip()}")
    return repository


def expected_outputs(repository: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    manifest, attempts, artifacts, coverage = collect_history(repository)
    outputs = dict(artifacts)
    outputs[
        "model/other-models/commit-artifacts/manifest.csv"
    ] = csv_bytes(MANIFEST_FIELDS, manifest)
    outputs[
        "model/other-models/commit-artifacts/model-attempts.csv"
    ] = csv_bytes(ATTEMPT_FIELDS, attempts)
    outputs[
        "model/other-models/commit-artifacts/coverage.json"
    ] = (json.dumps(coverage, indent=2, sort_keys=True) + "\n").encode()
    return outputs, coverage


def write_outputs(outputs: dict[str, bytes]) -> None:
    by_path = OUTPUT / "by-path"
    expected_artifacts = {
        ROOT / relative
        for relative in outputs
        if "/by-path/" in relative
    }
    if by_path.exists():
        unexpected = sorted(
            path for path in by_path.rglob("*") if path.is_file() and path not in expected_artifacts
        )
        if unexpected:
            rendered = "\n".join(str(path.relative_to(ROOT)) for path in unexpected[:20])
            raise RuntimeError(
                "refusing to delete unexpected files from commit-artifacts/by-path:\n"
                + rendered
            )
    for relative, data in outputs.items():
        path = ROOT / relative
        if not path.is_file() or path.read_bytes() != data:
            atomic_write(path, data)


def check_outputs(outputs: dict[str, bytes]) -> list[str]:
    errors: list[str] = []
    expected_paths = {ROOT / relative for relative in outputs}
    for path, data in ((ROOT / relative, data) for relative, data in outputs.items()):
        if not path.is_file():
            errors.append(f"missing: {path.relative_to(ROOT)}")
        elif path.read_bytes() != data:
            errors.append(f"content mismatch: {path.relative_to(ROOT)}")

    by_path = OUTPUT / "by-path"
    if by_path.exists():
        for path in sorted(item for item in by_path.rglob("*") if item.is_file()):
            if path not in expected_paths:
                errors.append(f"unexpected: {path.relative_to(ROOT)}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose and verify every committed historical model artifact."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="materialize the archive; without this flag the command only checks it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not BUNDLE.is_file():
        raise SystemExit(f"history bundle is missing: {BUNDLE}")

    with tempfile.TemporaryDirectory(prefix="ed-model-history-") as temporary:
        repository = prepare_repository(Path(temporary))
        outputs, coverage = expected_outputs(repository)
    if args.write:
        write_outputs(outputs)
    errors = check_outputs(outputs)
    if errors:
        print("Historical model artifact coverage: FAIL")
        for error in errors[:50]:
            print(error)
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more")
        return 1

    print("Historical model artifact coverage: PASS")
    print(
        f"{coverage['commit_count']} commits, "
        f"{coverage['source_path_count']} source paths, "
        f"{coverage['path_blob_version_count']} path/blob versions, "
        f"{coverage['unique_blob_count']} unique blobs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
