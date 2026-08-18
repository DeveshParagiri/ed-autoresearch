from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility.
    import tomli as tomllib


@dataclass(frozen=True)
class InstallEvent:
    status: str
    item_id: str
    message: str


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def source_root(project_root: Path, override: Path | None = None) -> Path:
    manifest = _load(project_root / "data" / "sources.toml")
    if override is not None:
        return override.expanduser().resolve()
    variable = manifest["store"]["environment_variable"]
    configured = os.environ.get(variable)
    if configured:
        return Path(configured).expanduser().resolve()
    return (project_root / manifest["store"]["default_relative_to_project"]).resolve()


def _same_link(destination: Path, source: Path) -> bool:
    if not destination.is_symlink():
        return False
    linked = Path(os.readlink(destination))
    if not linked.is_absolute():
        linked = destination.parent / linked
    return linked.resolve(strict=False) == source.resolve(strict=False)


def _same_location(destination: Path, source: Path) -> bool:
    return destination.exists() and destination.resolve() == source.resolve()


def _safe_child(root: Path, relative: str, *, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path: {relative!r}")
    root = root.resolve()
    candidate = root / path
    try:
        candidate.parent.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its root: {relative!r}") from exc
    return candidate


def _declared_child(root: Path, relative: str, *, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path: {relative!r}")
    return root / path


def _parent_is_local(project_root: Path, destination: Path) -> bool:
    try:
        destination.parent.resolve(strict=False).relative_to(project_root.resolve())
    except ValueError:
        return False
    return True


def _link(
    item_id: str,
    source: Path,
    destination: Path,
    *,
    required: bool,
    dry_run: bool,
    project_root: Path,
) -> InstallEvent:
    if not source.exists():
        return InstallEvent("missing-source" if required else "optional-missing", item_id, str(source))
    if _same_location(destination, source):
        return InstallEvent("ready", item_id, f"{destination} -> {source}")
    if _same_link(destination, source):
        return InstallEvent("ready", item_id, f"{destination} -> {source}")
    if destination.is_symlink():
        return InstallEvent("conflict", item_id, f"{destination} points to {os.readlink(destination)!r}")
    if destination.exists():
        return InstallEvent("conflict", item_id, f"{destination} exists and is not a symlink")
    if not _parent_is_local(project_root, destination):
        return InstallEvent("conflict", item_id, f"{destination} has a parent outside the project")
    if dry_run:
        return InstallEvent("would-link", item_id, f"{destination} -> {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)
    return InstallEvent("linked", item_id, f"{destination} -> {source}")


def install(project_root: Path, root: Path, *, dry_run: bool = False) -> tuple[InstallEvent, ...]:
    project_root = project_root.resolve()
    root = root.expanduser().resolve()
    catalog = _load(project_root / "data" / "catalog.toml").get("datasets", [])
    sources = _load(project_root / "data" / "sources.toml").get("sources", [])
    catalog_by_id = {record["id"]: record for record in catalog}
    source_by_id = {record["id"]: record for record in sources}
    if set(catalog_by_id) != set(source_by_id):
        raise ValueError("catalog and source manifest IDs do not match")

    events: list[InstallEvent] = []
    for item_id in sorted(catalog_by_id):
        dataset = catalog_by_id[item_id]
        source = source_by_id[item_id]
        destination = _declared_child(
            project_root,
            dataset["path"],
            label=f"dataset path for {item_id}",
        )
        required = bool(dataset.get("required", False))
        if source.get("acquisition") == "repository":
            if destination.exists():
                events.append(InstallEvent("ready", item_id, str(destination)))
            else:
                status = "missing-repository-file" if required else "optional-missing"
                events.append(InstallEvent(status, item_id, str(destination)))
            continue
        relative_source = source.get("source_path")
        if not isinstance(relative_source, str) or not relative_source:
            events.append(InstallEvent("invalid-source", item_id, "source_path is missing"))
            continue
        source_location = _safe_child(
            root,
            relative_source,
            label=f"source path for {item_id}",
        )
        members = source.get("members", [])
        if members:
            if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
                events.append(InstallEvent("conflict", item_id, str(destination)))
                continue
            if not destination.exists() and not _parent_is_local(project_root, destination):
                events.append(
                    InstallEvent(
                        "conflict",
                        item_id,
                        f"{destination} has a parent outside the project",
                    )
                )
                continue
            if not dry_run:
                destination.mkdir(parents=True, exist_ok=True)
            for member in members:
                if not isinstance(member, str) or not member:
                    events.append(InstallEvent("invalid-source", item_id, repr(member)))
                    continue
                events.append(
                    _link(
                        f"{item_id}:{member}",
                        _safe_child(
                            source_location,
                            member,
                            label=f"source member for {item_id}",
                        ),
                        _declared_child(
                            destination,
                            member,
                            label=f"destination member for {item_id}",
                        ),
                        required=required,
                        dry_run=dry_run,
                        project_root=project_root,
                    )
                )
            continue
        events.append(
            _link(
                item_id,
                source_location,
                destination,
                required=required,
                dry_run=dry_run,
                project_root=project_root,
            )
        )
    return tuple(events)
