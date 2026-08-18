from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility for the ED environment.
    import tomli as tomllib

from .catalog import load_catalog


REQUIRED_SOURCE_STRING_FIELDS = (
    "source_name",
    "version",
    "retrieval",
    "time_coverage",
    "spatial_coverage",
    "units",
    "preprocessing",
    "limitations",
    "license",
    "integrity",
)
PUBLIC_SOURCE_ACQUISITIONS = {
    "derived-public-source",
    "git",
    "ilamb-reference",
    "ilamb-reference-and-lab-seed",
    "official-download",
}


@dataclass(frozen=True)
class SourceSpec:
    item_id: str
    acquisition: str
    source_path: str | None
    retrieval: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class InstallEvent:
    status: str
    item_id: str
    message: str


def load_sources(project_root: Path) -> tuple[dict[str, Any], tuple[SourceSpec, ...]]:
    manifest_path = project_root / "data" / "sources.toml"
    with manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)
    if manifest.get("schema") != "data-sources/v1":
        raise ValueError(f"unsupported data source schema in {manifest_path}")

    specs: list[SourceSpec] = []
    seen: set[str] = set()
    for item in manifest.get("sources", []):
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"source entry without a valid id in {manifest_path}")
        if item_id in seen:
            raise ValueError(f"duplicate source id {item_id!r} in {manifest_path}")
        seen.add(item_id)
        acquisition = item.get("acquisition")
        retrieval = item.get("retrieval")
        source_path = item.get("source_path")
        members = item.get("members", [])
        if not isinstance(acquisition, str) or not acquisition:
            raise ValueError(f"source {item_id!r} has no acquisition method")
        if not isinstance(retrieval, str) or not retrieval:
            raise ValueError(f"source {item_id!r} has no retrieval instruction")
        for field in REQUIRED_SOURCE_STRING_FIELDS:
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"source {item_id!r} has no valid {field}")
        source_url = item.get("source_url")
        if source_url is not None and (
            not isinstance(source_url, str) or not source_url.strip()
        ):
            raise ValueError(f"source_url for {item_id!r} must be a nonempty string")
        if acquisition in PUBLIC_SOURCE_ACQUISITIONS and not source_url:
            raise ValueError(f"public source {item_id!r} has no source_url")
        experiments = item.get("experiments")
        if not isinstance(experiments, list) or not all(
            isinstance(value, str) and value for value in experiments
        ):
            raise ValueError(
                f"experiments for source {item_id!r} must be a list of experiment IDs"
            )
        if source_path is not None and not isinstance(source_path, str):
            raise ValueError(f"source_path for {item_id!r} must be a string")
        if acquisition != "repository" and not source_path:
            raise ValueError(f"source {item_id!r} has no source_path")
        if not isinstance(members, list) or not all(isinstance(value, str) for value in members):
            raise ValueError(f"members for {item_id!r} must be a string list")
        specs.append(
            SourceSpec(
                item_id=item_id,
                acquisition=acquisition,
                source_path=source_path,
                retrieval=retrieval,
                members=tuple(members),
            )
        )
    return manifest, tuple(specs)


def validate_manifest_coverage(project_root: Path) -> None:
    catalog_ids = {item["id"] for item in load_catalog(project_root).get("datasets", [])}
    _, specs = load_sources(project_root)
    source_ids = {spec.item_id for spec in specs}
    missing = sorted(catalog_ids - source_ids)
    extra = sorted(source_ids - catalog_ids)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing source entries: {', '.join(missing)}")
        if extra:
            details.append(f"unknown source entries: {', '.join(extra)}")
        raise ValueError("; ".join(details))


def default_source_root(project_root: Path, manifest: dict[str, Any]) -> Path:
    configured = os.environ.get(manifest["store"]["environment_variable"])
    if configured:
        return Path(configured).expanduser().resolve()
    return (project_root / manifest["store"]["default_relative_to_project"]).resolve()


def _catalog_by_id(project_root: Path) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in load_catalog(project_root).get("datasets", [])}


def _same_link(destination: Path, source: Path) -> bool:
    if not destination.is_symlink():
        return False
    link_value = Path(os.readlink(destination))
    if not link_value.is_absolute():
        link_value = destination.parent / link_value
    return link_value.resolve(strict=False) == source.resolve(strict=False)


def _same_location(destination: Path, source: Path) -> bool:
    return destination.exists() and destination.resolve() == source.resolve()


def _declared_child(root: Path, relative: str, *, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path: {relative!r}")
    return root / path


def _source_child(root: Path, relative: str, *, label: str) -> Path:
    candidate = _declared_child(root.resolve(), relative, label=label)
    try:
        candidate.parent.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its root: {relative!r}") from exc
    return candidate


def _parent_is_local(project_root: Path, destination: Path) -> bool:
    try:
        destination.parent.resolve(strict=False).relative_to(project_root.resolve())
    except ValueError:
        return False
    return True


def _link_one(
    item_id: str,
    source: Path,
    destination: Path,
    *,
    required: bool,
    dry_run: bool,
    project_root: Path,
) -> InstallEvent:
    if not source.exists():
        status = "missing-source" if required else "optional-missing"
        return InstallEvent(status, item_id, str(source))
    if _same_location(destination, source):
        return InstallEvent("ready", item_id, f"{destination} -> {source}")
    if _same_link(destination, source):
        return InstallEvent("ready", item_id, f"{destination} -> {source}")
    if destination.is_symlink():
        return InstallEvent(
            "conflict",
            item_id,
            f"{destination} points to {os.readlink(destination)!r}, expected {source}",
        )
    if destination.exists():
        return InstallEvent("conflict", item_id, f"{destination} exists and is not a symlink")
    if not _parent_is_local(project_root, destination):
        return InstallEvent("conflict", item_id, f"{destination} has a parent outside the project")
    if dry_run:
        return InstallEvent("would-link", item_id, f"{destination} -> {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)
    return InstallEvent("linked", item_id, f"{destination} -> {source}")


def install_links(
    project_root: Path,
    source_root: Path,
    *,
    dry_run: bool = False,
) -> tuple[InstallEvent, ...]:
    project_root = project_root.resolve()
    source_root = source_root.expanduser().resolve()
    validate_manifest_coverage(project_root)
    catalog = _catalog_by_id(project_root)
    _, specs = load_sources(project_root)
    events: list[InstallEvent] = []

    for spec in specs:
        catalog_item = catalog[spec.item_id]
        destination = _declared_child(
            project_root,
            catalog_item["path"],
            label=f"dataset path for {spec.item_id}",
        )
        required = bool(catalog_item.get("required", False))
        if spec.acquisition == "repository":
            if destination.exists():
                status = "ready"
            else:
                status = "missing-repository-file" if required else "optional-missing"
            events.append(InstallEvent(status, spec.item_id, str(destination)))
            continue
        if spec.source_path is None:
            events.append(
                InstallEvent("invalid-source", spec.item_id, "source_path is not declared")
            )
            continue

        source = _source_child(
            source_root,
            spec.source_path,
            label=f"source path for {spec.item_id}",
        )
        if spec.members:
            if destination.is_symlink() or (
                destination.exists() and not destination.is_dir()
            ):
                events.append(
                    InstallEvent(
                        "conflict",
                        spec.item_id,
                        f"{destination} must be a directory containing selected model links",
                    )
                )
                continue
            if not destination.exists() and not _parent_is_local(project_root, destination):
                events.append(
                    InstallEvent(
                        "conflict",
                        spec.item_id,
                        f"{destination} has a parent outside the project",
                    )
                )
                continue
            if not dry_run:
                destination.mkdir(parents=True, exist_ok=True)
            for member in spec.members:
                events.append(
                    _link_one(
                        f"{spec.item_id}:{member}",
                        _source_child(
                            source,
                            member,
                            label=f"source member for {spec.item_id}",
                        ),
                        _declared_child(
                            destination,
                            member,
                            label=f"destination member for {spec.item_id}",
                        ),
                        required=required,
                        dry_run=dry_run,
                        project_root=project_root,
                    )
                )
            continue
        events.append(
            _link_one(
                spec.item_id,
                source,
                destination,
                required=required,
                dry_run=dry_run,
                project_root=project_root,
            )
        )
    return tuple(events)


def recovery_instructions(project_root: Path, item_ids: set[str]) -> tuple[str, ...]:
    _, specs = load_sources(project_root)
    instructions: list[str] = []
    for spec in specs:
        base_id = spec.item_id
        if base_id in item_ids or any(item.startswith(f"{base_id}:") for item in item_ids):
            instructions.append(f"{base_id}: {spec.retrieval}")
    return tuple(instructions)
