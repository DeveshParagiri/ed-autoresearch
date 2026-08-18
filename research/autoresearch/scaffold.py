#!/usr/bin/env python3
"""Create a new autoresearch workspace from the bundled generic template."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCAFFOLD_ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = SCAFFOLD_ROOT / "template"
TOKEN_PATTERN = re.compile(r"__[A-Z][A-Z0-9_]*__")
TEMPLATE_PATHS = {
    ".gitignore",
    "README.md",
    "research.md",
    "memory.md",
    "pyproject.toml",
    "data/catalog.toml",
    "data/sources.toml",
    "evals/contracts/baseline-v1.json",
    "research/experiments/experiment.baseline/experiment.md",
    "scripts/check_workspace.py",
    "scripts/install_data.py",
    "scripts/run_experiment.py",
    "skills/autoresearch/SKILL.md",
    "skills/autoresearch/agents/openai.yaml",
    "src/autoresearch/__init__.py",
    "src/autoresearch/data_install.py",
    "src/autoresearch/optuna_records.py",
    "src/autoresearch/runner.py",
    "src/autoresearch/validation.py",
    "tests/test_workspace.py",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError("project name must contain at least one letter or number")
    return slug


def template_files() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in TEMPLATE_ROOT.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    )


def token_values(
    name: str,
    slug: str | None,
    description: str | None,
    mode: str = "mechanistic",
) -> dict[str, str]:
    project_slug = slugify(slug or name)
    project_name = name.strip()
    if not project_name:
        raise ValueError("project name cannot be empty")
    if mode not in {"mechanistic", "simulation", "hybrid"}:
        raise ValueError(f"unsupported execution mode: {mode}")
    project_description = (description or f"Persistent autoresearch workspace for {project_name}.").strip()
    return {
        "__PROJECT_NAME__": project_name,
        "__BASELINE_TITLE_YAML__": json.dumps(
            f"Establish the {project_name} baseline", ensure_ascii=False
        ),
        "__PROJECT_SLUG__": project_slug,
        "__PROJECT_DESCRIPTION__": project_description,
        "__PROJECT_DESCRIPTION_TOML__": json.dumps(project_description, ensure_ascii=False),
        "__EXECUTION_MODE__": mode,
        "__DATA_STORE_DESCRIPTION_TOML__": json.dumps(
            f"Canonical large-data root for {project_name}.", ensure_ascii=False
        ),
        "__DEFAULT_PROMPT_YAML__": json.dumps(
            f"Use $autoresearch to recover {project_name} state and take the next "
            "evidence-backed experiment action.",
            ensure_ascii=False,
        ),
        "__CREATED_AT__": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
        "__DATA_ROOT_ENV__": f"{project_slug.upper().replace('-', '_')}_DATA_ROOT",
    }


def render_text(text: str, values: dict[str, str]) -> str:
    for token, value in values.items():
        text = text.replace(token, value)
    unresolved = sorted(set(TOKEN_PATTERN.findall(text)))
    if unresolved:
        raise ValueError(f"unresolved template tokens: {', '.join(unresolved)}")
    return text


def check_template() -> tuple[str, ...]:
    errors: list[str] = []
    if not TEMPLATE_ROOT.is_dir():
        return (f"template directory is missing: {TEMPLATE_ROOT}",)

    relative_files = {str(path.relative_to(TEMPLATE_ROOT)) for path in template_files()}
    missing = sorted(TEMPLATE_PATHS - relative_files)
    if missing:
        errors.append(f"required template files are missing: {', '.join(missing)}")
    unexpected = sorted(relative_files - TEMPLATE_PATHS)
    if unexpected:
        errors.append(f"unexpected template files are present: {', '.join(unexpected)}")

    allowed_tokens = set(token_values("Example Project", "example-project", "Example.").keys())
    for source in template_files():
        try:
            text = source.read_text()
        except UnicodeDecodeError:
            errors.append(f"template contains a non-text file: {source.relative_to(TEMPLATE_ROOT)}")
            continue
        unknown = sorted(set(TOKEN_PATTERN.findall(text)) - allowed_tokens)
        if unknown:
            errors.append(
                f"{source.relative_to(TEMPLATE_ROOT)} has unknown tokens: {', '.join(unknown)}"
            )
    return tuple(errors)


def create_workspace(
    destination: Path,
    *,
    name: str,
    slug: str | None = None,
    description: str | None = None,
    mode: str = "mechanistic",
    dry_run: bool = False,
) -> tuple[Path, ...]:
    errors = check_template()
    if errors:
        raise ValueError("; ".join(errors))

    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    values = token_values(name, slug, description, mode)
    planned = tuple(destination / source.relative_to(TEMPLATE_ROOT) for source in template_files())
    if dry_run:
        return planned

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.scaffold-", dir=destination.parent)
    )
    try:
        for source, final_path in zip(template_files(), planned, strict=True):
            relative = final_path.relative_to(destination)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render_text(source.read_text(), values))
            target.chmod(source.stat().st_mode)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return planned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="validate the bundled template")

    create = subparsers.add_parser("create", help="create a new autoresearch workspace")
    create.add_argument("destination", type=Path)
    create.add_argument("--name", required=True, help="human-readable project name")
    create.add_argument("--slug", help="stable lowercase project identifier")
    create.add_argument("--description", help="one-sentence project description")
    create.add_argument(
        "--mode",
        choices=("mechanistic", "simulation", "hybrid"),
        default="mechanistic",
        help="baseline execution mode (default: mechanistic)",
    )
    create.add_argument("--dry-run", action="store_true", help="print files without writing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "check":
        errors = check_template()
        if errors:
            for error in errors:
                print(f"ERROR {error}")
            return 1
        print(f"template=ok files={len(template_files())}")
        return 0

    try:
        created = create_workspace(
            args.destination,
            name=args.name,
            slug=args.slug,
            description=args.description,
            mode=args.mode,
            dry_run=args.dry_run,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR {exc}")
        return 2

    action = "would-create" if args.dry_run else "created"
    for path in created:
        print(f"{action} {path}")
    print(f"workspace={args.destination.expanduser().resolve()} files={len(created)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
