#!/usr/bin/env python3
"""Render the current canonical figure suite from one experiment's selected run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from burned_area_figures import FIGURE_NAMES, configure_plotting, render_suite
from evaluate_burned_area import sha256, summarize_field


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "evals" / "contracts" / "burned-area-eval-v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render experiment-level figures from a selected run without changing "
            "the run's immutable evidence."
        )
    )
    parser.add_argument(
        "experiment",
        help="Experiment ID, experiment directory, or experiment.md path.",
    )
    parser.add_argument(
        "--label",
        help="Figure label for the staged candidate; defaults to the experiment title.",
    )
    parser.add_argument(
        "--contract",
        default=str(DEFAULT_CONTRACT.relative_to(ROOT)),
        help="Project-relative presentation contract (default: burned-area v2).",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"path leaves the project: {value}") from exc
    return path


def experiment_path(value: str) -> Path:
    supplied = Path(value)
    if supplied.is_absolute() or len(supplied.parts) > 1:
        path = supplied if supplied.is_absolute() else ROOT / supplied
    else:
        path = ROOT / "research" / "experiments" / value
    if path.is_dir():
        path = path / "experiment.md"
    path = path.resolve()
    experiment_root = (ROOT / "research" / "experiments").resolve()
    try:
        path.relative_to(experiment_root)
    except ValueError as exc:
        raise ValueError(f"experiment leaves research/experiments: {value}") from exc
    if path.name != "experiment.md" or not path.is_file():
        raise FileNotFoundError(f"experiment record not found: {path}")
    return path


def front_matter(path: Path) -> dict[str, Any]:
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path} has no YAML front matter")
    try:
        closing = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError(f"{path} has unclosed YAML front matter") from exc
    metadata = yaml.safe_load("\n".join(lines[1:closing]))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} front matter is not a mapping")
    return metadata


def verified_file(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is missing or is a symlink: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} checksum changed: expected {expected_sha256}, found {actual}"
        )
    return path


def render(
    record_path: Path,
    contract_path: Path,
    candidate_label: str | None,
) -> tuple[Path, ...]:
    metadata = front_matter(record_path)
    selected_run = metadata.get("selected_run")
    if not isinstance(selected_run, str) or not selected_run.startswith("run."):
        raise ValueError(f"{record_path} does not name a selected run")

    experiment = record_path.parent
    run = experiment / "runs" / selected_run
    if not run.is_dir():
        raise FileNotFoundError(f"selected run is missing: {run}")
    metrics = json.loads((run / "metrics.json").read_text())
    if not isinstance(metrics.get("candidate"), dict) or not isinstance(
        metrics.get("stock"), dict
    ):
        raise ValueError(f"selected run has no candidate and stock score vectors: {run}")

    contract = json.loads(contract_path.read_text())
    evaluation = contract["evaluation"]
    period = evaluation["period"]
    regions = evaluation["regions"]
    dimensions = {
        figure["filename"]: {
            "width": int(figure["dimensions"]["width"]),
            "height": int(figure["dimensions"]["height"]),
        }
        for figure in contract["figures"]
    }
    if tuple(dimensions) != FIGURE_NAMES:
        raise ValueError(f"canonical figure set changed: {tuple(dimensions)}")

    evaluation_record = json.loads(
        (run / "artifacts" / "evaluation.json").read_text()
    )
    candidate_record = evaluation_record["candidate_output"]
    candidate = (run / candidate_record["path"]).resolve()
    try:
        candidate.relative_to(run.resolve())
    except ValueError as exc:
        raise ValueError("selected candidate artifact leaves its run") from exc
    verified_file(
        candidate,
        candidate_record["sha256"],
        "selected candidate artifact",
    )

    stock_record = evaluation["stock"]
    stock = verified_file(
        project_path(stock_record["path"]),
        stock_record["sha256"],
        "stock ED field",
    )
    field_paths = {"candidate": candidate, "stock": stock}
    for benchmark in evaluation["benchmarks"]:
        field_paths[benchmark["id"]] = verified_file(
            project_path(benchmark["reference"]),
            benchmark["reference_sha256"],
            benchmark["label"],
        )

    fields: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for field_id, path in field_paths.items():
        digest = sha256(path)
        if digest not in summaries:
            summaries[digest] = summarize_field(
                path,
                variable_name=contract["candidate_output"]["variable"],
                start_year=int(period["start_year"]),
                end_year=int(period["end_year"]),
                regions=regions,
            )
        fields[field_id] = summaries[digest]

    label = candidate_label or str(metadata.get("title", "Candidate"))
    output = experiment / "figures"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ed-fire-figures-") as temporary:
        temporary_root = Path(temporary)
        rendered = temporary_root / "rendered"
        plt = configure_plotting(temporary_root / "mplconfig")
        paths = render_suite(
            plt,
            rendered,
            dimensions,
            {"Candidate": metrics["candidate"], "ED-stock": metrics["stock"]},
            fields,
            regions,
            evaluation["plot_scales"],
            candidate_label=label,
        )
        for path in paths:
            shutil.copy2(path, output / path.name)

    return tuple(output / name for name in FIGURE_NAMES)


def main() -> int:
    args = parse_args()
    record = experiment_path(args.experiment)
    contract = project_path(args.contract)
    if not contract.is_file():
        raise FileNotFoundError(f"presentation contract not found: {contract}")
    paths = render(record, contract, args.label)
    print(f"rendered {len(paths)} figures from {front_matter(record)['selected_run']}")
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.relative_to(ROOT)}  {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
