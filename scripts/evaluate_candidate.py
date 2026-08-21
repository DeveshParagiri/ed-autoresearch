#!/usr/bin/env python3
"""Run the one fixed ED-Fire ILAMB burned-area evaluation suite."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from burned_area_figures import FIGURE_NAMES, configure_plotting, render_suite
from evaluation_support import (
    ensure_within,
    link_file,
    run_ilamb,
    safe_relative,
    sha256,
    summarize_field,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "evals" / "burned-area.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one burned-area candidate against GFED5 and GFED4.1s."
    )
    parser.add_argument(
        "candidate",
        type=Path,
        help="NetCDF containing monthly burntArea on the fixed 0.5-degree grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or empty results directory inside this repository.",
    )
    return parser.parse_args()


def verify_protected_files(contract: dict[str, Any]) -> None:
    changed: list[str] = []
    for record in contract["protected_files"]:
        path = PROJECT_ROOT / safe_relative(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            changed.append(record["path"])
    if changed:
        raise ValueError("fixed evaluation inputs changed: " + ", ".join(changed))


def prepare_run(candidate: Path, output: Path) -> tuple[Path, dict[str, Any]]:
    source = candidate.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"candidate does not exist: {source}")

    run_root = ensure_within(output.expanduser(), PROJECT_ROOT)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir()
    (run_root / "figures").mkdir()

    contract = json.loads(CONTRACT_PATH.read_text())
    verify_protected_files(contract)
    shutil.copy2(CONTRACT_PATH, run_root / "contract.json")

    staged = run_root / safe_relative(contract["candidate_output"]["path"])
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, staged)
    write_json(
        run_root / "candidate.json",
        {
            "source": str(source),
            "source_sha256": sha256(source),
            "staged": str(staged.relative_to(run_root)),
            "staged_sha256": sha256(staged),
        },
    )
    return run_root, contract


def evaluate(run_root: Path, contract: dict[str, Any]) -> None:
    evaluation = contract["evaluation"]
    period = evaluation["period"]
    start_year = int(period["start_year"])
    end_year = int(period["end_year"])
    regions = evaluation["regions"]

    candidate = run_root / safe_relative(contract["candidate_output"]["path"])
    stock = PROJECT_ROOT / safe_relative(evaluation["stock"]["path"])
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"staged candidate is missing: {candidate}")
    if not stock.is_file() or sha256(stock) != evaluation["stock"]["sha256"]:
        raise ValueError("fixed native ED comparison is missing or changed")

    workspace = run_root / "work" / "evaluation"
    model_root = workspace / "models"
    reference_root = workspace / "references"
    for model_name, source in (("Candidate", candidate), ("ED-stock", stock)):
        destination = model_root / model_name / "burntArea.nc"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for benchmark in evaluation["benchmarks"]:
        source = PROJECT_ROOT / safe_relative(benchmark["reference"])
        if not source.is_file() or sha256(source) != benchmark["reference_sha256"]:
            raise ValueError(f"fixed benchmark is missing or changed: {benchmark['id']}")
        link_file(source, reference_root / safe_relative(benchmark["ilamb_source"]))

    executable = Path(evaluation["ilamb"]["executable"])
    if not executable.is_file():
        raise FileNotFoundError(f"ILAMB executable is missing: {executable}")

    scores_by_model: dict[str, dict[str, dict[str, float]]] = {
        "Candidate": {},
        "ED-stock": {},
    }
    invocations: list[dict[str, Any]] = []
    for benchmark in evaluation["benchmarks"]:
        scores, invocation = run_ilamb(
            project_root=PROJECT_ROOT,
            run_root=run_root,
            executable=executable,
            expected_version=evaluation["ilamb"]["version"],
            model_root=model_root,
            reference_root=reference_root,
            benchmark=benchmark,
        )
        for model in scores_by_model:
            scores_by_model[model][benchmark["id"]] = scores[model]
        invocations.append(invocation)

    field_paths = {
        "candidate": candidate,
        "stock": stock,
        **{
            benchmark["id"]: PROJECT_ROOT / safe_relative(benchmark["reference"])
            for benchmark in evaluation["benchmarks"]
        },
    }
    fields: dict[str, dict[str, Any]] = {}
    summary_cache: dict[str, dict[str, Any]] = {}
    for field_id, path in field_paths.items():
        digest = sha256(path)
        if digest not in summary_cache:
            summary_cache[digest] = summarize_field(
                path,
                variable_name=contract["candidate_output"]["variable"],
                start_year=start_year,
                end_year=end_year,
                regions=regions,
            )
        fields[field_id] = summary_cache[digest]

    candidate_artifact = run_root / "artifacts" / "model-output" / "burntArea.nc"
    candidate_artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, candidate_artifact)

    metrics = {
        "schema": "ed-fire-evaluation/v1",
        "period": f"{start_year}-01/{end_year}-12",
        "candidate": scores_by_model["Candidate"],
        "stock": scores_by_model["ED-stock"],
        "benchmark_sensitivity": {
            "candidate_overall_score_delta_gfed5_minus_gfed4_1s": (
                scores_by_model["Candidate"]["gfed5"]["overall_score"]
                - scores_by_model["Candidate"]["gfed4_1s"]["overall_score"]
            ),
            "stock_overall_score_delta_gfed5_minus_gfed4_1s": (
                scores_by_model["ED-stock"]["gfed5"]["overall_score"]
                - scores_by_model["ED-stock"]["gfed4_1s"]["overall_score"]
            ),
        },
    }
    write_json(run_root / "metrics.json", metrics)

    renderer = PROJECT_ROOT / "scripts" / "burned_area_figures.py"
    write_json(
        run_root / "artifacts" / "evaluation.json",
        {
            "schema": "ed-fire-evaluation-record/v2",
            "period": metrics["period"],
            "contract": str(CONTRACT_PATH.relative_to(PROJECT_ROOT)),
            "candidate_output": {
                "path": str(candidate_artifact.relative_to(run_root)),
                "sha256": sha256(candidate_artifact),
                "bytes": candidate_artifact.stat().st_size,
            },
            "stock_output": {
                "path": evaluation["stock"]["path"],
                "sha256": evaluation["stock"]["sha256"],
            },
            "benchmarks": [
                {
                    "id": benchmark["id"],
                    "path": benchmark["reference"],
                    "sha256": benchmark["reference_sha256"],
                    "config": benchmark["config"],
                    "config_sha256": sha256(
                        PROJECT_ROOT / safe_relative(benchmark["config"])
                    ),
                }
                for benchmark in evaluation["benchmarks"]
            ],
            "invocations": invocations,
            "regions": regions,
            "plot_scales": evaluation["plot_scales"],
            "renderer": {
                "path": str(renderer.relative_to(PROJECT_ROOT)),
                "sha256": sha256(renderer),
                "style": ["science", "no-latex", "bright"],
                "font_family": "Basier Square",
            },
        },
    )

    dimensions = {
        figure["filename"]: {
            "width": int(figure["dimensions"]["width"]),
            "height": int(figure["dimensions"]["height"]),
        }
        for figure in contract["figures"]
    }
    if tuple(dimensions) != FIGURE_NAMES:
        raise ValueError(f"fixed figure set changed: {tuple(dimensions)}")

    plt = configure_plotting(run_root / "work" / "mplconfig")
    render_suite(
        plt,
        run_root / "figures",
        dimensions,
        scores_by_model,
        fields,
        regions,
        evaluation["plot_scales"],
        candidate_label="Candidate",
    )
    verify_protected_files(contract)

    print(f"results={run_root.relative_to(PROJECT_ROOT)}")
    print(f"GFED5 overall={scores_by_model['Candidate']['gfed5']['overall_score']:.6f}")
    print(
        "GFED4.1s overall="
        f"{scores_by_model['Candidate']['gfed4_1s']['overall_score']:.6f}"
    )


def main() -> int:
    args = parse_args()
    run_root, contract = prepare_run(args.candidate, args.output)
    evaluate(run_root, contract)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        raise
