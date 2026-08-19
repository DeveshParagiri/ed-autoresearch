#!/usr/bin/env python3
"""Evaluate one candidate with the v2 ED-Fire burned-area presentation contract."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from burned_area_figures import FIGURE_NAMES, configure_plotting, render_suite
from evaluate_burned_area import (
    ensure_within,
    link_file,
    run_ilamb,
    safe_relative,
    sha256,
    summarize_field,
    write_json,
)


def main() -> int:
    import os

    project_root = Path(os.environ["AUTORESEARCH_PROJECT_ROOT"]).resolve()
    run_root = Path(os.environ["AUTORESEARCH_RUN_ROOT"]).resolve()
    ensure_within(run_root, project_root)
    contract = json.loads((run_root / "contract.json").read_text())
    evaluation = contract["evaluation"]
    period = evaluation["period"]
    start_year = int(period["start_year"])
    end_year = int(period["end_year"])
    regions = evaluation["regions"]

    candidate = run_root / safe_relative(contract["candidate_output"]["path"])
    stock = project_root / safe_relative(evaluation["stock"]["path"])
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(
            f"candidate output is missing or is a symlink: {candidate}"
        )
    if not stock.is_file() or sha256(stock) != evaluation["stock"]["sha256"]:
        raise ValueError("locked stock ED output is missing or changed")

    workspace = run_root / "work" / "evaluation"
    model_root = workspace / "models"
    reference_root = workspace / "references"
    for model_name, source in (("Candidate", candidate), ("ED-stock", stock)):
        destination = model_root / model_name / "burntArea.nc"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for benchmark in evaluation["benchmarks"]:
        source = project_root / safe_relative(benchmark["reference"])
        if not source.is_file() or sha256(source) != benchmark["reference_sha256"]:
            raise ValueError(
                f"locked benchmark is missing or changed: {benchmark['id']}"
            )
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
            project_root=project_root,
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
            benchmark["id"]: project_root
            / safe_relative(benchmark["reference"])
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

    renderer = project_root / "scripts" / "burned_area_figures.py"
    evaluation_record = {
        "schema": "ed-fire-evaluation-record/v2",
        "period": metrics["period"],
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
                    project_root / safe_relative(benchmark["config"])
                ),
            }
            for benchmark in evaluation["benchmarks"]
        ],
        "invocations": invocations,
        "regions": regions,
        "plot_scales": evaluation["plot_scales"],
        "renderer": {
            "path": str(renderer.relative_to(project_root)),
            "sha256": sha256(renderer),
            "style": ["science", "no-latex", "bright"],
            "font_family": "Basier Square",
        },
    }
    write_json(run_root / "artifacts" / "evaluation.json", evaluation_record)

    dimensions = {
        figure["filename"]: {
            "width": int(figure["dimensions"]["width"]),
            "height": int(figure["dimensions"]["height"]),
        }
        for figure in contract["figures"]
    }
    if tuple(dimensions) != FIGURE_NAMES:
        raise ValueError(f"canonical figure set changed: {tuple(dimensions)}")
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

    print(
        f"evaluated candidate against GFED5 and GFED4.1s for "
        f"{start_year}-{end_year}"
    )
    print(
        f"GFED5 overall="
        f"{scores_by_model['Candidate']['gfed5']['overall_score']:.6f}"
    )
    print(
        f"GFED4.1s overall="
        f"{scores_by_model['Candidate']['gfed4_1s']['overall_score']:.6f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        raise
