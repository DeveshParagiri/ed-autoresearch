"""Host-side ``evaluate()`` tool.

This tool will run official ILAMB once for the current committed model, append
its Git commit and three-decimal global and regional scores to
``autoresearch/results.tsv``, and return ephemeral diagnostic figures. The
ILAMB configuration and outputs are temporary runtime data rather than
repository files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import ILAMB

from scripts.fast_ilamb import GFED5_SHA256, GFED_REGIONS
from scripts.figures import generate_figures
from scripts.progress import DEFAULT_OUTPUT as PROGRESS_OUTPUT
from scripts.progress import render as render_progress
from scripts.runtime import (
    GFED5_PATH,
    RESULTS_PATH,
    ROOT,
    ModelError,
    load_inputs,
    load_model,
    predict_current,
    rounded_score,
    score_text,
    validate_model,
    write_candidate,
)


LEDGER_FIELDS = (
    "commit",
    "overall",
    "bias",
    "rmse",
    "seasonal",
    "spatial",
    "regional",
    "inputs",
    "description",
)
SCORE_FIELDS = {
    "Overall Score": "overall",
    "Bias Score": "bias",
    "RMSE Score": "rmse",
    "Seasonal Cycle Score": "seasonal",
    "Spatial Distribution Score": "spatial",
}
ILAMB_CONFIG = """[h1: Ecosystem and Carbon Cycle]
bgcolor = "#ECFFE6"

[h2: Burned Area]
variable       = "burntArea"
alternate_vars = "burntFractionAll"
weight         = 4
cmap           = "OrRd"
mass_weighting = True
ctype          = "ConfBurntArea"

[GFED5]
source         = "DATA/burntArea/GFED5/burntArea.nc"
weight         = 20
"""


class EvaluationError(RuntimeError):
    """An official-evaluation or ledger error suitable for CLI output."""


def _model_commit() -> str:
    """Return the commit containing the exact current model.py."""
    model_path = "autoresearch/model.py"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", model_path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise EvaluationError("commit model.py before running official evaluation")
    changed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", model_path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if changed.returncode == 1:
        raise EvaluationError("commit model.py before running official evaluation")
    if changed.returncode != 0:
        raise EvaluationError("could not verify the committed model.py state")
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0:
        raise EvaluationError("could not resolve the model Git commit")
    value = commit.stdout.strip()
    if len(value) != 40:
        raise EvaluationError("Git returned an invalid model commit")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_ledger(path: Path = RESULTS_PATH) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    if not lines:
        raise EvaluationError("results.tsv is empty and has no header")
    if tuple(lines[0].split("\t")) != LEDGER_FIELDS:
        raise EvaluationError("results.tsv header does not match the fixed schema")
    rows: list[dict[str, str]] = []
    for number, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        values = line.split("\t")
        if len(values) != len(LEDGER_FIELDS):
            raise EvaluationError(f"results.tsv line {number} has {len(values)} fields")
        rows.append(dict(zip(LEDGER_FIELDS, values, strict=True)))
    return rows


def _validate_metadata(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    if not args.description.strip():
        raise EvaluationError("--description cannot be empty")
    if any(character in args.description for character in ("\t", "\n", "\r")):
        raise EvaluationError("--description cannot contain tabs or newlines")


def _parse_official_scores(path: Path) -> dict[str, dict[str, float]]:
    expected_regions = tuple(GFED_REGIONS)
    scores: dict[str, dict[str, float]] = {region: {} for region in expected_regions}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Model") != "Candidate":
                continue
            region = row.get("Region", "")
            metric = SCORE_FIELDS.get(row.get("ScalarName", ""))
            if region not in scores or metric is None:
                continue
            value = float(row["Data"])
            if not value == value or value in (float("inf"), float("-inf")):
                raise EvaluationError(f"ILAMB returned non-finite {region}.{metric}")
            scores[region][metric] = value
    for values in scores.values():
        if "overall" not in values and all(
            metric in values for metric in ("bias", "rmse", "seasonal", "spatial")
        ):
            values["overall"] = (
                values["bias"]
                + 2.0 * values["rmse"]
                + values["seasonal"]
                + values["spatial"]
            ) / 5.0
    missing = [
        f"{region}.{metric}"
        for region, values in scores.items()
        for metric in SCORE_FIELDS.values()
        if metric not in values
    ]
    if missing:
        raise EvaluationError("ILAMB scalar database is incomplete: " + ", ".join(missing))
    return scores


def _official_evaluation(prediction: Any) -> dict[str, dict[str, float]]:
    if _sha256(GFED5_PATH) != GFED5_SHA256:
        raise EvaluationError("evals/gfed5.nc does not match the locked benchmark")
    if str(ILAMB.__version__) != "2.7.3":
        raise EvaluationError(f"expected ILAMB 2.7.3, found {ILAMB.__version__}")
    executable = Path(sys.executable).with_name("ilamb-run")
    if not executable.is_file():
        raise EvaluationError(f"repository ILAMB executable is missing: {executable}")

    with tempfile.TemporaryDirectory(prefix="ed-fire-ilamb-") as temporary:
        root = Path(temporary)
        candidate = root / "models" / "Candidate" / "burntArea.nc"
        write_candidate(prediction, candidate)
        reference = root / "references" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.symlink_to(GFED5_PATH.resolve())
        config = root / "gfed5.cfg"
        config.write_text(ILAMB_CONFIG)
        build = root / "build"
        mpl = root / "mplconfig"
        mpl.mkdir()

        command = [
            str(executable),
            "--config",
            str(config),
            "--model_root",
            str(root / "models"),
            "--models",
            "Candidate",
            "--study_limits",
            "2001",
            "2016",
            "--regions",
            *GFED_REGIONS.keys(),
            "--build_dir",
            str(build),
            "--skip_plots",
            "--title",
            "ED-Fire official GFED5 evaluation",
        ]
        environment = os.environ.copy()
        environment.update(
            ILAMB_ROOT=str(root / "references"),
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(mpl),
            PYTHONNOUSERSITE="1",
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = "\n".join(completed.stderr.splitlines()[-12:])
            raise EvaluationError(
                f"ILAMB failed with exit code {completed.returncode}:\n{detail}"
            )
        scalar_database = build / "scalar_database.csv"
        if not scalar_database.is_file():
            raise EvaluationError("ILAMB did not produce scalar_database.csv")
        return _parse_official_scores(scalar_database)


def _recorded_scores(scores: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        region: {metric: rounded_score(value) for metric, value in values.items()}
        for region, values in scores.items()
    }


def _append_result(
    args: argparse.Namespace,
    model: Any,
    scores: dict[str, dict[str, float]],
    commit: str,
) -> None:
    recorded = _recorded_scores(scores)
    global_scores = recorded["global"]
    regional = {
        region: recorded[region]
        for region in GFED_REGIONS
        if region != "global"
    }
    values = {
        "commit": commit,
        "overall": score_text(global_scores["overall"]),
        "bias": score_text(global_scores["bias"]),
        "rmse": score_text(global_scores["rmse"]),
        "seasonal": score_text(global_scores["seasonal"]),
        "spatial": score_text(global_scores["spatial"]),
        "regional": json.dumps(regional, sort_keys=True, separators=(",", ":")),
        "inputs": json.dumps(list(model.INPUTS), separators=(",", ":")),
        "description": args.description.strip(),
    }
    with RESULTS_PATH.open("a") as handle:
        handle.write("\t".join(values[field] for field in LEDGER_FIELDS) + "\n")


def run(args: argparse.Namespace) -> int:
    """Evaluate the current model once and record its official result."""
    try:
        rows = _read_ledger()
        _validate_metadata(args, rows)
        commit = _model_commit()
        model = load_model()
        inputs, _ = validate_model(model)
        data = load_inputs(inputs)
        prediction = predict_current(model, data)
        scores = _official_evaluation(prediction)
        _append_result(args, model, scores, commit)
        recorded = _recorded_scores(scores)
        global_scores = recorded["global"]
        print(
            "official "
            + " ".join(
                [
                    f"overall={score_text(global_scores['overall'])}",
                    f"bias={score_text(global_scores['bias'])}",
                    f"rmse={score_text(global_scores['rmse'])}",
                    f"seasonal={score_text(global_scores['seasonal'])}",
                    f"spatial={score_text(global_scores['spatial'])}",
                ]
            )
        )
        print(
            "regional="
            + json.dumps(
                {
                    region: recorded[region]
                    for region in GFED_REGIONS
                    if region != "global"
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        print(f"recorded={RESULTS_PATH}")
        print(f"commit={commit}")
        try:
            render_progress(RESULTS_PATH, PROGRESS_OUTPUT)
        except Exception as error:
            print(
                f"ar evaluate: result recorded, but progress graph failed: {error}",
                file=sys.stderr,
            )
        try:
            comparison, seasonal = generate_figures(prediction)
            print(f"comparison={comparison}")
            print(f"seasonal_cycle={seasonal}")
        except Exception as error:
            print(
                f"ar evaluate: result recorded, but diagnostics failed: {error}; "
                "run ar figures to recreate them",
                file=sys.stderr,
            )
    except (EvaluationError, ModelError, OSError, ValueError) as error:
        print(f"ar evaluate: {error}", file=sys.stderr)
        return 2
    return 0
