#!/usr/bin/env python3
"""Validate and preserve the model A-I replay and comparable figure suites."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import struct
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = (
    "A-legacy",
    "B-legacy",
    "C-legacy",
    "C",
    "D",
    "E",
    "F",
    "G",
    "G6",
    "G7",
    "H",
    "I",
    "Ibest",
)
OVERVIEWS = (
    "models-a-i-gfed5-score-summary.png",
    "models-a-i-gfed5-mean-maps.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def run_root() -> Path:
    value = os.environ.get("AUTORESEARCH_RUN_ROOT")
    if not value:
        raise RuntimeError("AUTORESEARCH_RUN_ROOT is required")
    return Path(value).resolve()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main() -> int:
    run = run_root()
    replay = run / "work" / "models-a-i"
    figure_root = replay / "figures"
    artifact_root = run / "artifacts"
    official_figure_root = run / "figures"

    registry = tomllib.loads((ROOT / "model" / "registry.toml").read_text())
    registry_models = {model["id"]: model for model in registry["models"]}
    if tuple(registry_models) != MODEL_IDS:
        raise ValueError("model/registry.toml does not contain the fixed A-I model order")

    verification = read_json(replay / "verification.json")
    if verification.get("schema") != "ed-fire-model-replay/v1":
        raise ValueError("unexpected replay verification schema")
    if verification.get("protected_files_unchanged") is not True:
        raise ValueError("the replay changed a protected benchmark")

    checks = verification.get("checks")
    outputs = verification.get("outputs")
    if not isinstance(checks, dict) or not isinstance(outputs, dict):
        raise ValueError("replay verification is missing checks or outputs")

    with (replay / "metrics.csv").open(newline="") as handle:
        replay_rows = {row["model_id"]: row for row in csv.DictReader(handle)}
    if tuple(replay_rows) != MODEL_IDS:
        raise ValueError("replay metrics do not contain the fixed A-I model order")

    v2_contract = read_json(ROOT / "evals" / "contracts" / "burned-area-eval-v2.json")
    figure_specs = {
        figure["filename"]: (
            int(figure["dimensions"]["width"]),
            int(figure["dimensions"]["height"]),
        )
        for figure in v2_contract["figures"]
    }
    expected_figure_names = tuple(figure_specs)

    model_index: dict[str, Any] = {}
    canonical_count = 0
    for model_id in MODEL_IDS:
        status = str(checks.get(model_id, {}).get("status", ""))
        if not status.startswith("pass"):
            raise ValueError(f"{model_id} did not pass replay verification: {status}")

        output = outputs.get(model_id)
        if not isinstance(output, dict):
            raise ValueError(f"{model_id} has no recorded replay output")
        output_path = ROOT / str(output.get("path", ""))
        if not output_path.is_file() or sha256(output_path) != output.get("sha256"):
            raise ValueError(f"{model_id} replay output does not match its recorded hash")

        suite_path = figure_root / model_id / "suite.json"
        suite = read_json(suite_path)
        if suite.get("schema") != "ed-fire-model-figure-suite/v1":
            raise ValueError(f"{model_id} has an unexpected figure-suite schema")
        if suite.get("model_id") != model_id:
            raise ValueError(f"{model_id} figure suite identifies another model")
        if suite.get("eligible_under_active_candidate_contract") is not False:
            raise ValueError(f"{model_id} figure suite omits its active-contract boundary")
        suite_figures = suite.get("figures")
        if not isinstance(suite_figures, list):
            raise ValueError(f"{model_id} figure suite has no figure manifest")
        if tuple(item.get("filename") for item in suite_figures) != expected_figure_names:
            raise ValueError(f"{model_id} figure suite is incomplete or out of order")

        durable_model_root = artifact_root / "models" / model_id
        for item in suite_figures:
            name = item["filename"]
            source = figure_root / model_id / name
            if not source.is_file() or sha256(source) != item.get("sha256"):
                raise ValueError(f"{model_id}/{name} does not match its figure manifest")
            if png_size(source) != figure_specs[name]:
                raise ValueError(f"{model_id}/{name} has the wrong dimensions")
            copy_file(source, durable_model_root / name)
            canonical_count += 1
        copy_file(suite_path, durable_model_root / "suite.json")

        candidate_scores = suite.get("scores", {}).get("Candidate", {})
        model_index[model_id] = {
            "display_name": registry_models[model_id]["display_name"],
            "protocol": registry_models[model_id]["protocol"],
            "replay_status": registry_models[model_id]["replay_status"],
            "verification": status,
            "candidate_output_sha256": output["sha256"],
            "figures": [
                f"models/{model_id}/{name}" for name in expected_figure_names
            ],
            "scores": candidate_scores,
        }

    for name in OVERVIEWS:
        source = figure_root / name
        if png_size(source) != (1800, 1200):
            raise ValueError(f"{name} has the wrong dimensions")
        copy_file(source, official_figure_root / name)

    copy_file(replay / "metrics.csv", artifact_root / "replay-metrics.csv")
    copy_file(replay / "verification.json", artifact_root / "replay-verification.json")
    (artifact_root / "model-index.json").write_text(
        json.dumps(
            {
                "schema": "ed-fire-model-index/v1",
                "models": model_index,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    metrics = {
        "models": {
            "total": len(MODEL_IDS),
            "replay_verified": len(MODEL_IDS),
            "fully_reproduced": sum(
                registry_models[model_id]["replay_status"] == "reproduced"
                for model_id in MODEL_IDS
            ),
            "partial": sum(
                registry_models[model_id]["replay_status"] == "partial"
                for model_id in MODEL_IDS
            ),
            "phase_tie_drift": sum(
                "phase-tie-drift" in registry_models[model_id]["replay_status"]
                for model_id in MODEL_IDS
            ),
            "noncomparable": sum(
                "noncomparable" in registry_models[model_id]["replay_status"]
                for model_id in MODEL_IDS
            ),
        },
        "figures": {
            "canonical": canonical_count,
            "overviews": len(OVERVIEWS),
        },
        "protected_files_unchanged": True,
        "results": model_index,
    }
    (run / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Verified {len(MODEL_IDS)} models, {canonical_count} canonical figures, "
        f"and {len(OVERVIEWS)} overview figures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
