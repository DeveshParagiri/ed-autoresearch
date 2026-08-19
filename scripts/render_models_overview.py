#!/usr/bin/env python3
"""Render a readable A-I map overview from an experiment's selected run."""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from burned_area_figures import configure_plotting
from evaluate_burned_area import sha256
from figure_models import MODEL_IDS, REFERENCE, annual_percent, global_mha
from render_experiment_figures import experiment_path, front_matter, verified_file


ROOT = Path(__file__).resolve().parents[1]
WIDTH = 1800
HEIGHT = 3200
DPI = 150
FILENAME = "models-a-i-gfed5-mean-maps.png"
DISPLAY_NAMES = {
    "A-legacy": "Model A",
    "B-legacy": "Model B",
    "C-legacy": "Model C · GFED4.1s fit",
    "C": "Model C · GFED5 refit",
    "D": "Model D",
    "E": "Model E",
    "F": "Model F",
    "G": "Model G",
    "G6": "Model G6",
    "G7": "Model G7",
    "H": "Model H",
    "I": "Model I",
    "Ibest": "Model Ibest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a two-column A-I map sheet from an experiment's selected run "
            "without changing that run's immutable evidence."
        )
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default="experiment.models-a-i-reproduction",
        help="Experiment ID, directory, or experiment.md path.",
    )
    return parser.parse_args()


def model_output(run: Path, model_id: str, protocol: str, expected_hash: str) -> Path:
    protocol_root = {
        "abc_gfed4_1s": "abc-gfed4.1s",
        "coupled_gfed5": "coupled-gfed5",
    }[protocol]
    candidates = (
        run
        / "work"
        / "models-a-i"
        / protocol_root
        / "models"
        / model_id
        / "burntArea.nc",
        ROOT
        / "model"
        / "reproduced"
        / protocol_root
        / "models"
        / model_id
        / "burntArea.nc",
    )
    for path in candidates:
        if path.is_file() and sha256(path) == expected_hash:
            return path
    raise FileNotFoundError(
        f"no hash-matched output is available for {model_id}; rerun the experiment"
    )


def plot(
    record: Path,
    metrics: dict[str, Any],
    run: Path,
    contract: dict[str, Any],
) -> Path:
    results = metrics.get("results")
    if not isinstance(results, dict) or set(results) != set(MODEL_IDS):
        raise ValueError("selected run does not contain the complete A-I result set")

    protected = {
        item["path"]: item["sha256"] for item in contract["protected_files"]
    }
    reference_relative = REFERENCE.relative_to(ROOT).as_posix()
    reference = verified_file(
        REFERENCE,
        protected[reference_relative],
        "GFED5 burned-area reference",
    )
    reference_field, latitude, longitude = annual_percent(reference)
    reference_mha = global_mha(reference_field, latitude, longitude)
    panels: list[tuple[str, np.ndarray, str]] = [
        ("GFED5 benchmark", reference_field, f"{reference_mha:.3f} Mha yr$^{{-1}}$")
    ]

    for model_id in MODEL_IDS:
        result = results[model_id]
        path = model_output(
            run,
            model_id,
            str(result["protocol"]),
            str(result["candidate_output_sha256"]),
        )
        field, model_latitude, model_longitude = annual_percent(path)
        if not np.array_equal(model_latitude, latitude) or not np.array_equal(
            model_longitude, longitude
        ):
            raise ValueError(f"{model_id} does not use the shared comparison grid")
        panels.append(
            (
                DISPLAY_NAMES[model_id],
                field,
                (
                    f"GFED5 Overall {result['scores']['gfed5']['overall_score']:.3f}"
                    f" · {global_mha(field, latitude, longitude):.3f} Mha yr$^{{-1}}$"
                ),
            )
        )

    maximum = 80.0
    with tempfile.TemporaryDirectory(prefix="ed-fire-model-overview-") as temporary:
        plt = configure_plotting(Path(temporary) / "mplconfig")
        figure, axes = plt.subplots(7, 2, constrained_layout=True)
        figure.set_size_inches(WIDTH / DPI, HEIGHT / DPI)
        image = None
        for index, (axis, (title, values, subtitle)) in enumerate(
            zip(axes.flat, panels, strict=True)
        ):
            image = axis.imshow(
                values,
                origin="lower",
                extent=(-180, 180, -90, 90),
                vmin=0,
                vmax=maximum,
                cmap="OrRd",
                aspect="equal",
            )
            axis.set_title(title, fontweight="bold", fontsize=12, pad=21)
            axis.text(
                0.5,
                1.015,
                subtitle,
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
            )
            axis.set_xlim(-180, 180)
            axis.set_ylim(-90, 90)
            axis.set_xticks((-120, 0, 120), ("120°W", "0°", "120°E"))
            axis.set_yticks((-60, 0, 60), ("60°S", "0°", "60°N"))
            if index % 2:
                axis.tick_params(labelleft=False)

        if image is None:
            raise RuntimeError("no map panels were rendered")
        colorbar = figure.colorbar(
            image,
            ax=axes,
            orientation="horizontal",
            shrink=0.92,
            pad=0.012,
            aspect=45,
        )
        colorbar.set_label("burned area (% of grid-cell area yr$^{-1}$)")
        figure.suptitle(
            "ED-Fire Models A-I on one GFED5 comparison scale, 2001–2016",
            fontsize=17,
        )

        output = record.parent / "figures" / FILENAME
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output,
            dpi=DPI,
            metadata={"Software": "ED-Fire selected-run presentation"},
        )
        plt.close(figure)

    header = output.read_bytes()[:24]
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or struct.unpack(">II", header[16:24]) != (WIDTH, HEIGHT)
    ):
        raise ValueError(f"unexpected rendered dimensions: {output}")
    return output


def main() -> int:
    args = parse_args()
    record = experiment_path(args.experiment)
    metadata = front_matter(record)
    selected_run = metadata.get("selected_run")
    if not isinstance(selected_run, str) or not selected_run.startswith("run."):
        raise ValueError(f"{record} does not name a selected run")
    run = record.parent / "runs" / selected_run
    metrics = json.loads((run / "metrics.json").read_text())
    contract = json.loads((run / "contract.json").read_text())
    output = plot(record, metrics, run, contract)
    print(f"{output.relative_to(ROOT)}  {sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
