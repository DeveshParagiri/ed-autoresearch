#!/usr/bin/env python3
"""Render quarantined historical ED-Fire comparisons and canonical suites."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from burned_area_figures import (
    BASIER_REGULAR,
    BASIER_SEMIBOLD,
    FIGURE_NAMES,
    FONT_FAMILY,
    configure_plotting,
    render_suite,
)
from evaluate_burned_area import link_file, read_scores, run_ilamb, sha256, summarize_field

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "model" / "other-models" / "reproduced"
METRICS = REPLAY / "metrics.csv"
VERIFICATION = REPLAY / "verification.json"
MODEL_ROOT = REPLAY / "coupled-gfed5" / "models"
REFERENCE = ROOT / "data" / "benchmarks" / "observations" / "gfed5-burned-area.nc"
CONTRACT = ROOT / "evals" / "contracts" / "burned-area-eval-v2.json"
OUTPUT = REPLAY / "figures"
EVALUATIONS = REPLAY / "standardized-evaluations"
REGISTRY = ROOT / "model" / "other-models" / "registry.toml"
STOCK = ROOT / "data" / "benchmarks" / "comparison-models" / "ilamb" / "EDv3" / "burntArea.nc"

MODEL_IDS = ("C", "D", "E", "F", "G", "G6", "G7", "H", "I", "Ibest")
SCORE_SERIES = (
    ("bias_score", "Bias", "#4C78A8"),
    ("rmse_score", "RMSE", "#F58518"),
    ("seasonal_cycle_score", "Seasonal", "#54A24B"),
    ("spatial_distribution_score", "Spatial", "#E45756"),
    ("overall_score", "Overall", "#111111"),
)
WIDTH = 1800
HEIGHT = 1200
DPI = 150


def load_metrics() -> dict[str, dict[str, float]]:
    if not METRICS.exists() or not VERIFICATION.exists():
        raise FileNotFoundError(
            "run scripts/reproduce_other_models.py --models all --evaluate first"
        )
    verification = json.loads(VERIFICATION.read_text())
    failed = [
        model_id
        for model_id in MODEL_IDS
        if not verification["checks"].get(model_id, {}).get("status", "").startswith(
            "pass"
        )
    ]
    if failed:
        raise RuntimeError("unverified historical models: " + ", ".join(failed))

    rows: dict[str, dict[str, float]] = {}
    with METRICS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            model_id = row["model_id"]
            if model_id not in MODEL_IDS:
                continue
            rows[model_id] = {
                key: float(value)
                for key, value in row.items()
                if key not in {"protocol", "model_id", "verification"} and value
            }
    missing = [model_id for model_id in MODEL_IDS if model_id not in rows]
    if missing:
        raise RuntimeError("metrics are missing models: " + ", ".join(missing))
    return rows


def annual_percent(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path, decode_times=False) as dataset:
        if "burntArea" not in dataset:
            raise ValueError(f"{path} has no burntArea variable")
        variable = dataset["burntArea"]
        values = variable.isel(time=slice(0, 192)).values.astype(np.float64)
        latitude = dataset.lat.values.astype(np.float64)
        longitude = dataset.lon.values.astype(np.float64)
        units = str(variable.attrs.get("units", "1")).strip().lower()
    if values.shape != (192, 360, 720):
        raise ValueError(f"{path} does not contain the 2001-2016 comparison grid")
    if units in {"%", "percent", "percentage"}:
        values *= 0.01
    elif units not in {"1", "fraction", "fractional"}:
        raise ValueError(f"unsupported burned-area units in {path}: {units!r}")
    valid = np.isfinite(values).any(axis=0)
    yearly = np.nansum(values.reshape(16, 12, 360, 720), axis=1)
    annual = np.mean(yearly, axis=0) * 100.0
    annual[~valid] = np.nan
    return annual, latitude, longitude


def global_mha(annual: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> float:
    dlat = np.deg2rad(abs(float(latitude[1] - latitude[0])))
    dlon = np.deg2rad(abs(float(longitude[1] - longitude[0])))
    radians = np.deg2rad(latitude)
    area = (
        6371000.0**2
        * dlon
        * (np.sin(radians + dlat / 2) - np.sin(radians - dlat / 2))
    )[:, None] * np.ones((1, len(longitude)))
    return float(np.nansum((annual / 100.0) * area) / 1e10)


def plotting() -> Any:
    return configure_plotting(REPLAY / ".mplconfig")


def save(plt: Any, figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(WIDTH / DPI, HEIGHT / DPI)
    figure.savefig(
        path,
        dpi=DPI,
        metadata={"Software": "ED-Fire historical replay"},
    )
    plt.close(figure)
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    if struct.unpack(">II", header[16:24]) != (WIDTH, HEIGHT):
        raise ValueError(f"wrong figure dimensions: {path}")


def plot_scores(plt: Any, metrics: dict[str, dict[str, float]]) -> Path:
    figure, axes = plt.subplots(
        2,
        1,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.35, 1.0)},
    )
    positions = np.arange(len(MODEL_IDS))
    for key, label, color in SCORE_SERIES:
        axes[0].plot(
            positions,
            [metrics[model_id][key] for model_id in MODEL_IDS],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=label,
            color=color,
        )
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(positions, MODEL_IDS)
    axes[0].set_ylabel("ILAMB score")
    axes[0].set_title("Official GFED5 score components")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, ncol=5, loc="lower center")

    areas = [metrics[model_id]["mha_per_year"] for model_id in MODEL_IDS]
    reference_field, latitude, longitude = annual_percent(REFERENCE)
    reference_mha = global_mha(reference_field, latitude, longitude)
    bars = axes[1].bar(positions, areas, color="#4C78A8")
    bars[MODEL_IDS.index("F")].set_color("#9D755D")
    bars[MODEL_IDS.index("F")].set_hatch("//")
    axes[1].axhline(
        reference_mha,
        color="#111111",
        linestyle="--",
        linewidth=1.5,
        label=f"GFED5 ({reference_mha:.0f} Mha yr$^{{-1}}$)",
    )
    axes[1].set_ylim(0, 1300)
    axes[1].set_xticks(positions, MODEL_IDS)
    axes[1].set_ylabel("burned area (Mha yr$^{-1}$)")
    axes[1].set_title("Global annual magnitude; Model F is noncomparable")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    for bar, value in zip(bars, areas, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 18,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.suptitle(
        "Historical ED-Fire model ladder, 2001-2016",
        fontsize=16,
    )
    path = OUTPUT / "historical-gfed5-score-ladder.png"
    save(plt, figure, path)
    return path


def plot_maps(plt: Any, metrics: dict[str, dict[str, float]]) -> Path:
    contract = json.loads(CONTRACT.read_text())
    maximum = float(contract["evaluation"]["plot_scales"]["annual_percent_max"])
    reference, latitude, longitude = annual_percent(REFERENCE)
    reference_mha = global_mha(reference, latitude, longitude)
    panels: list[tuple[str, np.ndarray, str]] = [
        ("GFED5", reference, f"{reference_mha:.0f} Mha yr$^{{-1}}$")
    ]
    for model_id in MODEL_IDS:
        field, model_latitude, model_longitude = annual_percent(
            MODEL_ROOT / model_id / "burntArea.nc"
        )
        if not np.array_equal(model_latitude, latitude) or not np.array_equal(
            model_longitude, longitude
        ):
            raise ValueError(f"{model_id} does not use the GFED5 comparison grid")
        panels.append(
            (
                model_id,
                field,
                (
                    f"Overall {metrics[model_id]['overall_score']:.4f} · "
                    f"{metrics[model_id]['mha_per_year']:.0f} Mha yr$^{{-1}}$"
                ),
            )
        )

    figure, axes = plt.subplots(4, 3, constrained_layout=True)
    image = None
    for axis, (title, values, subtitle) in zip(axes.flat, panels, strict=False):
        image = axis.imshow(
            values,
            origin="lower",
            extent=(-180, 180, -90, 90),
            vmin=0,
            vmax=maximum,
            cmap="OrRd",
            aspect="auto",
        )
        axis.set_title(title, fontweight="bold", pad=12)
        axis.text(
            0.5,
            1.01,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
        )
        axis.set_xticks((-180, 0, 180))
        axis.set_yticks((-90, 0, 90))
    unused = list(axes.flat[len(panels) :])
    if unused:
        boundary = unused[0]
        boundary.axis("off")
        boundary.set_title("Protocol boundary", fontweight="bold")
        boundary.text(
            0.05,
            0.82,
            (
                "Historical evidence only\n\n"
                "C, D, F, H: GFED5-derived mask\n"
                "E, G, G6, G7, I, Ibest: GFED4.1s-derived mask\n"
                "F: global magnitude pinned to GFED5\n\n"
                "None is admissible to the clean research line."
            ),
            transform=boundary.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            linespacing=1.5,
        )
    for axis in unused[1:]:
        axis.axis("off")
    if image is None:
        raise RuntimeError("no map panels were created")
    figure.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        shrink=0.65,
        pad=0.04,
        label="burned area (% of grid-cell area yr$^{-1}$)",
    )
    figure.suptitle(
        "Historical GFED5 model fields on one fixed scale, 2001-2016",
        fontsize=16,
    )
    path = OUTPUT / "historical-gfed5-mean-maps.png"
    save(plt, figure, path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fixed historical comparisons and canonical model suites."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Ibest"],
        metavar="MODEL",
        help="Archived model IDs to render, or 'all' (default: Ibest).",
    )
    return parser.parse_args()


def model_registry() -> dict[str, dict[str, Any]]:
    registry = tomllib.loads(REGISTRY.read_text())
    return {model["id"]: model for model in registry["models"]}


def select_models(
    requested: list[str], registry: dict[str, dict[str, Any]]
) -> list[str]:
    if requested == ["all"]:
        return list(registry)
    if "all" in requested:
        raise ValueError("'all' cannot be combined with model IDs")
    unknown = [model_id for model_id in requested if model_id not in registry]
    if unknown:
        raise ValueError("unknown historical model IDs: " + ", ".join(unknown))
    return list(dict.fromkeys(requested))


def model_output(model: dict[str, Any]) -> Path:
    protocol_root = {
        "abc_gfed4_1s": "abc-gfed4.1s",
        "coupled_gfed5": "coupled-gfed5",
    }[model["protocol"]]
    path = REPLAY / protocol_root / "models" / model["id"] / "burntArea.nc"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing replay output for {model['id']}; run reproduce_other_models.py first"
        )
    return path


def evaluation_signature(
    model_id: str,
    candidate: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    evaluation = contract["evaluation"]
    return {
        "model_id": model_id,
        "candidate_sha256": sha256(candidate),
        "stock_sha256": evaluation["stock"]["sha256"],
        "ilamb_version": evaluation["ilamb"]["version"],
        "benchmarks": [
            {
                "id": benchmark["id"],
                "reference_sha256": benchmark["reference_sha256"],
                "config_sha256": sha256(ROOT / benchmark["config"]),
            }
            for benchmark in evaluation["benchmarks"]
        ],
    }


def signature_id(signature: dict[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def evaluate_model(
    model_id: str,
    candidate: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, dict[str, dict[str, float]]], Path]:
    evaluation = contract["evaluation"]
    signature = evaluation_signature(model_id, candidate, contract)
    run_root = EVALUATIONS / model_id / signature_id(signature)
    for directory in ("artifacts", "figures", "logs", "work"):
        (run_root / directory).mkdir(parents=True, exist_ok=True)

    cache_record = run_root / "evaluation.json"
    scalar_paths = {
        benchmark["id"]: run_root
        / "artifacts"
        / "ilamb"
        / benchmark["id"]
        / "scalar_database.csv"
        for benchmark in evaluation["benchmarks"]
    }
    if cache_record.is_file() and all(path.is_file() for path in scalar_paths.values()):
        record = json.loads(cache_record.read_text())
        if record.get("signature") == signature:
            scores_by_model = {"Candidate": {}, "ED-stock": {}}
            for benchmark_id, path in scalar_paths.items():
                scores = read_scores(path)
                for model in scores_by_model:
                    scores_by_model[model][benchmark_id] = scores[model]
            return scores_by_model, run_root

    model_root = run_root / "work" / "evaluation" / "models"
    reference_root = run_root / "work" / "evaluation" / "references"
    link_file(candidate, model_root / "Candidate" / "burntArea.nc")
    link_file(STOCK, model_root / "ED-stock" / "burntArea.nc")
    for benchmark in evaluation["benchmarks"]:
        link_file(
            ROOT / benchmark["reference"],
            reference_root / benchmark["ilamb_source"],
        )

    scores_by_model = {"Candidate": {}, "ED-stock": {}}
    invocations: list[dict[str, Any]] = []
    executable = Path(evaluation["ilamb"]["executable"])
    for benchmark in evaluation["benchmarks"]:
        scores, invocation = run_ilamb(
            project_root=ROOT,
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

    cache_record.write_text(
        json.dumps(
            {
                "schema": "ed-fire-historical-standardized-evaluation/v1",
                "signature": signature,
                "admissible_for_current_research": False,
                "scores": scores_by_model,
                "invocations": invocations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return scores_by_model, run_root


def summarize(
    path: Path,
    contract: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    digest = sha256(path)
    if digest not in cache:
        evaluation = contract["evaluation"]
        period = evaluation["period"]
        cache[digest] = summarize_field(
            path,
            variable_name=contract["candidate_output"]["variable"],
            start_year=int(period["start_year"]),
            end_year=int(period["end_year"]),
            regions=evaluation["regions"],
        )
    return cache[digest]


def common_fields(
    contract: dict[str, Any], cache: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    evaluation = contract["evaluation"]
    fields = {"stock": summarize(STOCK, contract, cache)}
    for benchmark in evaluation["benchmarks"]:
        fields[benchmark["id"]] = summarize(
            ROOT / benchmark["reference"], contract, cache
        )
    return fields


def figure_dimensions(contract: dict[str, Any]) -> dict[str, dict[str, int]]:
    dimensions = {
        figure["filename"]: {
            "width": int(figure["dimensions"]["width"]),
            "height": int(figure["dimensions"]["height"]),
        }
        for figure in contract["figures"]
    }
    if tuple(dimensions) != FIGURE_NAMES:
        raise ValueError(f"canonical figure set changed: {tuple(dimensions)}")
    return dimensions


def write_suite_record(
    output: Path,
    model: dict[str, Any],
    candidate: Path,
    evaluation_root: Path,
    paths: tuple[Path, ...],
    scores: dict[str, dict[str, dict[str, float]]],
    contract: dict[str, Any],
) -> None:
    figure_specs = {figure["filename"]: figure for figure in contract["figures"]}
    record = {
        "schema": "ed-fire-historical-figure-suite/v1",
        "model_id": model["id"],
        "display_name": model["display_name"],
        "protocol": model["protocol"],
        "admissible_for_current_research": False,
        "contract": {
            "id": contract["id"],
            "path": str(CONTRACT.relative_to(ROOT)),
            "sha256": sha256(CONTRACT),
        },
        "candidate_output": {
            "path": str(candidate.relative_to(ROOT)),
            "sha256": sha256(candidate),
        },
        "evaluation": str(evaluation_root.relative_to(ROOT)),
        "scores": scores,
        "figures": [
            {
                "filename": path.name,
                "sha256": sha256(path),
                "width": figure_specs[path.name]["dimensions"]["width"],
                "height": figure_specs[path.name]["dimensions"]["height"],
            }
            for path in paths
        ],
        "style": ["science", "no-latex", "bright"],
        "typography": {
            "family": FONT_FAMILY,
            "files": [
                {
                    "weight": "regular",
                    "path": str(BASIER_REGULAR.relative_to(ROOT)),
                    "sha256": sha256(BASIER_REGULAR),
                },
                {
                    "weight": "semibold",
                    "path": str(BASIER_SEMIBOLD.relative_to(ROOT)),
                    "sha256": sha256(BASIER_SEMIBOLD),
                },
            ],
        },
    }
    (output / "suite.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    args = parse_args()
    registry = model_registry()
    selected = select_models(args.models, registry)
    metrics = load_metrics()
    plt = plotting()
    for path in (plot_scores(plt, metrics), plot_maps(plt, metrics)):
        print(path.relative_to(ROOT))

    contract = json.loads(CONTRACT.read_text())
    dimensions = figure_dimensions(contract)
    summary_cache: dict[str, dict[str, Any]] = {}
    shared_fields = common_fields(contract, summary_cache)
    for model_id in selected:
        model = registry[model_id]
        candidate = model_output(model)
        scores, evaluation_root = evaluate_model(model_id, candidate, contract)
        fields = {
            **shared_fields,
            "candidate": summarize(candidate, contract, summary_cache),
        }
        output = OUTPUT / model_id
        paths = render_suite(
            plt,
            output,
            dimensions,
            scores,
            fields,
            contract["evaluation"]["regions"],
            contract["evaluation"]["plot_scales"],
            candidate_label=model["display_name"],
        )
        write_suite_record(
            output,
            model,
            candidate,
            evaluation_root,
            paths,
            scores,
            contract,
        )
        for path in paths:
            print(path.relative_to(ROOT))
    print("Historical figures are descriptive evidence, not promotion outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
