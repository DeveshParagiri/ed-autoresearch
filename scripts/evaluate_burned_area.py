#!/usr/bin/env python3
"""Evaluate one burned-area candidate under the active ED-Fire contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset, num2date


SCORE_FIELDS = {
    "Benchmark Period Mean (intersection)": "benchmark_period_mean_percent",
    "Model Period Mean (intersection)": "model_period_mean_percent",
    "Bias": "bias_percent",
    "Bias Score": "bias_score",
    "RMSE": "rmse_percent",
    "RMSE Score": "rmse_score",
    "Phase Shift": "phase_shift_months",
    "Seasonal Cycle Score": "seasonal_cycle_score",
    "Spatial Distribution Score": "spatial_distribution_score",
    "Overall Score": "overall_score",
}
SCORE_COMPONENTS = (
    ("bias_score", "Bias"),
    ("rmse_score", "RMSE"),
    ("seasonal_cycle_score", "Seasonal"),
    ("spatial_distribution_score", "Spatial"),
    ("overall_score", "Overall"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected a nonempty relative path, got {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    return path


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes allowed root: {path}")
    return resolved


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def selected_indices(dataset: Dataset, start_year: int, end_year: int) -> list[int]:
    time = dataset.variables.get("time")
    if time is None or not hasattr(time, "units"):
        raise ValueError("burned-area file has no decodable time coordinate")
    dates = num2date(
        time[:],
        time.units,
        getattr(time, "calendar", "standard"),
        only_use_cftime_datetimes=True,
    )
    indices = [index for index, value in enumerate(dates) if start_year <= value.year <= end_year]
    expected = (end_year - start_year + 1) * 12
    if len(indices) != expected:
        raise ValueError(f"expected {expected} monthly values in {start_year}-{end_year}, found {len(indices)}")
    observed = [(dates[index].year, dates[index].month) for index in indices]
    required = [(year, month) for year in range(start_year, end_year + 1) for month in range(1, 13)]
    if observed != required:
        raise ValueError("burned-area time coordinate is not a complete ordered monthly sequence")
    return indices


def region_masks(lat: np.ndarray, lon: np.ndarray, regions: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    longitude, latitude = np.meshgrid(lon, lat)
    masks: dict[str, np.ndarray] = {}
    for region in regions:
        region_id = region["id"]
        bounds = region.get("bounds")
        if bounds is None:
            masks[region_id] = np.ones((lat.size, lon.size), dtype=bool)
            continue
        west, east, south, north = (float(bounds[key]) for key in ("west", "east", "south", "north"))
        masks[region_id] = (
            (longitude >= west)
            & (longitude <= east)
            & (latitude >= south)
            & (latitude <= north)
        )
    return masks


def summarize_field(
    path: Path,
    *,
    variable_name: str,
    start_year: int,
    end_year: int,
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    with Dataset(path) as dataset:
        if variable_name not in dataset.variables:
            raise ValueError(f"{path} does not contain {variable_name}")
        variable = dataset.variables[variable_name]
        if variable.ndim != 3 or variable.dimensions != ("time", "lat", "lon"):
            raise ValueError(f"{path} has unsupported {variable_name} dimensions: {variable.dimensions}")
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        if lat.shape != (360,) or lon.shape != (720,):
            raise ValueError(f"{path} must use the 0.5 degree 360 by 720 grid")
        expected_lat = np.arange(-89.75, 90.0, 0.5)
        expected_lon = np.arange(-179.75, 180.0, 0.5)
        if not np.allclose(lat, expected_lat) or not np.allclose(lon, expected_lon):
            raise ValueError(f"{path} coordinates do not match the locked comparison grid")

        units = str(getattr(variable, "units", "1")).strip().lower()
        if units in {"1", "fraction", "fractional"}:
            scale = 1.0
        elif units in {"%", "percent", "percentage"}:
            scale = 0.01
        else:
            raise ValueError(f"{path} has unsupported burned-area units: {units!r}")

        indices = selected_indices(dataset, start_year, end_year)
        masks = region_masks(lat, lon, regions)
        weights = np.cos(np.deg2rad(lat))[:, None]
        total = np.zeros((lat.size, lon.size), dtype=np.float64)
        count = np.zeros((lat.size, lon.size), dtype=np.uint16)
        seasonal_total = {region["id"]: np.zeros(12, dtype=np.float64) for region in regions}
        seasonal_count = {region["id"]: np.zeros(12, dtype=np.uint16) for region in regions}

        for position, index in enumerate(indices):
            values = np.ma.filled(variable[index], np.nan).astype(np.float64) * scale
            valid = np.isfinite(values)
            if valid.any():
                minimum = float(np.nanmin(values))
                maximum = float(np.nanmax(values))
                if minimum < -1e-7 or maximum > 1.0 + 1e-7:
                    raise ValueError(f"{path} contains burned-area fractions outside [0, 1]")
            total[valid] += values[valid]
            count[valid] += 1
            month = position % 12
            for region in regions:
                region_id = region["id"]
                mask = valid & masks[region_id]
                denominator = float(np.sum(weights * mask))
                if denominator <= 0:
                    continue
                seasonal_total[region_id][month] += float(np.nansum(values * weights * mask) / denominator)
                seasonal_count[region_id][month] += 1

    mean_monthly = np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    seasonal = {
        region["id"]: np.divide(
            seasonal_total[region["id"]],
            seasonal_count[region["id"]],
            out=np.full(12, np.nan, dtype=np.float64),
            where=seasonal_count[region["id"]] > 0,
        )
        * 100.0
        for region in regions
    }
    return {
        "lat": lat,
        "lon": lon,
        "mean_monthly_fraction": mean_monthly,
        "annual_percent": mean_monthly * 1200.0,
        "seasonal_percent": seasonal,
        "regional_annual_percent": {
            region_id: float(np.nansum(values)) for region_id, values in seasonal.items()
        },
    }


def read_scores(path: Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {"Candidate": {}, "ED-stock": {}}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Region") != "global" or row.get("Model") not in scores:
                continue
            key = SCORE_FIELDS.get(row.get("ScalarName", ""))
            if key is None:
                continue
            value = float(row["Data"])
            if not math.isfinite(value):
                raise ValueError(f"ILAMB returned a non-finite {row['ScalarName']} for {row['Model']}")
            scores[row["Model"]][key] = value
    missing: list[str] = []
    for model, values in scores.items():
        for key in SCORE_FIELDS.values():
            if key not in values:
                missing.append(f"{model}.{key}")
    if missing:
        raise ValueError("ILAMB scalar database is incomplete: " + ", ".join(missing))
    return scores


def run_ilamb(
    *,
    project_root: Path,
    run_root: Path,
    executable: Path,
    expected_version: str,
    model_root: Path,
    reference_root: Path,
    benchmark: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    benchmark_id = benchmark["id"]
    build_dir = run_root / "work" / "evaluation" / "ilamb" / benchmark_id
    build_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "--config",
        str(project_root / safe_relative(benchmark["config"])),
        "--model_root",
        str(model_root),
        "--models",
        "Candidate",
        "ED-stock",
        "--regions",
        "global",
        "--build_dir",
        str(build_dir),
        "--title",
        f"ED-Fire {benchmark['label']} locked evaluation",
    ]
    environment = os.environ.copy()
    environment.update(
        ILAMB_ROOT=str(reference_root),
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(run_root / "work" / "mplconfig"),
        PYTHONNOUSERSITE="1",
    )
    (run_root / "work" / "mplconfig").mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = run_root / "logs" / f"ilamb-{benchmark_id}.stdout.log"
    stderr = run_root / "logs" / f"ilamb-{benchmark_id}.stderr.log"
    stdout.write_text(completed.stdout)
    stderr.write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ILAMB {benchmark_id} failed with exit code {completed.returncode}; see {stderr.name}"
        )

    scalar_database = build_dir / "scalar_database.csv"
    if not scalar_database.is_file():
        raise FileNotFoundError(f"ILAMB did not write {scalar_database}")
    durable = run_root / "artifacts" / "ilamb" / benchmark_id
    durable.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scalar_database, durable / "scalar_database.csv")
    for name in ("scores.csv", "scalar_database.json"):
        source = build_dir / name
        if source.is_file():
            shutil.copy2(source, durable / name)

    version_python = executable.with_name("python")
    version_result = subprocess.run(
        [str(version_python), "-c", "import ILAMB; print(ILAMB.__version__)"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    version = version_result.stdout.strip()
    if version != expected_version:
        raise ValueError(f"expected ILAMB {expected_version}, found {version}")
    invocation = {
        "benchmark": benchmark_id,
        "argv": command,
        "exit_code": completed.returncode,
        "stdout": str(stdout.relative_to(run_root)),
        "stderr": str(stderr.relative_to(run_root)),
        "ilamb_version": version,
        "ilamb_executable": str(executable),
        "ilamb_executable_sha256": sha256(executable),
        "scalar_database_sha256": sha256(scalar_database),
    }
    return read_scores(scalar_database), invocation


def configure_plotting(run_root: Path) -> Any:
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(run_root / "work" / "mplconfig")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def save_figure(plt: Any, figure: Any, path: Path, dimensions: dict[str, int]) -> None:
    dpi = 120
    figure.set_size_inches(dimensions["width"] / dpi, dimensions["height"] / dpi)
    figure.savefig(path, dpi=dpi, metadata={"Software": "ED-Fire trusted evaluator"})
    plt.close(figure)
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"figure is not a valid PNG: {path.name}")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != (dimensions["width"], dimensions["height"]):
        raise ValueError(f"figure has wrong dimensions: {path.name} is {width} by {height}")


def plot_score_summary(
    plt: Any,
    path: Path,
    dimensions: dict[str, int],
    scores: dict[str, dict[str, dict[str, float]]],
) -> None:
    figure, axes = plt.subplots(2, 2, constrained_layout=True)
    labels = [label for _, label in SCORE_COMPONENTS]
    positions = np.arange(len(labels))
    for axis, benchmark_id, title in (
        (axes[0, 0], "gfed5", "GFED5"),
        (axes[0, 1], "gfed4_1s", "GFED4.1s"),
    ):
        candidate = [scores["Candidate"][benchmark_id][key] for key, _ in SCORE_COMPONENTS]
        stock = [scores["ED-stock"][benchmark_id][key] for key, _ in SCORE_COMPONENTS]
        axis.bar(positions - 0.18, stock, width=0.36, label="ED-stock", color="#4C78A8")
        axis.bar(positions + 0.18, candidate, width=0.36, label="Candidate", color="#F58518")
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 1)
        axis.set_ylabel("ILAMB score")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)

    axes[1, 0].axis("off")
    table_rows = []
    for model in ("ED-stock", "Candidate"):
        table_rows.append(
            [
                model,
                f"{scores[model]['gfed5']['model_period_mean_percent']:.4f}",
                f"{scores[model]['gfed4_1s']['model_period_mean_percent']:.4f}",
                f"{scores[model]['gfed5']['overall_score']:.4f}",
                f"{scores[model]['gfed4_1s']['overall_score']:.4f}",
            ]
        )
    table = axes[1, 0].table(
        cellText=table_rows,
        colLabels=["Model", "Mean vs G5 (%)", "Mean vs G4 (%)", "G5 overall", "G4 overall"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    axes[1, 0].set_title("Global summary")

    axes[1, 1].axis("off")
    lines = ["Locked evaluation", "Period: 2001-2016", "Region: global", "Official ILAMB scalar database"]
    axes[1, 1].text(0.05, 0.78, "\n".join(lines), va="top", ha="left", fontsize=13, linespacing=1.6)
    figure.suptitle("ED-Fire benchmark score summary", fontsize=16)
    save_figure(plt, figure, path, dimensions)


def plot_spatial_fields(
    plt: Any,
    path: Path,
    dimensions: dict[str, int],
    fields: dict[str, dict[str, Any]],
    scales: dict[str, float],
) -> None:
    figure, axes = plt.subplots(2, 3, constrained_layout=True)
    field_limit = float(scales["annual_percent_max"])
    residual_limit = float(scales["residual_percent_abs"])
    panels = (
        ("GFED5", fields["gfed5"]["annual_percent"], "field"),
        ("GFED4.1s", fields["gfed4_1s"]["annual_percent"], "field"),
        ("ED-stock", fields["stock"]["annual_percent"], "field"),
        ("Candidate", fields["candidate"]["annual_percent"], "field"),
        ("Candidate minus GFED5", fields["candidate"]["annual_percent"] - fields["gfed5"]["annual_percent"], "residual"),
        ("Candidate minus GFED4.1s", fields["candidate"]["annual_percent"] - fields["gfed4_1s"]["annual_percent"], "residual"),
    )
    for axis, (title, values, kind) in zip(axes.flat, panels, strict=True):
        if kind == "field":
            image = axis.imshow(values, origin="lower", extent=(-180, 180, -90, 90), vmin=0, vmax=field_limit, cmap="OrRd", aspect="auto")
            label = "burned area (% yr⁻¹)"
        else:
            image = axis.imshow(values, origin="lower", extent=(-180, 180, -90, 90), vmin=-residual_limit, vmax=residual_limit, cmap="RdBu_r", aspect="auto")
            label = "difference (% yr⁻¹)"
        axis.set_title(title)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
        figure.colorbar(image, ax=axis, orientation="horizontal", shrink=0.82, pad=0.08, label=label)
    figure.suptitle("Mean annual burned area, 2001-2016", fontsize=16)
    save_figure(plt, figure, path, dimensions)


def plot_seasonal_cycles(
    plt: Any,
    path: Path,
    dimensions: dict[str, int],
    fields: dict[str, dict[str, Any]],
    regions: list[dict[str, Any]],
    scales: dict[str, float],
) -> None:
    figure, axes = plt.subplots(2, 4, constrained_layout=True, sharex=True, sharey=True)
    months = np.arange(1, 13)
    styles = (
        ("gfed5", "GFED5", "#111111", "-"),
        ("gfed4_1s", "GFED4.1s", "#777777", "--"),
        ("stock", "ED-stock", "#4C78A8", "-"),
        ("candidate", "Candidate", "#F58518", "-"),
    )
    for axis, region in zip(axes.flat, regions, strict=True):
        region_id = region["id"]
        for field_id, label, color, linestyle in styles:
            axis.plot(
                months,
                fields[field_id]["seasonal_percent"][region_id],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
            )
        axis.set_title(region["label"])
        axis.set_xlim(1, 12)
        axis.set_ylim(0, float(scales["seasonal_percent_max"]))
        axis.set_xticks((1, 3, 5, 7, 9, 11))
        axis.grid(alpha=0.2)
    for axis in axes[1, :]:
        axis.set_xlabel("month")
    for axis in axes[:, 0]:
        axis.set_ylabel("burned area (% month⁻¹)")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    figure.suptitle("Global and regional seasonal cycles, 2001-2016", fontsize=16)
    save_figure(plt, figure, path, dimensions)


def plot_spatial_scatter(
    plt: Any,
    path: Path,
    dimensions: dict[str, int],
    fields: dict[str, dict[str, Any]],
    scales: dict[str, float],
) -> None:
    figure, axes = plt.subplots(2, 2, constrained_layout=True)
    maximum = float(scales["scatter_percent_max"])
    panels = (
        ("gfed5", "candidate", "Candidate vs GFED5"),
        ("gfed4_1s", "candidate", "Candidate vs GFED4.1s"),
        ("gfed5", "stock", "ED-stock vs GFED5"),
        ("gfed4_1s", "stock", "ED-stock vs GFED4.1s"),
    )
    for axis, (reference_id, model_id, title) in zip(axes.flat, panels, strict=True):
        reference = fields[reference_id]["annual_percent"]
        model = fields[model_id]["annual_percent"]
        valid = np.isfinite(reference) & np.isfinite(model) & (reference > 0)
        axis.hexbin(reference[valid], model[valid], gridsize=75, extent=(0, maximum, 0, maximum), mincnt=1, bins="log", cmap="viridis")
        axis.plot((0, maximum), (0, maximum), color="#D62728", linewidth=1.2)
        axis.set_xlim(0, maximum)
        axis.set_ylim(0, maximum)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("benchmark burned area (% yr⁻¹)")
        axis.set_ylabel("model burned area (% yr⁻¹)")
        axis.grid(alpha=0.15)
    figure.suptitle("Cell-level spatial distribution on benchmark fire cells", fontsize=16)
    save_figure(plt, figure, path, dimensions)


def plot_benchmark_sensitivity(
    plt: Any,
    path: Path,
    dimensions: dict[str, int],
    scores: dict[str, dict[str, dict[str, float]]],
    fields: dict[str, dict[str, Any]],
    regions: list[dict[str, Any]],
    scales: dict[str, float],
) -> None:
    figure, axes = plt.subplots(2, 2, constrained_layout=True)
    labels = [label for _, label in SCORE_COMPONENTS]
    positions = np.arange(len(labels))
    for axis, model, title in (
        (axes[0, 0], "Candidate", "Candidate: GFED5 minus GFED4.1s"),
        (axes[0, 1], "ED-stock", "ED-stock: GFED5 minus GFED4.1s"),
    ):
        delta = [scores[model]["gfed5"][key] - scores[model]["gfed4_1s"][key] for key, _ in SCORE_COMPONENTS]
        axis.bar(positions, delta, color="#59A14F")
        axis.axhline(0, color="#222222", linewidth=0.8)
        axis.set_xticks(positions, labels)
        axis.set_ylim(-1, 1)
        axis.set_ylabel("score difference")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)

    region_labels = [region["label"] for region in regions]
    candidate = fields["candidate"]["regional_annual_percent"]
    stock = fields["stock"]["regional_annual_percent"]
    gfed5 = fields["gfed5"]["regional_annual_percent"]
    gfed4 = fields["gfed4_1s"]["regional_annual_percent"]
    bias5 = [candidate[region["id"]] - gfed5[region["id"]] for region in regions]
    bias4 = [candidate[region["id"]] - gfed4[region["id"]] for region in regions]
    width = 0.36
    axes[1, 0].bar(np.arange(len(regions)) - width / 2, bias5, width, label="vs GFED5", color="#4C78A8")
    axes[1, 0].bar(np.arange(len(regions)) + width / 2, bias4, width, label="vs GFED4.1s", color="#9C755F")
    axes[1, 0].axhline(0, color="#222222", linewidth=0.8)
    axes[1, 0].set_xticks(np.arange(len(regions)), region_labels, rotation=35, ha="right")
    axes[1, 0].set_ylim(-float(scales["regional_bias_percent_abs"]), float(scales["regional_bias_percent_abs"]))
    axes[1, 0].set_ylabel("candidate bias (% yr⁻¹)")
    axes[1, 0].set_title("Regional benchmark sensitivity")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.2)

    candidate_delta = [candidate[region["id"]] - stock[region["id"]] for region in regions]
    axes[1, 1].bar(np.arange(len(regions)), candidate_delta, color="#F58518")
    axes[1, 1].axhline(0, color="#222222", linewidth=0.8)
    axes[1, 1].set_xticks(np.arange(len(regions)), region_labels, rotation=35, ha="right")
    axes[1, 1].set_ylim(-float(scales["regional_bias_percent_abs"]), float(scales["regional_bias_percent_abs"]))
    axes[1, 1].set_ylabel("candidate minus stock (% yr⁻¹)")
    axes[1, 1].set_title("Change from native ED")
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.suptitle("Benchmark and regional sensitivity, 2001-2016", fontsize=16)
    save_figure(plt, figure, path, dimensions)


def main() -> int:
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
        raise FileNotFoundError(f"candidate output is missing or is a symlink: {candidate}")
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
            raise ValueError(f"locked benchmark is missing or changed: {benchmark['id']}")
        link_file(source, reference_root / safe_relative(benchmark["ilamb_source"]))

    executable = Path(evaluation["ilamb"]["executable"])
    if not executable.is_file():
        raise FileNotFoundError(f"ILAMB executable is missing: {executable}")
    scores_by_model: dict[str, dict[str, dict[str, float]]] = {"Candidate": {}, "ED-stock": {}}
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
            benchmark["id"]: project_root / safe_relative(benchmark["reference"])
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

    evaluation_record = {
        "schema": "ed-fire-evaluation-record/v1",
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
                "config_sha256": sha256(project_root / safe_relative(benchmark["config"])),
            }
            for benchmark in evaluation["benchmarks"]
        ],
        "invocations": invocations,
        "regions": regions,
        "plot_scales": evaluation["plot_scales"],
    }
    write_json(run_root / "artifacts" / "evaluation.json", evaluation_record)

    plt = configure_plotting(run_root)
    figure_specs = {figure["filename"]: figure for figure in contract["figures"]}
    figures_root = run_root / "figures"
    plot_score_summary(plt, figures_root / "01-score-summary.png", figure_specs["01-score-summary.png"]["dimensions"], scores_by_model)
    plot_spatial_fields(plt, figures_root / "02-mean-burned-area.png", figure_specs["02-mean-burned-area.png"]["dimensions"], fields, evaluation["plot_scales"])
    plot_seasonal_cycles(plt, figures_root / "03-seasonal-cycles.png", figure_specs["03-seasonal-cycles.png"]["dimensions"], fields, regions, evaluation["plot_scales"])
    plot_spatial_scatter(plt, figures_root / "04-spatial-distribution.png", figure_specs["04-spatial-distribution.png"]["dimensions"], fields, evaluation["plot_scales"])
    plot_benchmark_sensitivity(plt, figures_root / "05-benchmark-sensitivity.png", figure_specs["05-benchmark-sensitivity.png"]["dimensions"], scores_by_model, fields, regions, evaluation["plot_scales"])

    print(f"evaluated candidate against GFED5 and GFED4.1s for {start_year}-{end_year}")
    print(f"GFED5 overall={scores_by_model['Candidate']['gfed5']['overall_score']:.6f}")
    print(f"GFED4.1s overall={scores_by_model['Candidate']['gfed4_1s']['overall_score']:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        raise
