#!/usr/bin/env python3
"""Shared utilities for the fixed ED-Fire ILAMB evaluation."""

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
