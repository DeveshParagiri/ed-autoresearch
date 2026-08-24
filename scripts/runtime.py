"""Shared host-side loading and candidate-output utilities."""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from types import ModuleType

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "autoresearch" / "model.py"
INPUTS_DIR = ROOT / "autoresearch" / "inputs"
RESULTS_PATH = ROOT / "autoresearch" / "results.tsv"
GFED5_PATH = ROOT / "evals" / "gfed5.nc"
SCORE_QUANTUM = Decimal("0.001")


class ModelError(RuntimeError):
    """A model or prepared-input contract error suitable for CLI output."""


def rounded_score(value: float) -> float:
    """Round a displayed or recorded score to three decimals, half up."""
    if not np.isfinite(value):
        raise ValueError(f"score is not finite: {value}")
    return float(Decimal(str(float(value))).quantize(SCORE_QUANTUM, rounding=ROUND_HALF_UP))


def score_text(value: float, *, signed: bool = False) -> str:
    rounded = rounded_score(value)
    return f"{rounded:+.3f}" if signed else f"{rounded:.3f}"


def load_model(path: Path = MODEL_PATH) -> ModuleType:
    spec = importlib.util.spec_from_file_location("ed_fire_current_model", path)
    if spec is None or spec.loader is None:
        raise ModelError(f"cannot load model: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ModelError(f"cannot import model.py: {error}") from error
    return module


def declared_names(model: ModuleType, attribute: str) -> tuple[str, ...]:
    value = getattr(model, attribute, None)
    if not isinstance(value, (tuple, list)):
        raise ModelError(f"model.py must declare {attribute} as a tuple of names")
    names = tuple(value)
    if any(not isinstance(name, str) or not name for name in names):
        raise ModelError(f"every {attribute} entry must be a nonempty string")
    if len(names) != len(set(names)):
        raise ModelError(f"{attribute} contains duplicate names")
    return names


def validate_model(
    model: ModuleType,
    *,
    require_components: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inputs = declared_names(model, "INPUTS")
    components = declared_names(model, "COMPONENTS")
    if require_components and not components:
        raise ModelError("model.py declares no COMPONENTS to ablate")
    params = getattr(model, "PARAMS", None)
    if not isinstance(params, Mapping):
        raise ModelError("model.py must declare PARAMS as a mapping")
    search_space = getattr(model, "SEARCH_SPACE", None)
    if not isinstance(search_space, Mapping):
        raise ModelError("model.py must declare SEARCH_SPACE as a mapping")
    prediction = getattr(model, "predict", None)
    if not callable(prediction):
        raise ModelError("model.py must define predict(data, params, components)")
    signature = inspect.signature(prediction)
    if "params" not in signature.parameters or "components" not in signature.parameters:
        raise ModelError("predict() must accept the params and components keywords")
    return inputs, components


def _input_index(inputs_dir: Path = INPUTS_DIR) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(inputs_dir.glob("*.nc")):
        with Dataset(path) as dataset:
            for name, variable in dataset.variables.items():
                if variable.dimensions != ("time", "lat", "lon"):
                    continue
                if name in index:
                    raise ModelError(
                        f"input variable {name!r} appears in both "
                        f"{index[name].name} and {path.name}"
                    )
                index[name] = path
    return index


def load_inputs(
    names: Sequence[str],
    inputs_dir: Path = INPUTS_DIR,
) -> dict[str, np.ndarray]:
    index = _input_index(inputs_dir)
    unknown = [name for name in names if name not in index]
    if unknown:
        available = ", ".join(sorted(index))
        raise ModelError(
            f"unknown INPUTS name(s): {', '.join(unknown)}; "
            f"available variables: {available}"
        )

    data: dict[str, np.ndarray] = {}
    for name in names:
        with Dataset(index[name]) as dataset:
            values = np.ma.asarray(dataset.variables[name][:])
        if values.shape != (192, 180, 360):
            raise ModelError(
                f"input {name!r} has shape {values.shape}, expected (192, 180, 360)"
            )
        if np.ma.getmaskarray(values).any():
            raise ModelError(f"input {name!r} contains missing values")
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ModelError(f"input {name!r} contains non-finite values")
        data[name] = array
    return data


def load_land_mask(inputs_dir: Path = INPUTS_DIR) -> np.ndarray:
    """Derive the fixed 1-degree land domain from prepared non-benchmark inputs."""
    climate = inputs_dir / "climate.nc"
    ecosystem = inputs_dir / "ed.nc"
    with Dataset(climate) as dataset:
        precipitation = np.asarray(dataset.variables["annual_precipitation"][:])
        temperature = np.asarray(dataset.variables["air_temperature"][:])
    with Dataset(ecosystem) as dataset:
        natural = np.asarray(dataset.variables["natural_vegetation_fraction"][:])
        secondary = np.asarray(dataset.variables["secondary_vegetation_fraction"][:])
    land = (
        (np.max(precipitation, axis=0) > 0.0)
        | (np.max(np.abs(temperature), axis=0) > 1e-6)
        | (np.max(natural, axis=0) > 0.0)
        | (np.max(secondary, axis=0) > 0.0)
    )
    if land.shape != (180, 360) or not land.any():
        raise ModelError("prepared inputs did not yield a valid land mask")
    return land


def validate_prediction(prediction: np.ndarray) -> np.ndarray:
    array = np.asarray(prediction, dtype=np.float32)
    if array.shape != (192, 180, 360):
        raise ModelError(
            f"predict() returned shape {array.shape}, expected (192, 180, 360)"
        )
    if not np.isfinite(array).all():
        raise ModelError("predict() returned non-finite burned fractions")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < -1e-7 or maximum > 1.0 + 1e-7:
        raise ModelError(
            f"predict() returned burned fractions outside [0, 1]: {minimum}, {maximum}"
        )
    return array


def predict_current(
    model: ModuleType,
    data: Mapping[str, np.ndarray],
    *,
    params: Mapping[str, float] | None = None,
) -> np.ndarray:
    values = dict(model.PARAMS)
    if params is not None:
        values.update(params)
    return validate_prediction(model.predict(data, params=values, components=None))


def write_candidate(prediction: np.ndarray, path: Path) -> None:
    """Write a temporary 0.5-degree monthly candidate for official ILAMB."""
    array = validate_prediction(prediction)
    path.parent.mkdir(parents=True, exist_ok=True)

    month_lengths = np.asarray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    starts = np.concatenate([[0], np.cumsum(np.tile(month_lengths, 16))[:-1]])
    ends = starts + np.tile(month_lengths, 16)
    times = starts + 14.0
    lat = np.arange(-89.75, 90.0, 0.5, dtype=np.float64)
    lon = np.arange(-179.75, 180.0, 0.5, dtype=np.float64)

    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("time", 192)
        dataset.createDimension("lat", 360)
        dataset.createDimension("lon", 720)
        dataset.createDimension("nb", 2)
        dataset.title = "Temporary ED-Fire candidate"
        dataset.Conventions = "CF-1.8"

        time = dataset.createVariable("time", "f8", ("time",))
        time[:] = times
        time.units = "days since 2001-01-01 00:00:00"
        time.calendar = "noleap"
        time.standard_name = "time"
        time.axis = "T"
        time.bounds = "time_bounds"
        time_bounds = dataset.createVariable("time_bounds", "f8", ("time", "nb"))
        time_bounds[:] = np.column_stack([starts, ends])
        time_bounds.units = time.units
        time_bounds.calendar = time.calendar

        latitude = dataset.createVariable("lat", "f8", ("lat",))
        latitude[:] = lat
        latitude.units = "degrees_north"
        latitude.standard_name = "latitude"
        latitude.axis = "Y"
        latitude.bounds = "lat_bounds"
        lat_bounds = dataset.createVariable("lat_bounds", "f8", ("lat", "nb"))
        lat_bounds[:] = np.column_stack([lat - 0.25, lat + 0.25])

        longitude = dataset.createVariable("lon", "f8", ("lon",))
        longitude[:] = lon
        longitude.units = "degrees_east"
        longitude.standard_name = "longitude"
        longitude.axis = "X"
        longitude.bounds = "lon_bounds"
        lon_bounds = dataset.createVariable("lon_bounds", "f8", ("lon", "nb"))
        lon_bounds[:] = np.column_stack([lon - 0.25, lon + 0.25])

        burned = dataset.createVariable(
            "burntArea",
            "f4",
            ("time", "lat", "lon"),
            zlib=True,
            complevel=4,
            chunksizes=(1, 180, 360),
            fill_value=np.float32(1e20),
        )
        burned.units = "1"
        burned.standard_name = "burnt_area_fraction"
        burned.long_name = "burned area fraction"
        for index in range(array.shape[0]):
            burned[index] = np.repeat(np.repeat(array[index], 2, axis=0), 2, axis=1)
