"""Resolve the next residual into local ecological states on 9f957d7.

Coordinates assign held blocks only. Candidate masks and reported states use
current inputs or prefix-causal summaries, never regions or coordinates.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402


PINNED = "9f957d7"
MONTH_DAYS = np.tile(
    np.asarray((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), dtype=np.float64),
    16,
)
MONTH_DAYS[np.asarray((3, 7, 11, 15)) * 12 + 1] = 29.0


def load_pinned():
    source = subprocess.run(
        ["git", "show", f"{PINNED}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType(f"model_{PINNED}")
    exec(compile(source, f"{PINNED}:autoresearch/model.py", "exec"), module.__dict__)
    return module


def antecedent(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def residual(observed, predicted, area, mask):
    numerator = float(np.sum(area[mask] * (observed[mask] - predicted[mask])))
    denominator = float(
        np.sum(area[mask] * 0.5 * (observed[mask] + predicted[mask]))
    )
    return numerator / (denominator + 1e-15)


def main() -> None:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    prediction = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    obs_annual = np.average(observed, axis=0, weights=MONTH_DAYS)
    pred_annual = np.average(prediction, axis=0, weights=MONTH_DAYS)
    fire_weight = area * obs_annual
    ranking = np.argsort(fire_weight.ravel())[::-1]
    coverage = np.cumsum(fire_weight.ravel()[ranking]) / fire_weight.sum()
    cells = ranking[: int(np.searchsorted(coverage, 0.90) + 1)]
    rows, cols = cells // 360, cells % 360
    folds = ((rows // 15) + 3 * (cols // 15)) % 4

    def mean(name):
        return np.asarray(data[name][:, rows, cols], dtype=np.float64).mean(axis=0)

    rain = np.clip(
        np.asarray(data["monthly_precipitation"][:, rows, cols], dtype=np.float64),
        0.0,
        None,
    )
    annual_rain = (12.0 * antecedent(rain, 12.0)).mean(axis=0)
    temperature = mean("air_temperature")
    gpp = np.clip(
        np.asarray(data["gpp"][:, rows, cols], dtype=np.float64), 0.0, None
    )
    fine_fuel = antecedent(gpp, 12.0).mean(axis=0)
    fine_fuel = fine_fuel / (fine_fuel + 0.35)
    lai = mean("leaf_area_index")
    biomass = mean("aboveground_biomass")
    canopy = mean("natural_canopy_height")
    secondary_canopy = mean("secondary_canopy_height")
    natural = mean("natural_vegetation_fraction")
    secondary = mean("secondary_vegetation_fraction")
    primary = mean("luh2_primary_fraction")
    crop = mean("luh2_cropland_fraction")
    range_ = mean("luh2_rangeland_fraction")
    pasture = mean("luh2_pasture_fraction")
    urban = mean("luh2_urban_fraction")
    lightning = mean("lightning_flash_rate")
    continuity = 1.0 / (1.0 + 2.0 * crop**1.5 + 5.0 * urban)
    natural_open = natural * 8.0 / (canopy + 8.0)
    secondary_open = secondary * 8.0 / (secondary_canopy + 8.0)
    surface = (
        (1.0 - crop)
        * fine_fuel
        * np.clip(range_ + pasture + natural_open + secondary_open, 0.0, 2.0)
        * continuity
    )
    woody = (
        natural * canopy / (canopy + 8.0)
        + secondary * secondary_canopy / (secondary_canopy + 8.0)
    ) * biomass / (biomass + 1.0)
    crop_capacity = crop * fine_fuel
    surface_share = surface / (0.05 + surface + woody + crop_capacity)
    rain_spread = np.sqrt(
        np.maximum(antecedent(np.square(rain), 12.0) - np.square(antecedent(rain, 12.0)), 0.0)
    ).mean(axis=0)

    states = {
        "warm_low_lai_supported_surface": (
            (temperature >= 15.0)
            & (lai < 1.5)
            & (surface_share >= 0.35)
            & (fine_fuel >= 0.55)
            & (annual_rain >= 300.0)
            & (annual_rain < 1500.0)
        ),
        "warm_low_lai_mixed": (
            (temperature >= 15.0)
            & (lai < 1.5)
            & (surface_share < 0.35)
            & (fine_fuel >= 0.4)
        ),
        "warm_low_productivity_range": (
            (temperature >= 15.0)
            & (range_ >= 0.2)
            & (biomass < 0.5)
            & (annual_rain >= 250.0)
            & (annual_rain < 1500.0)
        ),
        "warm_productive_range": (
            (temperature >= 15.0)
            & (range_ >= 0.2)
            & (biomass >= 0.5)
            & (annual_rain >= 250.0)
            & (annual_rain < 1500.0)
        ),
        "warm_fragmented_low_lai": (
            (temperature >= 15.0)
            & (lai < 1.5)
            & (continuity < 0.75)
            & (fine_fuel >= 0.4)
        ),
        "warm_tall_primary": (
            (temperature >= 15.0) & (canopy >= 15.0) & (primary >= 0.5)
        ),
        "warm_tall_secondary": (
            (temperature >= 15.0)
            & (secondary_canopy >= 10.0)
            & (secondary >= 0.25)
        ),
        "warm_rare_lightning": (temperature >= 15.0) & (lightning < 0.01),
        "cold_rare_lightning": (temperature < 8.0) & (lightning < 0.01),
        "warm_aseasonal_rain": (temperature >= 15.0) & (rain_spread < 35.0),
        "cold_aseasonal_rain": (temperature < 8.0) & (rain_spread < 35.0),
    }
    obs = obs_annual[rows, cols]
    pred = pred_annual[rows, cols]
    selected_area = area[rows, cols]
    selected_total = float(np.sum(selected_area * obs))
    print("ANNUAL_CANDIDATE_STATES share residual held cells")
    for name, mask in states.items():
        share = float(np.sum(selected_area[mask] * obs[mask]) / selected_total)
        held = np.asarray(
            [
                residual(obs, pred, selected_area, mask & (folds == fold))
                for fold in range(4)
            ]
        )
        stable = bool(np.all(held > 0.0) or np.all(held < 0.0))
        print(
            f"{name:34s} share={share:.4f} residual="
            f"{residual(obs,pred,selected_area,mask):+.4f} "
            f"held={'stable' if stable else 'mixed'}:"
            + ",".join(f"{value:+.3f}" for value in held)
            + f" cells={mask.sum()}"
        )


if __name__ == "__main__":
    main()
