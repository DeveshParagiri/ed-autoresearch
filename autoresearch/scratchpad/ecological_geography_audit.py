"""Audit spatial plausibility without exposing geography to the fire model.

Country polygons and GFED regions are used only after prediction.  They never
enter ``model.py`` or any fit.  The country table includes every Natural Earth
country resolved by at least one 1-degree land-cell centre; tiny countries that
cannot be represented on the model grid are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import shapefile
from netCDF4 import Dataset
from shapely.geometry import box, shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import (  # noqa: E402
    GFED5_PATH,
    load_inputs,
    load_land_mask,
    load_model,
    validate_prediction,
)


DEFAULT_SHP = Path("/tmp/ed-fire-country-audit/ne_10m_admin_0_countries.shp")
DEFAULT_TSV = Path(__file__).with_name("country_ecological_audit.tsv")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def cycle_and_annual(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cycle = values.reshape(16, 12, 180, 360).mean(axis=0)
    return cycle, cycle.sum(axis=0)


def area_statistics(
    mask: np.ndarray,
    model_cycle: np.ndarray,
    model_annual: np.ndarray,
    obs_cycle: np.ndarray,
    obs_annual: np.ndarray,
    area: np.ndarray,
) -> dict[str, float | int | str]:
    weights = area * mask
    model_km2 = float(np.sum(model_annual * weights) / 1e6)
    obs_km2 = float(np.sum(obs_annual * weights) / 1e6)
    model_monthly = np.sum(model_cycle * weights[None, ...], axis=(1, 2))
    obs_monthly = np.sum(obs_cycle * weights[None, ...], axis=(1, 2))
    model_peak = int(np.argmax(model_monthly))
    obs_peak = int(np.argmax(obs_monthly))
    phase = abs(model_peak - obs_peak)
    phase = min(phase, 12 - phase)
    return {
        "cells": int(np.count_nonzero(mask)),
        "model_km2_y": model_km2,
        "obs_km2_y": obs_km2,
        "ratio": model_km2 / obs_km2 if obs_km2 > 1e-9 else float("inf"),
        "model_peak": MONTHS[model_peak],
        "obs_peak": MONTHS[obs_peak],
        "phase_months": phase,
    }


def country_masks(path: Path) -> tuple[dict[tuple[str, str], np.ndarray], list[str]]:
    reader = shapefile.Reader(str(path), encoding="utf-8")
    fields = [field[0] for field in reader.fields[1:]]
    admin_index = fields.index("ADMIN")
    iso_index = fields.index("ADM0_A3")
    records = reader.shapeRecords()
    cell_boxes = [
        box(float(col) - 180.0, float(row) - 90.0, float(col) - 179.0, float(row) - 89.0)
        for row in range(180)
        for col in range(360)
    ]
    tree = STRtree(cell_boxes)

    masks: dict[tuple[str, str], np.ndarray] = {}
    unresolved: list[str] = []
    for index, item in enumerate(records):
        name = str(item.record[admin_index])
        iso3 = str(item.record[iso_index])
        geometry = shape(item.shape.__geo_interface__)
        country_cells = tree.query(geometry, predicate="intersects")
        if not len(country_cells):
            unresolved.append(name)
            continue
        mask = np.zeros(180 * 360, dtype=np.float64)
        for cell in country_cells:
            # Each model cell spans one square degree, so the planar overlap
            # area is also the fractional cell coverage. Latitude-dependent
            # physical area is applied separately below.
            mask[int(cell)] = min(1.0, geometry.intersection(cell_boxes[int(cell)]).area)
        masks[(name, iso3)] = mask.reshape(180, 360)
    return masks, sorted(unresolved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", type=Path, default=DEFAULT_SHP)
    parser.add_argument("--output", type=Path, default=DEFAULT_TSV)
    args = parser.parse_args()

    model = load_model()
    data = load_inputs(model.INPUTS)
    prediction = validate_prediction(model.predict(data, dict(model.PARAMS), None))
    model_cycle, model_annual = cycle_and_annual(prediction)
    with Dataset(GFED5_PATH) as dataset:
        reference = np.asarray(dataset.variables["burntArea"][:192])
    observation = reference.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)) / 100.0
    obs_cycle, obs_annual = cycle_and_annual(observation)

    evaluator = GFED5Evaluator(GFED5_PATH)
    area = evaluator.area.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land = load_land_mask()
    scores = evaluator.score(prediction)

    print("OFFICIAL_REGIONS")
    print("name\toverall\tmodel_km2_y\tobs_km2_y\tratio\tmodel_peak\tobs_peak\tphase_months")
    for name, outside in evaluator.regions.items():
        inside = ~outside.reshape(180, 2, 360, 2).all(axis=(1, 3))
        values = area_statistics(inside & land, model_cycle, model_annual, obs_cycle, obs_annual, area)
        print(
            f"{name}\t{scores[name]['overall_score']:.4f}\t{values['model_km2_y']:.1f}\t"
            f"{values['obs_km2_y']:.1f}\t{values['ratio']:.3f}\t{values['model_peak']}\t"
            f"{values['obs_peak']}\t{values['phase_months']}"
        )

    annual_rain = data["annual_precipitation"].reshape(16, 12, 180, 360).mean(axis=(0, 1))
    temperature = data["air_temperature"].reshape(16, 12, 180, 360).mean(axis=(0, 1))
    lai = data["leaf_area_index"].reshape(16, 12, 180, 360).mean(axis=(0, 1))
    canopy = data["natural_canopy_height"].reshape(16, 12, 180, 360).mean(axis=(0, 1))
    biomass = data["aboveground_biomass"].reshape(16, 12, 180, 360).mean(axis=(0, 1))
    natural = data["natural_vegetation_fraction"].reshape(16, 12, 180, 360).mean(axis=(0, 1))
    primary = data["luh2_primary_fraction"].reshape(16, 12, 180, 360).mean(axis=(0, 1))
    crop = data["luh2_cropland_fraction"].reshape(16, 12, 180, 360).mean(axis=(0, 1))
    range_ = data["luh2_rangeland_fraction"].reshape(16, 12, 180, 360).mean(axis=(0, 1))
    regimes = {
        "intact_tropical_closed_canopy": (temperature >= 20) & (annual_rain >= 1200) & (canopy >= 20) & (lai >= 3) & (natural >= 0.7) & (primary >= 0.5),
        "tropical_closed_canopy": (temperature >= 20) & (annual_rain >= 1200) & (canopy >= 20) & (lai >= 3) & (natural >= 0.7),
        "temperate_closed_canopy": (temperature >= 5) & (temperature < 20) & (canopy >= 15) & (lai >= 2.5) & (natural >= 0.6),
        "boreal_forest": (temperature < 5) & (canopy >= 10) & (natural >= 0.6),
        "tropical_open_woodland": (temperature >= 20) & (annual_rain >= 500) & (annual_rain < 1500) & (canopy >= 5) & (canopy < 20) & (natural >= 0.5),
        "productive_rangeland": (range_ >= 0.4) & (annual_rain >= 250) & (annual_rain < 1500) & (biomass >= 0.2),
        "cropland_dominant": crop >= 0.5,
        "arid_low_fuel": (annual_rain < 250) & (biomass < 0.3) & (lai < 1.0),
    }
    print("ECOLOGICAL_REGIMES")
    print("name\tcells\tmodel_km2_y\tobs_km2_y\tratio\tmodel_peak\tobs_peak\tphase_months")
    for name, mask in regimes.items():
        values = area_statistics(mask & land, model_cycle, model_annual, obs_cycle, obs_annual, area)
        print(
            f"{name}\t{values['cells']}\t{values['model_km2_y']:.1f}\t{values['obs_km2_y']:.1f}\t"
            f"{values['ratio']:.3f}\t{values['model_peak']}\t{values['obs_peak']}\t{values['phase_months']}"
        )

    countries, unresolved = country_masks(args.countries)
    rows: list[dict[str, float | int | str]] = []
    for (name, iso3), mask in countries.items():
        values = area_statistics(mask, model_cycle, model_annual, obs_cycle, obs_annual, area)
        rows.append({"country": name, "iso3": iso3, **values})
    rows.sort(key=lambda row: str(row["country"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    meaningful = [row for row in rows if row["cells"] >= 3 and row["obs_km2_y"] >= 100.0]
    over = sorted(meaningful, key=lambda row: float(row["ratio"]), reverse=True)[:20]
    under = sorted(meaningful, key=lambda row: float(row["ratio"]))[:20]
    print(f"COUNTRIES resolved={len(rows)} meaningful={len(meaningful)} unresolved_at_1deg={len(unresolved)} output={args.output}")
    print("MOST_OVERPREDICTED_MEANINGFUL")
    for row in over:
        print(f"{row['country']}\t{row['ratio']:.3f}\t{row['model_km2_y']:.1f}\t{row['obs_km2_y']:.1f}\t{row['phase_months']}")
    print("MOST_UNDERPREDICTED_MEANINGFUL")
    for row in under:
        print(f"{row['country']}\t{row['ratio']:.3f}\t{row['model_km2_y']:.1f}\t{row['obs_km2_y']:.1f}\t{row['phase_months']}")
    print("UNRESOLVED_AT_1DEG " + "; ".join(unresolved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
