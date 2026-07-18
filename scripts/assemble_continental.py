"""Assemble paper Model E (per-continent params + fuel form) into one global BA field.

Reads region params from models/paper/E/ (see assembly.json).
Writes ilamb/MODELS/paper/Model-E/burntArea.nc (and legacy MODELS_CONTINENTAL path).

Run:
  python scripts/assemble_continental.py
  ASSEMBLY=ho python scripts/assemble_continental.py   # held-out years train fit
  ASSEMBLY=cell python scripts/assemble_continental.py # held-out cells train fit
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

os.environ.setdefault("SEASONAL_TRANSFORM", "1")
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rmc", REPO / "scripts" / "reproduce_modelC.py")
rmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rmc)

REGION_BOX = {
    "Africa": (-20, 52, -36, 18),
    "S.America": (-82, -34, -56, 14),
    "N.America": (-168, -52, 14, 74),
    "Boreal": (40, 180, 48, 78),
    "SEAsia": (60, 150, -11, 30),
    "Australia": (112, 154, -44, -10),
    "Europe": (-12, 40, 36, 72),
}

# Prefer models/paper/E; fall back to legacy models/C names.
ASSEMBLY = os.environ.get("ASSEMBLY", "best")
_PAPER_E = REPO / "models" / "paper" / "E"
_LEGACY_C = REPO / "models" / "C"

_PRESETS = {
    "best": (
        {
            "Africa": "africa.json",
            "Boreal": "boreal.json",
            "S.America": "samerica.json",
            "SEAsia": "seasia.json",
            "Europe": "europe.json",
        },
        "Model-E",
        "fallback.json",
    ),
    "ho": (
        {
            "Africa": "africa_ho.json",
            "Boreal": "boreal_ho.json",
            "S.America": "samerica_ho.json",
            "SEAsia": "seasia_ho.json",
            "Europe": "europe_ho.json",
        },
        "Model-E-ho",
        "fallback.json",
    ),
    "cell": (
        {
            "Africa": "africa_cell.json",
            "Boreal": "boreal_cell.json",
            "S.America": "samerica_cell.json",
            "SEAsia": "seasia_cell.json",
            "Europe": "europe_cell.json",
        },
        "Model-E-cell",
        "fallback.json",
    ),
}
# Legacy name aliases for old params files under models/C
_LEGACY_ALIAS = {
    "africa.json": "params.africafuel.json",
    "boreal.json": "params.borealseas.json",
    "samerica.json": "params.samerica.json",
    "seasia.json": "params.seasiaseas.json",
    "europe.json": "params.europeseas.json",
    "fallback.json": "params.spatial.k1.json",
    "africa_ho.json": "params.africaho.json",
    "boreal_ho.json": "params.borealho.json",
    "samerica_ho.json": "params.samericaho.json",
    "seasia_ho.json": "params.seasiaho.json",
    "europe_ho.json": "params.europeho.json",
    "africa_cell.json": "params.africacell.json",
    "boreal_cell.json": "params.borealcell.json",
    "samerica_cell.json": "params.samericacell.json",
    "seasia_cell.json": "params.seasiacell.json",
    "europe_cell.json": "params.europecell.json",
}

REGION_PARAMS, OUT_NAME, FALLBACK = _PRESETS[ASSEMBLY]


def resolve_params_file(name: str) -> Path:
    for base in (_PAPER_E, _LEGACY_C):
        p = base / name
        if p.is_file():
            return p
    leg = _LEGACY_ALIAS.get(name)
    if leg:
        p = _LEGACY_C / leg
        if p.is_file():
            return p
    raise FileNotFoundError(f"region params not found: {name}")


def load_params(name: str):
    p = resolve_params_file(name)
    blob = json.load(open(p))
    return blob["params"] if "params" in blob else blob


def transform(rate):
    rc = np.minimum(rate, rmc.FIRE_MAX_RATE)
    return 1.0 - np.exp(-rc / 12.0)


def land_mask_1deg():
    """Prefer existing Model-C land mask; else GFED-active cells."""
    for cand in [
        REPO / "ilamb/MODELS/paper/Model-C/burntArea.nc",
        REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc",
    ]:
        if cand.is_file():
            can = xr.open_dataset(cand)["burntArea"]
            land_hd = np.isfinite(can.values[0])
            land_1d = land_hd.reshape(180, 2, 360, 2).any(axis=(1, 3))
            lat = can.lat.values
            lon = can.lon.values
            can.close()
            return land_1d, lat, lon
    # Fallback: any GFED fire
    obs = rmc.load_gfed_1deg()
    land_1d = (obs > 0).any(axis=0)
    lat = np.arange(-89.75, 90.0, 0.5)
    lon = np.arange(-179.75, 180.0, 0.5)
    return land_1d, lat, lon


def main():
    d = rmc.load_drivers()
    nlat, nlon = 180, 360
    lat1 = -89.5 + np.arange(nlat) * 1.0
    lon1 = -179.5 + np.arange(nlon) * 1.0
    LON, LAT = np.meshgrid(lon1, lat1)

    with np.errstate(over="ignore", invalid="ignore"):
        base = transform(rmc.fire_C(d, load_params(FALLBACK)))
    pred = base.copy()
    assigned = np.zeros((nlat, nlon), bool)

    for reg, pfile in REGION_PARAMS.items():
        try:
            ppath = resolve_params_file(pfile)
        except FileNotFoundError:
            print(f"[skip] {reg}: {pfile} not found")
            continue
        b = REGION_BOX[reg]
        box = (LON >= b[0]) & (LON <= b[1]) & (LAT >= b[2]) & (LAT <= b[3])
        box = box & ~assigned
        with np.errstate(over="ignore", invalid="ignore"):
            pr = transform(rmc.fire_C(d, load_params(pfile)))
        pred[:, box] = pr[:, box]
        assigned |= box
        print(f"[set] {reg}: {int(box.sum())} cells from {ppath.name}")

    land_1d, lat, lon = land_mask_1deg()
    pred_hd = rmc.uncoarsen(np.where(land_1d[None], pred, np.nan).astype(np.float32))
    YEARS = list(range(2001, 2017))
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
    ds = xr.Dataset(
        {
            "burntArea": (
                ("time", "lat", "lon"),
                pred_hd,
                {
                    "units": "1",
                    "standard_name": "burnt_area_fraction",
                    "long_name": "Burnt Area Fraction",
                },
            )
        },
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={
            "title": "Model E (continental + fuel amplitude)",
            "assembly": ASSEMBLY,
            "Conventions": "CF-1.7",
        },
    )
    ds = rmc.add_cf_bounds(ds)
    tu = "days since 2001-01-01 00:00:00"
    enc = {
        "burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
        "time": {"units": tu, "calendar": "noleap", "dtype": "float64"},
        "time_bounds": {"units": tu, "calendar": "noleap", "dtype": "float64"},
    }

    out_paths = [
        REPO / "ilamb" / "MODELS" / "paper" / OUT_NAME / "burntArea.nc",
    ]
    # Keep legacy path for older figure scripts / ILAMB configs
    if ASSEMBLY == "best":
        out_paths.append(
            REPO / "ilamb" / "MODELS_CONTINENTAL" / "ED-ModelC-continental" / "burntArea.nc"
        )
    elif ASSEMBLY == "ho":
        out_paths.append(
            REPO / "ilamb" / "MODELS_CONTINENTAL_HO" / "ED-ModelC-continental-ho" / "burntArea.nc"
        )
    elif ASSEMBLY == "cell":
        out_paths.append(
            REPO
            / "ilamb"
            / "MODELS_CONTINENTAL_CELL"
            / "ED-ModelC-continental-cell"
            / "burntArea.nc"
        )

    for dst in out_paths:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".nc.tmp")
        ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
        os.replace(tmp, dst)
        print(f"[write] {dst.relative_to(REPO)}")


if __name__ == "__main__":
    main()
