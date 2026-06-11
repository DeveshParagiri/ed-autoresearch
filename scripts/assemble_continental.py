"""
Workstream C assembly: stitch per-continent fitted params into ONE global Model C
prediction (each cell uses its continent's params; cells outside any fitted region
fall back to the global spatial-k1). This is the first cut of the meeting's "one
unified model that explains them" - a piecewise stitch with hard continent borders
(a later step can smooth the seams).

Writes ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc (scoreable with
official ILAMB) and prints the global + per-continent r/sigma/slope via score_spatial
so we can see whether per-continent tuning raised the global correlation r (the wall).

Run: SEASONAL_TRANSFORM=1 python scripts/assemble_continental.py
Region->params mapping is the REGION_PARAMS dict below; edit it as fits complete.
"""
import importlib.util, sys, types, json, os
from pathlib import Path
import numpy as np, xarray as xr, cftime

os.environ.setdefault("SEASONAL_TRANSFORM", "1")
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rmc", REPO / "scripts" / "reproduce_modelC.py")
rmc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rmc)

# Continent boxes (must match the optimizer's _REGION_BOX) and the params file fitted
# for each. Cells in no box use FALLBACK (the global A+B winner, spatial-k1).
REGION_BOX = {
    "Africa": (-20, 52, -36, 18), "S.America": (-82, -34, -56, 14),
    "N.America": (-168, -52, 14, 74), "Boreal": (40, 180, 48, 78),
    "SEAsia": (60, 150, -11, 30), "Australia": (112, 154, -44, -10),
    "Europe": (-12, 40, 36, 72),
}
REGION_PARAMS = {                       # edit as per-continent fits complete
    # Africa: the FUEL-form fit (fuel-scaled amplitude) raised Africa r 0.469 -> 0.664
    # (~the 0.676 driver ceiling) by adding the positive savanna-fuel signal the global
    # GPP hump was missing. This is the FORM change that the plain regional re-tune
    # (params.africa.json) could not achieve. Use it.
    "Africa":    "params.africafuel.json",
    "Boreal":    "params.boreal.json",
    "S.America": "params.samerica.json",
}
FALLBACK = "params.spatial.k1.json"


def load_params(name):
    return json.load(open(REPO / "models" / "C" / name))["params"]


def transform(rate):
    rc = np.minimum(rate, rmc.FIRE_MAX_RATE)
    return 1.0 - np.exp(-rc / 12.0)     # SEASONAL_TRANSFORM form


d = rmc.load_drivers()
nlat, nlon = 180, 360
lat = -89.5 + np.arange(nlat) * 1.0
lon = -179.5 + np.arange(nlon) * 1.0
LON, LAT = np.meshgrid(lon, lat)

# Base prediction = fallback everywhere, then overwrite each fitted region's cells.
with np.errstate(over="ignore", invalid="ignore"):
    base = transform(rmc.fire_C(d, load_params(FALLBACK)))     # (192,180,360)
pred = base.copy()
assigned = np.zeros((nlat, nlon), bool)

for reg, pfile in REGION_PARAMS.items():
    if not (REPO / "models" / "C" / pfile).exists():
        print(f"[skip] {reg}: {pfile} not found yet")
        continue
    b = REGION_BOX[reg]
    box = (LON >= b[0]) & (LON <= b[1]) & (LAT >= b[2]) & (LAT <= b[3])
    box = box & ~assigned                                      # first-come wins on overlap
    with np.errstate(over="ignore", invalid="ignore"):
        pr = transform(rmc.fire_C(d, load_params(pfile)))
    pred[:, box] = pr[:, box]
    assigned |= box
    print(f"[set] {reg}: {int(box.sum())} cells from {pfile}")

# Apply land mask + write a 0.5deg scoreable nc
can = xr.open_dataset(REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc")["burntArea"]
land_hd = np.isfinite(can.values[0])
land_1d = land_hd.reshape(180, 2, 360, 2).any(axis=(1, 3))
pred_hd = rmc.uncoarsen(np.where(land_1d[None], pred, np.nan).astype(np.float32))
YEARS = list(range(2001, 2017))
times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
ds = xr.Dataset(
    {"burntArea": (("time", "lat", "lon"), pred_hd,
                   {"units": "1", "standard_name": "burnt_area_fraction",
                    "long_name": "Burnt Area Fraction"})},
    coords={"time": times, "lat": can.lat.values, "lon": can.lon.values},
    attrs={"title": "ED-ModelC continental (per-continent params stitched)",
           "Conventions": "CF-1.7"})
ds = rmc.add_cf_bounds(ds)
tu = "days since 2001-01-01 00:00:00"
enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
       "time": {"units": tu, "calendar": "noleap", "dtype": "float64"},
       "time_bounds": {"units": tu, "calendar": "noleap", "dtype": "float64"}}
outdir = REPO / "ilamb/MODELS_CONTINENTAL/ED-ModelC-continental"
outdir.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(outdir / "burntArea.nc", encoding=enc, format="NETCDF4_CLASSIC")
print(f"\n[write] {outdir / 'burntArea.nc'}")
print("Score it: python scripts/score_spatial.py "
      "ilamb/MODELS/ED-ModelC-final/burntArea.nc "
      f"{(outdir / 'burntArea.nc').relative_to(REPO).as_posix()}")
print("and official ILAMB on ilamb/MODELS_CONTINENTAL (CLAUDE.md recipe).")
