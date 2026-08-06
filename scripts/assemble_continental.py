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

os.environ.setdefault("SEASONAL_TRANSFORM", "1")   # callers override for C/D-form assemblies (Model G)
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
# ASSEMBLY preset (env): "best" = the production continental model; "ho"/"cell" =
# the held-out-years / held-out-cells train-fit variants (for validation only).
# N.America and Australia are kept on the global fallback (their fits regressed r).
ASSEMBLY = os.environ.get("ASSEMBLY", "best")
_PRESETS = {
    "best": ({"Africa": "params.africafuel.json", "Boreal": "params.borealseas.json",
              "S.America": "params.samerica.json", "SEAsia": "params.seasiaseas.json",
              "Europe": "params.europeseas.json"},
             "ED-ModelC-continental"),
    "ho":   ({"Africa": "params.africaho.json", "Boreal": "params.borealho.json",
              "S.America": "params.samericaho.json", "SEAsia": "params.seasiaho.json",
              "Europe": "params.europeho.json"},
             "ED-ModelC-continental-ho"),
    "cell": ({"Africa": "params.africacell.json", "Boreal": "params.borealcell.json",
              "S.America": "params.samericacell.json", "SEAsia": "params.seasiacell.json",
              "Europe": "params.europecell.json"},
             "ED-ModelC-continental-cell"),
    # "clean" = the single-lever paper E: identical FORM changes to "best" but every
    # region fit with the PURE-SPATIAL objective (SEAS_W=0), matching Model D's objective
    # exactly. Only the 3 seasonal-blended regions differ from "best" (borealseas->boreal,
    # seasiaseas->seasia, europeseas->europe); Africa/S.America were already pure-spatial.
    # This makes the D->E step change the functional-form lever alone (objective held).
    # "G" = Model G, Model C's 12-param form and C's S_overall objective, fitted
    # PER CONTINENT (2026-08-06). Isolates the spatial-parameterization column, which
    # D->E confounds with the form change. Legacy transform and Model C as the fallback,
    # so the ONLY difference from Model C is that parameters vary by region.
    "G":     ({"Africa": "params.G_Africa.json", "Boreal": "params.G_Boreal.json",
               "S.America": "params.G_SAmerica.json", "SEAsia": "params.G_SEAsia.json",
               "Europe": "params.G_Europe.json"},
              "ED-ModelG-continental"),
    # "G7" = Model G with ALL SEVEN regions fitted. The five-region "G" preset above exists
    # only to match Model E's clean assembly cell-for-cell. G's entire remaining excess sat in
    # N.America (3.54x observed) and Australia, the two regions E leaves on the global fallback,
    # so those were fitted too on 2026-08-06.
    "G7":    ({"Africa": "params.G_Africa.json", "Boreal": "params.G_Boreal.json",
               "S.America": "params.G_SAmerica.json", "SEAsia": "params.G_SEAsia.json",
               "Europe": "params.G_Europe.json", "N.America": "params.G_NAmerica.json",
               "Australia": "params.G_Australia.json"},
              "ED-ModelG7-continental"),
    "clean": ({"Africa": "params.africafuel.json", "Boreal": "params.boreal.json",
               "S.America": "params.samerica.json", "SEAsia": "params.seasia.json",
               "Europe": "params.europe.json"},
              "ED-ModelC-continental-clean"),
}
REGION_PARAMS, _OUT_NAME = _PRESETS[ASSEMBLY]
FALLBACK = os.environ.get("ASSEMBLE_FALLBACK", "params.spatial.k1.json")


def load_params(name):
    return json.load(open(REPO / "models" / "C" / name))["params"]


_SEASONAL = os.environ.get("SEASONAL_TRANSFORM", "1") == "1"


def transform(rate):
    rc = np.minimum(rate, rmc.FIRE_MAX_RATE)
    if _SEASONAL:
        return 1.0 - np.exp(-rc / 12.0)          # E/F form (per-month disturbance)
    return (1.0 - np.exp(-rc)) / 12.0            # C/D legacy form (even spread)


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
_PARENT = {"best": "MODELS_CONTINENTAL", "ho": "MODELS_CONTINENTAL_HO",
           "cell": "MODELS_CONTINENTAL_CELL",
           "clean": "MODELS_CONTINENTAL_CLEAN",
           "G": "MODELS_CONTINENTAL_G",
           "G7": "MODELS_CONTINENTAL_G7"}[ASSEMBLY]
# ASSEMBLE_OUTDIR (env) overrides the output dir, e.g. to regenerate E into the paper
# folder without touching the existing MODELS_CONTINENTAL outputs.
_OUTDIR_ENV = os.environ.get("ASSEMBLE_OUTDIR", "")
outdir = Path(_OUTDIR_ENV) if _OUTDIR_ENV else REPO / "ilamb" / _PARENT / _OUT_NAME
outdir.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(outdir / "burntArea.nc", encoding=enc, format="NETCDF4_CLASSIC")
print(f"\n[write] {outdir / 'burntArea.nc'}")
print("Score it: python scripts/score_spatial.py "
      "ilamb/MODELS/ED-ModelC-final/burntArea.nc "
      f"{(outdir / 'burntArea.nc').relative_to(REPO).as_posix()}")
print("and official ILAMB on ilamb/MODELS_CONTINENTAL (CLAUDE.md recipe).")
