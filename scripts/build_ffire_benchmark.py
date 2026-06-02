"""
build_ffire_benchmark.py
========================
Build the fire carbon emissions (fFire) benchmark pipeline:

  1. Extract GFED4.1s fire carbon emissions from HDF5 → reference fFire.nc
     (coarsen 0.25-deg → 0.5-deg, convert gC/m2/month → kg/m2/s)

  2. Compute per-cell emission factor from GFED:
     EF [kg_C/m2/s per unit burned_fraction] = GFED_fFire / GFED_burned_fraction

  3. Estimate Model A/B/C fFire = burned_area × EF → write model fFire NC files

  4. Write ILAMB config for fFire confrontation

  5. Run ilamb-run and print scores

Run from repo root:
    python scripts/build_ffire_benchmark.py
"""

import os
import warnings
from pathlib import Path

import cftime
import h5py
import netCDF4 as nc4
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO       = Path(__file__).resolve().parents[1]
GFED_DIR   = REPO / "data" / "gfed"
MODELS_DIR = REPO / "models"
TRENDY_DIR = Path(r"C:\Users\owusu\OneDrive\Documents\UMD STUFF\ED AUTORESEARCH\fire_autoresearch\data\raw\trendy")
ILAMB_ROOT = REPO / "ilamb_ref_official"
OUT_FFIRE  = REPO / "out_ffire"          # model fFire NC files
OUT_DIR    = REPO / "ilamb_out_ffire"    # ILAMB output

for d in [OUT_FFIRE, ILAMB_ROOT / "DATA" / "fFire" / "GFED4.1S"]:
    d.mkdir(parents=True, exist_ok=True)

YEARS     = list(range(2001, 2017))
N_MONTHS  = 192
DAYS_PER_MONTH = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=np.float64)

# standard 0.5-deg grid (S->N)
LAT05 = np.arange(-89.75, 90.0, 0.5)   # (360,)
LON05 = np.arange(-179.75, 180.0, 0.5) # (720,)

print("=" * 60)
print("build_ffire_benchmark.py")
print("=" * 60)


# ── Helper: coarsen 0.25-deg (720,1440) → 0.5-deg (360,720) ──────────────────
def coarsen_025_to_05(arr):
    """Block-mean 2×2 on last two spatial axes."""
    return arr.reshape(*arr.shape[:-2], 360, 2, 720, 2).mean(axis=(-3, -1))


# ── Step 1: Build GFED reference fFire NetCDF ─────────────────────────────────
print("\n[1/4] Building GFED fFire reference NetCDF...")

gfed_ffire  = np.zeros((N_MONTHS, 360, 720), dtype=np.float32)   # kg/m2/s
gfed_ba     = np.zeros((N_MONTHS, 360, 720), dtype=np.float32)   # burned fraction

idx = 0
for yr in YEARS:
    fp = GFED_DIR / f"GFED4.1s_{yr}.hdf5"
    with h5py.File(fp, "r") as f:
        for m in range(1, 13):
            # Carbon emissions: gC/m2/month at 0.25-deg (720,1440) — stored N->S, flip
            c_em = np.array(f[f"emissions/{m:02d}/C"][:], dtype=np.float64)  # (720,1440)
            c_em = c_em[::-1, :]   # flip S->N

            # Burned fraction at 0.25-deg (720,1440) — stored N->S, flip
            ba   = np.array(f[f"burned_area/{m:02d}/burned_fraction"][:], dtype=np.float64)
            ba   = ba[::-1, :]

            # Coarsen to 0.5-deg
            c_05 = coarsen_025_to_05(c_em)   # (360,720) gC/m2/month
            ba_05= coarsen_025_to_05(ba)      # (360,720) fraction

            # Convert gC/m2/month → kg/m2/s
            sec_month = DAYS_PER_MONTH[(m-1) % 12] * 86400.0
            ff_kgs = (c_05 / 1000.0) / sec_month   # kg/m2/s

            gfed_ffire[idx] = np.nan_to_num(ff_kgs,  nan=0.0).astype(np.float32)
            gfed_ba[idx]    = np.nan_to_num(ba_05,   nan=0.0).astype(np.float32)
            idx += 1
    print(f"  {yr} done")

# Mask ocean (cells with zero burned area AND zero emissions throughout)
land_mask = (gfed_ba > 0).any(axis=0) | (gfed_ffire > 0).any(axis=0)
gfed_ffire[:, ~land_mask] = np.nan
gfed_ba[:,    ~land_mask] = np.nan

# Write GFED fFire reference NC
times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
TIME_UNITS = "days since 2001-01-01 00:00:00"
ref_path = ILAMB_ROOT / "DATA" / "fFire" / "GFED4.1S" / "fFire.nc"

ds_ref = xr.Dataset(
    {"fFire": (("time", "lat", "lon"), gfed_ffire,
               {"units": "kg m-2 s-1",
                "long_name": "Fire Carbon Emissions",
                "standard_name": "fire_carbon_flux"})},
    coords={"time": ("time", times), "lat": ("lat", LAT05), "lon": ("lon", LON05)},
    attrs={"title": "GFED4.1s fire carbon emissions 2001-2016",
           "source": "GFED4.1s HDF5 emissions/C, coarsened to 0.5-deg",
           "Conventions": "CF-1.7"})
enc = {"fFire": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
       "time":  {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"}}
ds_ref.to_netcdf(ref_path, encoding=enc, format="NETCDF4_CLASSIC")
print(f"  Saved GFED fFire reference → {ref_path}")

# Global annual total check
lat_rad = np.deg2rad(LAT05)
cell_area = (0.5 * np.pi / 180.0) ** 2 * 6.371e6 ** 2  # approx per 0.5-deg cell
# More accurate: area = R^2 * dlon_rad * (sin(lat+0.25) - sin(lat-0.25))
dlon_rad = np.deg2rad(0.5)
area_1d  = (6.371e6**2) * dlon_rad * (np.sin(np.deg2rad(LAT05 + 0.25)) -
                                        np.sin(np.deg2rad(LAT05 - 0.25)))
area_2d  = np.broadcast_to(area_1d[:, np.newaxis], (360, 720))  # m2

ann_ffire = np.nansum(gfed_ffire[:12]) * area_2d.mean() * 360 * 720
total_pgc = float(np.nansum(gfed_ffire[:12] * area_2d[np.newaxis, :, :]) * 1000 * 86400 * 365 / 12 / 1e15)
print(f"  GFED 2001 annual total fFire: ~{total_pgc:.2f} PgC/yr  (expected ~1.9-2.1)")


# ── Step 2: Compute per-cell emission factor ───────────────────────────────────
print("\n[2/4] Computing per-cell emission factor...")

# EF [kg/m2/s per unit burned_fraction]
# Use multi-year mean to get stable per-cell factor
mean_ff = np.nanmean(gfed_ffire, axis=0)   # (360,720)
mean_ba = np.nanmean(gfed_ba,    axis=0)   # (360,720)

# Avoid division by zero; only compute where burned area exists
ef = np.where(mean_ba > 1e-6, mean_ff / mean_ba, 0.0).astype(np.float32)
print(f"  Emission factor range (non-zero): {ef[ef>0].min():.3e} – {ef[ef>0].max():.3e} kg/m2/s per fraction")
print(f"  Land cells with valid EF: {(ef>0).sum()}")


# ── Step 3: Estimate Model A/B/C fFire and write NC files ─────────────────────
print("\n[3/4] Estimating and writing model fFire NC files...")

def load_model_ba(name):
    p = MODELS_DIR / f"ED-Model{name}-final" / "burntArea.nc"
    ds = xr.open_dataset(p, decode_times=False)
    arr = ds["burntArea"].values.astype(np.float32)
    ds.close()
    return np.where(np.isfinite(arr), arr, 0.0)

def write_ffire_nc(name, ffire_arr):
    dst = OUT_FFIRE / f"ED-Model{name}-fFire"
    dst.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"fFire": (("time", "lat", "lon"), ffire_arr.astype(np.float32),
                   {"units": "kg m-2 s-1",
                    "long_name": "Fire Carbon Emissions",
                    "standard_name": "fire_carbon_flux"})},
        coords={"time": ("time", times), "lat": ("lat", LAT05), "lon": ("lon", LON05)},
        attrs={"title": f"ED-Model{name} fire carbon emissions (burned_area x EF)",
               "Conventions": "CF-1.7"})
    enc = {"fFire": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time":  {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"}}
    tmp = dst / "fFire.nc.tmp"
    ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, dst / "fFire.nc")
    total = float(np.nansum(ffire_arr[:12] * area_2d[np.newaxis, :, :]) * 1000 * 86400 * 365 / 12 / 1e15)
    print(f"  Model {name}: wrote fFire.nc  (2001 annual ~{total:.2f} PgC/yr)")

for name in ["A", "B", "C"]:
    ba = load_model_ba(name)                             # (192,360,720) fraction
    ff = (ba * ef[np.newaxis, :, :]).astype(np.float32) # (192,360,720) kg/m2/s
    ff[:, ~land_mask] = np.nan
    write_ffire_nc(name, ff)


# ── Step 4: Write ILAMB config for fFire ──────────────────────────────────────
print("\n[4/4] Writing ILAMB fFire config...")

cfg_content = """[h1: Ecosystem and Carbon Cycle]
bgcolor = "#ECFFE6"

[h2: Fire Carbon Emissions]
variable       = "fFire"
weight         = 4
cmap           = "OrRd"
mass_weighting = True

    [GFED4.1S]
    source        = "DATA/fFire/GFED4.1S/fFire.nc"
    weight        = 20
"""
cfg_path = REPO / "ilamb_ffire.cfg"
cfg_path.write_text(cfg_content)
print(f"  Saved config → {cfg_path}")

# ── Step 5: Run ILAMB ─────────────────────────────────────────────────────────
print("\n[5/4] Running ilamb-run for fFire...")

import subprocess, sys
env = os.environ.copy()
env["ILAMB_ROOT"] = str(ILAMB_ROOT)

ilamb_run = str(Path(sys.executable).parent / "ilamb-run")
cmd = [sys.executable, ilamb_run,
       "--config",     str(cfg_path),
       "--model_root", str(OUT_FFIRE),
       "--regions",    "global",
       "--build_dir",  str(OUT_DIR)]

r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
if r.stderr:
    errs = [l for l in r.stderr.split("\n") if "Error" in l or "error" in l or "Traceback" in l]
    if errs: print("ERRORS:", "\n".join(errs))

# ── Print scores ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("fFire ILAMB Scores")
print("=" * 60)
csv_path = OUT_DIR / "scalar_database.csv"
if csv_path.exists():
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    for row in rows:
        if row.get("ScalarName") in ("Overall Score", "Bias Score", "RMSE Score",
                                      "Seasonal Cycle Score", "Spatial Distribution Score"):
            print(f"  {row['Model']:25s} {row['ScalarName']:35s} {float(row['Data']):.4f}")
else:
    print("  scalar_database.csv not found — check ilamb_out_ffire/")
