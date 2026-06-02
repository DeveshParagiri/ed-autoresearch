"""
calibrate_and_score_ffire.py
============================
Stage A finalization:

  1. Compute the GFED reference annual global total (kg/s) from
     ilamb_ref_official/DATA/fFire/GFED4.1S/fFire.nc
  2. For each model in out_ffire_v2/ED-Model{A,B,C}-fFire/fFire.nc, compute its
     total and the scalar multiplier needed to match GFED.  Write calibrated
     copies to out_ffire_v2_cal/.
  3. Copy the GFED land mask + write a JASMIN-style ilamb_ffire.cfg that SKIPS
     RMSE / Seasonal Cycle / Phase / IAV (so Overall = (Bias + Spatial)/2,
     matching the JASMIN fFire page scoring).
  4. Run ilamb-run.
  5. Print a score table comparable to JASMIN's.

Run from repo root:
    python scripts/calibrate_and_score_ffire.py
"""
import os, shutil, subprocess, sys
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

REPO      = Path(__file__).resolve().parents[1]
REF_NC    = REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED4.1S" / "fFire.nc"
SRC_DIR   = REPO / "out_ffire_v2"
CAL_DIR   = REPO / "out_ffire_v2_cal"
ILAMB_DIR = REPO / "ilamb_ref_official"
OUT_DIR   = REPO / "ilamb_out_ffire_v2"
CAL_DIR.mkdir(parents=True, exist_ok=True)

YEARS    = list(range(2001, 2017))
TIMES    = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
TIME_UNITS = "days since 2001-01-01 00:00:00"

def monthly_bnds():
    lo = [cftime.DatetimeNoLeap(y, m, 1) for y in YEARS for m in range(1, 13)]
    hi = []
    for y in YEARS:
        for m in range(1, 13):
            ny = y + (1 if m == 12 else 0); nm = 1 if m == 12 else m + 1
            hi.append(cftime.DatetimeNoLeap(ny, nm, 1))
    return np.array(list(zip(lo, hi)))
TBNDS = monthly_bnds()


def cell_area_05deg(lat):
    R = 6.371e6
    dlon = np.deg2rad(0.5)
    a1d  = (R**2) * dlon * (np.sin(np.deg2rad(lat + 0.25)) -
                             np.sin(np.deg2rad(lat - 0.25)))
    return np.broadcast_to(a1d[:, None], (lat.size, 720)).astype(np.float64)


def global_annual_total_pgyr(ffire_kgm2s, area, secs_per_year=86400*365.0):
    """kg/m²/s × m² × s/yr / 1e12 → Pg/yr, mean across years."""
    m = np.isfinite(ffire_kgm2s)
    return float(np.where(m, ffire_kgm2s, 0.0).mean(axis=0).sum() * area.mean() if False
                 else np.nansum(ffire_kgm2s * area[None,:,:]) * secs_per_year
                      / 1e12 / ffire_kgm2s.shape[0] * 12)


print("=" * 60)
print("calibrate_and_score_ffire.py")
print("=" * 60)

# ── 1. GFED reference total ───────────────────────────────────────────────────
print("\n[1/5] Loading GFED reference and computing total ...")
with xr.open_dataset(REF_NC, decode_times=False) as ds:
    ref = ds["fFire"].values.astype(np.float64)   # (192,360,720) kg/m²/s
    lat = ds["lat"].values
ref = np.where(np.isfinite(ref), ref, 0.0)
area = cell_area_05deg(lat)                        # (360,720) m²

# Mean annual total = (∫∫ flux × area dt) / years
#                   = Σ_t Σ_x  flux(t,x) × area(x) × seconds_in_month(t)
sec_per_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=np.float64) * 86400.0
sec_tile      = np.tile(sec_per_month, 16)         # (192,)
ref_per_cell_month = ref * area[None,:,:] * sec_tile[:,None,None]   # kgC per month per cell
ref_total_pgyr     = ref_per_cell_month.sum() / 1e12 / 16           # Pg/yr avg
print(f"  GFED 2001-2016 global total: {ref_total_pgyr:.3f} Pg/yr  (literature ~2.0)")


# ── 2. Calibrate each model by a single scale factor ──────────────────────────
print("\n[2/5] Calibrating model fFire to match GFED annual total ...")

def total_pgyr(arr):
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr.mean(axis=0).sum() * 0  # placeholder unused

def total_pgyr_real(arr):
    arr0 = np.where(np.isfinite(arr), arr, 0.0)
    return (arr0 * area[None,:,:] * sec_tile[:,None,None]).sum() / 1e12 / 16

scales = {}
for name in ["A", "B", "C"]:
    src = SRC_DIR / f"ED-Model{name}-fFire" / "fFire.nc"
    with xr.open_dataset(src, decode_times=False) as ds:
        mf = ds["fFire"].values.astype(np.float64)
        var_attrs = dict(ds["fFire"].attrs)
        global_attrs = dict(ds.attrs)
    orig = total_pgyr_real(mf)
    k = ref_total_pgyr / orig
    scales[name] = k
    mf_cal = (mf * k).astype(np.float32)
    mf_cal = np.where(np.isfinite(mf), mf_cal, np.nan)
    cal = total_pgyr_real(mf_cal.astype(np.float64))
    print(f"  Model {name}: {orig:.2f} Pg/yr → scale ×{k:.4f} → {cal:.3f} Pg/yr")

    # Write calibrated NC
    dst_dir = CAL_DIR / f"ED-Model{name}-fFire"
    dst_dir.mkdir(parents=True, exist_ok=True)
    var_attrs.update({"calibration_scale": float(k),
                      "calibration_target_Pg_per_yr": float(ref_total_pgyr)})
    out = xr.Dataset(
        {"fFire":        (("time","lat","lon"), mf_cal, var_attrs),
         "time_bounds":  (("time","nv"), TBNDS)},
        coords={"time": ("time", TIMES,
                          {"bounds": "time_bounds", "standard_name": "time", "axis": "T"}),
                "lat":  ("lat", lat.astype(np.float64)),
                "lon":  ("lon", np.arange(-179.75, 180.0, 0.5).astype(np.float64))},
        attrs=global_attrs)
    enc = {"fFire":       {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time":        {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"}}
    tmp = dst_dir / "fFire.nc.tmp"
    out.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, dst_dir / "fFire.nc")


# ── 3. Write JASMIN-style fFire config ────────────────────────────────────────
print("\n[3/5] Writing JASMIN-style ilamb_ffire.cfg (Bias + Spatial only) ...")
cfg = """[h1: Ecosystem and Carbon Cycle]
bgcolor = "#ECFFE6"

[h2: Fire Carbon Emissions]
variable          = "fFire"
weight            = 4
cmap              = "OrRd"
mass_weighting    = True
skip_rmse         = True
skip_iav          = True
skip_cycle        = True

    [GFED4.1S]
    source        = "DATA/fFire/GFED4.1S/fFire.nc"
    weight        = 20
"""
cfg_path = REPO / "ilamb_ffire_v2.cfg"
cfg_path.write_text(cfg)
print(f"  → {cfg_path}")


# ── 4. Run ilamb-run ──────────────────────────────────────────────────────────
print("\n[4/5] Running ilamb-run on calibrated fFire ...")
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
env = os.environ.copy()
env["ILAMB_ROOT"] = str(ILAMB_DIR)
env["PYTHONIOENCODING"] = "utf-8"
ilamb_run = str(Path(sys.executable).parent / "ilamb-run")
cmd = [sys.executable, ilamb_run,
       "--config",     str(cfg_path),
       "--model_root", str(CAL_DIR),
       "--regions",    "global",
       "--build_dir",  str(OUT_DIR)]
r = subprocess.run(cmd, env=env, cwd=str(REPO),
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout[-2000:])
if r.stderr:
    errs = [l for l in r.stderr.splitlines() if "rror" in l.lower() or "raceback" in l.lower()]
    if errs: print("STDERR:", "\n".join(errs[-10:]))


# ── 5. Print JASMIN-style score table ─────────────────────────────────────────
print("\n[5/5] Score table (JASMIN-comparable)\n")
import csv
csv_path = OUT_DIR / "scalar_database.csv"
if not csv_path.exists():
    print("  scalar_database.csv not found"); sys.exit(0)

rows = list(csv.DictReader(open(csv_path)))
print(f"{'Model':20s}  {'Period Mean':>11s}  {'Bias':>8s}  {'BiasScore':>9s}  {'Spatial':>8s}  {'Overall':>8s}")
print("-" * 75)
for m in sorted(set(r["Model"] for r in rows if r["Model"] != "Benchmark")):
    v = {r["ScalarName"]: r["Data"] for r in rows
         if r["Model"] == m and r["Region"] == "global"}
    f = lambda key: (v.get(key, "") or "")
    def num(s):
        try: return f"{float(s):.4f}"
        except: return "  -   "
    # JASMIN-style: only show available scores
    print(f"{m:20s}  {num(f('Period Mean global')):>11s}  "
          f"{num(f('Bias global')):>8s}  "
          f"{num(f('Bias Score')):>9s}  "
          f"{num(f('Spatial Distribution Score')):>8s}  "
          f"{num(f('Overall Score')):>8s}")

print("\nDone. HTML: " + str(OUT_DIR / "EcosystemandCarbonCycle" / "FireCarbonEmissions"))
