"""
compute_ffire.py  — process-based fire carbon flux (Stage A, Option 2)
=====================================================================

For each of our models (A, B, C), produce fFire [kg C m-2 s-1] from:

    fFire = burned_fraction × Σ_k ( C_k × β_k(θ) ) / Δt

following the CLM-Li / SPITFIRE / ED formulation.

Fuel pools (partitioned from ED simulation outputs):
    C_leaf  = 0.05 · AGB          (photosynthesizing foliage)
    C_fine  = 0.10 · AGB          (twigs, dead fine wood < 6 mm)
    C_coarse= 0.85 · AGB          (stems, branches > 6 mm)
    C_litter= 0.15 · soil_C       (surface litter + duff)

Combustion completeness (moisture-dependent, Li et al. 2012 form):
    β_k(θ) = β_k,max · m(θ)
    m(θ)   = clip( D_bar / D_ref , 0, 1 )          # 1 = very dry, 0 = saturated

β_k,max (van Leeuwen 2014, CLM5 defaults):
    leaf    : 0.90
    fine    : 0.80
    coarse  : 0.35
    litter  : 0.80

All tunable constants live in the CONSTANTS block so they can be re-optimized
against GFED fFire in Stage B (re-running Optuna).

Inputs read:
    ed_simulation/EDv3_global_simulation_1981_2016.nc  (AGB, soil_C, LAI)
    models/ED-Model{A,B,C}-final/burntArea.nc          (burned fraction)
    data/crujra/dbar_monthly.npy                        (192,180,360) mm deficit

Output written:
    out_ffire_v2/ED-Model{A,B,C}-fFire/fFire.nc         CF-compliant

Run from repo root:
    python scripts/compute_ffire.py
"""
import os
from pathlib import Path

import cftime
import netCDF4 as nc4
import numpy as np
import xarray as xr

# ── CALIBRATABLE CONSTANTS ────────────────────────────────────────────────────
POOL_FRAC = {          # fraction of AGB (or soil_C for litter) in each pool
    "leaf":   0.05,
    "fine":   0.10,
    "coarse": 0.85,
    "litter": 0.15,    # of soil_C
}
BETA_MAX = {           # max combustion completeness (dry conditions)
    "leaf":   0.90,
    "fine":   0.80,
    "coarse": 0.35,
    "litter": 0.80,
}
D_REF = 150.0          # mm; D_bar above this → fully dry fuel (completeness = β_max)
# ──────────────────────────────────────────────────────────────────────────────

REPO       = Path(__file__).resolve().parents[1]
ED_SIM     = Path(r"C:\Users\owusu\OneDrive\Documents\UMD STUFF\ED AUTORESEARCH\fire_autoresearch\data\raw\ed_simulation\EDv3_global_simulation_1981_2016.nc")
MODELS_DIR = REPO / "models"
DBAR_NPY   = REPO / "data" / "crujra" / "dbar_monthly.npy"
OUT_DIR    = REPO / "out_ffire_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS      = list(range(2001, 2017))
N_MONTHS   = 192
START_IDX  = (2001 - 1981) * 12          # ED sim starts 1981-01 → 2001-01 at idx 240
DAYS_PER_M = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=np.float64)
TIMES      = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
TIME_UNITS = "days since 2001-01-01 00:00:00"


def build_time_bounds():
    lo = [cftime.DatetimeNoLeap(y, m, 1) for y in YEARS for m in range(1, 13)]
    hi = []
    for y in YEARS:
        for m in range(1, 13):
            ny = y + (1 if m == 12 else 0)
            nm = 1 if m == 12 else m + 1
            hi.append(cftime.DatetimeNoLeap(ny, nm, 1))
    return np.array(list(zip(lo, hi)))
TBNDS = build_time_bounds()


def upsample_1deg_to_05deg(arr):
    """Repeat each (…,180,360) cell to (…,360,720). Preserves values."""
    return np.repeat(np.repeat(arr, 2, axis=-2), 2, axis=-1)


print("=" * 60)
print("compute_ffire.py — process-based fFire")
print("=" * 60)

# ── 1. Load ED simulation carbon pools (2001-2016) ─────────────────────────────
print("\n[1/4] Loading ED simulation AGB / soil_C / LAI ...")
with nc4.Dataset(ED_SIM) as ds:
    # Simulation is N→S; flip to S→N to match our models and GFED
    agb    = np.array(ds["AGB"][START_IDX:START_IDX+N_MONTHS, ::-1, :], dtype=np.float32)
    soilC  = np.array(ds["soil_C"][START_IDX:START_IDX+N_MONTHS, ::-1, :], dtype=np.float32)
    lai    = np.array(ds["LAI"][START_IDX:START_IDX+N_MONTHS, ::-1, :], dtype=np.float32)
    ed_lat = np.array(ds["lat"][::-1], dtype=np.float64)
    ed_lon = np.array(ds["lon"][:],    dtype=np.float64)

agb    = np.nan_to_num(agb,   nan=0.0, posinf=0.0, neginf=0.0)
soilC  = np.nan_to_num(soilC, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  AGB    range: {agb.min():.2f} - {agb.max():.2f}   mean={agb.mean():.2f} kgC/m²")
print(f"  soil_C range: {soilC.min():.2f} - {soilC.max():.2f}   mean={soilC.mean():.2f} kgC/m²")
print(f"  LAI    range: {lai.min():.2f} - {lai.max():.2f}   mean={lai.mean():.2f} m²/m²")


# ── 2. Load D-bar, upsample to 0.5° ────────────────────────────────────────────
print("\n[2/4] Loading D-bar monthly and upsampling 1° → 0.5° ...")
dbar_1 = np.load(DBAR_NPY)                            # (192, 180, 360)
dbar   = upsample_1deg_to_05deg(dbar_1).astype(np.float32)  # (192, 360, 720)
print(f"  D-bar shape: {dbar.shape}   range: {dbar.min():.1f} – {dbar.max():.1f} mm")


# ── 3. Build combustion-completeness fields β_k(θ) ────────────────────────────
print("\n[3/4] Computing pool-wise combustion completeness ...")
# m(θ) ∈ [0,1] — dryness factor (1 = fully dry fuel)
m_theta = np.clip(dbar / D_REF, 0.0, 1.0)              # (192, 360, 720)

beta = {k: (BETA_MAX[k] * m_theta).astype(np.float32)    # same shape
        for k in BETA_MAX}
# Summary
for k in beta:
    v = beta[k][beta[k] > 0]
    if v.size:
        print(f"  β_{k:6s}  mean={v.mean():.3f}  max={v.max():.3f}")


# ── 4. Partition carbon pools and compute fFire per model ─────────────────────
print("\n[4/4] Computing fFire per model ...")

# Pre-compute pool carbon densities (don't depend on the burntArea model)
C_pool = {
    "leaf":   POOL_FRAC["leaf"]   * agb,        # kgC/m²
    "fine":   POOL_FRAC["fine"]   * agb,
    "coarse": POOL_FRAC["coarse"] * agb,
    "litter": POOL_FRAC["litter"] * soilC,
}

# Combustible carbon [kgC/m²] that would be released if an entire cell burned in month t
C_combust = np.zeros_like(agb, dtype=np.float32)       # (192, 360, 720)
for k in C_pool:
    C_combust += C_pool[k] * beta[k]
print(f"  Potential combustible C (mean): {C_combust.mean():.3f} kgC/m²")
print(f"  Potential combustible C (95th): {np.quantile(C_combust, 0.95):.3f} kgC/m²")


def load_model_ba(name):
    p = MODELS_DIR / f"ED-Model{name}-final" / "burntArea.nc"
    with xr.open_dataset(p, decode_times=False) as ds:
        arr = ds["burntArea"].values.astype(np.float32)  # (192,360,720) fraction
        lat = ds["lat"].values
        lon = ds["lon"].values
    return np.nan_to_num(arr, nan=0.0), lat, lon


def write_ffire(name, ffire_arr, lat, lon):
    dst = OUT_DIR / f"ED-Model{name}-fFire"
    dst.mkdir(parents=True, exist_ok=True)
    out = xr.Dataset(
        {"fFire":       (("time", "lat", "lon"), ffire_arr.astype(np.float32),
                          {"units": "kg m-2 s-1",
                           "long_name": "Fire Carbon Flux",
                           "standard_name": "fire_carbon_flux"}),
         "time_bounds": (("time", "nv"), TBNDS)},
        coords={"time": ("time", TIMES,
                          {"bounds": "time_bounds", "standard_name": "time", "axis": "T"}),
                "lat":  ("lat", lat.astype(np.float64)),
                "lon":  ("lon", lon.astype(np.float64))},
        attrs={"title": f"ED-Model{name} fFire (process-based, Stage A)",
               "method": "fFire = burned_fraction × Σ_k (POOL_FRAC_k · C_src_k) × β_k(θ) / Δt",
               "constants_POOL_FRAC": str(POOL_FRAC),
               "constants_BETA_MAX":  str(BETA_MAX),
               "constants_D_REF_mm":  D_REF,
               "Conventions": "CF-1.7"})
    enc = {"fFire":       {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time":        {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"}}
    tmp = dst / "fFire.nc.tmp"
    out.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, dst / "fFire.nc")


# Earth-area weights for the 0.5° grid, for PgC/yr diagnostic
dlon_rad = np.deg2rad(0.5)
R = 6.371e6
area_1d  = (R**2) * dlon_rad * (np.sin(np.deg2rad(ed_lat + 0.25)) -
                                 np.sin(np.deg2rad(ed_lat - 0.25)))    # (360,)
area_2d  = np.broadcast_to(area_1d[:, None], (360, 720)).astype(np.float64)

# seconds per month (broadcast to time axis)
sec_per_month = (DAYS_PER_M * 86400.0)                                 # (12,)
sec_tile      = np.tile(sec_per_month, len(YEARS))[:, None, None]       # (192,1,1)

for name in ["A", "B", "C"]:
    ba, lat, lon = load_model_ba(name)                                  # (192,360,720)
    # Fuel consumed this month [kgC/m²/month] = burned_fraction × combustible C
    fuel_consumed = ba * C_combust                                      # (192,360,720)
    # Convert to kgC/m²/s
    ffire = (fuel_consumed / sec_tile).astype(np.float32)

    # Apply GFED land mask (ocean → NaN) by reusing combustible-C>0 or just AGB>0
    land = (agb > 0).any(axis=0)
    ffire[:, ~land] = np.nan

    write_ffire(name, ffire, lat, lon)

    # Diagnostic: annual global total (PgC/yr)
    total_PgC = float(np.nansum(fuel_consumed * area_2d[None, :, :])) / 1e12 / len(YEARS)
    print(f"  Model {name}: fFire.nc written  "
          f"(annual global total ≈ {total_PgC:.2f} PgC/yr)")

print("\n" + "=" * 60)
print(f"Done. Output: {OUT_DIR}")
print("Expected GFED total: ~2.0 PgC/yr globally (Giglio 2013).")
print("If totals are too high, lower β_MAX or D_REF.  Too low: raise them.")
print("These constants are tunable in Stage B Optuna re-run.")
print("=" * 60)
