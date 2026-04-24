"""
prep_monthly_inputs.py
Extract and save monthly .npy input arrays needed by reproduce.py.

Reads from:
  - fire_autoresearch/data/raw/crujra/CRUJRA_v3.5_climate_YYYY.nc
  - fire_autoresearch/data/raw/ed_simulation/EDv3_global_simulation_1981_2016.nc
  - fire_autoresearch/data/raw/ed_trendy_v14/  (already in right format)

Writes to:
  - ed-autoresearch-dev/data/crujra/   dbar_monthly.npy, t_deep_monthly.npy, p_ann_monthly.npy
  - ed-autoresearch-dev/data/ed_static/ h_natr_monthly.npy, h_scnd_monthly.npy,
                                         f_natr_monthly.npy, f_scnd_monthly.npy
  - ed-autoresearch-dev/data/trendy_v14/ (symlink/copy from raw)
  - ed-autoresearch-dev/data/gfed/       (symlink/copy from raw)

Shape convention: (N_MONTHS=192, 180, 360) at 1-deg, 2001-2016.
Run once before reproduce.py.
"""

import shutil
from pathlib import Path

import netCDF4 as nc
import numpy as np

# ---- Thornthwaite PET constants ---------------------------------------------
# Days per month (non-leap year)
DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64)
# Julian day of the middle of each month
MID_DOY = np.array([16, 47, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350], dtype=np.float64)
# 1-deg latitude centres (S->N), matches coarsened output
LAT_1DEG = np.arange(-89.5, 90.0, 1.0, dtype=np.float64)  # (180,)

# ---- Paths ------------------------------------------------------------------
import os
REPO      = Path(__file__).resolve().parents[1]
SRC_ROOT  = Path(os.environ.get("ED_RAW_DATA", "/Users/devparagiri/Research/ED/data"))
CRUJRA    = SRC_ROOT / "observations" / "crujra"
ED_SIM    = SRC_ROOT / "ed-simulation" / "EDv3_global_simulation_1981_2016.nc"
TRENDY_V14= SRC_ROOT / "trendy_v14_ed_vars"
GFED_RAW  = SRC_ROOT / "observations" / "gfed"

OUT_CRU   = REPO / "data" / "crujra"
OUT_STATIC= REPO / "data" / "ed_static"
OUT_TRENDY= REPO / "data" / "trendy_v14"
OUT_GFED  = REPO / "data" / "gfed"

YEARS = list(range(2001, 2017))   # 16 years = 192 months
N_MONTHS = 192

for d in [OUT_CRU, OUT_STATIC, OUT_TRENDY, OUT_GFED]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("prep_monthly_inputs.py")
print("=" * 60)


# ---- Helper: coarsen 0.5-deg (360,720) -> 1-deg (180,360) ------------------
def coarsen(arr):
    """Block-mean 2x2 on last two spatial axes."""
    return arr.reshape(*arr.shape[:-2], 180, 2, 360, 2).mean(axis=(-3, -1)).astype(np.float32)


# ---- Thornthwaite PET helpers -----------------------------------------------
def daylight_hours_monthly():
    """Mean photoperiod (hours/day) for each month × 1-deg latitude band.

    Returns: (12, 180) float32 — rows = months Jan-Dec, cols = S->N latitudes.
    Uses solar declination at mid-month and computes the sunrise hour angle.
    Polar day/night clipped to [0, 24].
    """
    lat_rad = np.deg2rad(LAT_1DEG)   # (180,)
    L = np.zeros((12, 180), dtype=np.float64)
    for m in range(12):
        J = MID_DOY[m]
        # Solar declination (radians) — Spencer (1971) approximation
        decl = -np.deg2rad(23.45) * np.cos(2.0 * np.pi * (J + 10.0) / 365.25)
        # Sunrise/sunset hour angle (handle polar day/night)
        cos_omega = -np.tan(lat_rad) * np.tan(decl)
        cos_omega = np.clip(cos_omega, -1.0, 1.0)
        omega = np.arccos(cos_omega)          # (180,) radians
        L[m]  = 2.0 * omega * (24.0 / (2.0 * np.pi))   # convert to hours
    return L.astype(np.float32)   # (12, 180)


# Pre-compute once — doesn't change between years
_L_MONTHLY = daylight_hours_monthly()   # (12, 180)


def thornthwaite_pet(temp_C_12, L_monthly):
    """Thornthwaite (1948) PET for one calendar year.

    Parameters
    ----------
    temp_C_12 : (12, 180, 360) float32  — monthly mean air temp in °C
    L_monthly  : (12, 180)     float32  — photoperiod hours at each latitude

    Returns
    -------
    pet : (12, 180, 360) float32  — PET in mm/month
          Zero where T <= 0 (no evaporation below freezing).
    """
    temp_C_12 = temp_C_12.astype(np.float64)

    # Annual heat index I: sum_m max(T_m/5, 0)^1.514  →  (180, 360)
    I = np.sum(np.maximum(temp_C_12 / 5.0, 0.0) ** 1.514, axis=0)
    I = np.maximum(I, 1e-6)   # avoid /0 in cells with year-round frost

    # Thornthwaite exponent a  →  (180, 360)
    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.79e-2 * I + 0.492

    pet = np.zeros((12, 180, 360), dtype=np.float64)
    for m in range(12):
        T_m  = temp_C_12[m]                              # (180, 360)
        N_m  = DAYS_PER_MONTH[m]                         # days in month
        L_m  = L_monthly[m, :, np.newaxis].astype(np.float64)  # (180, 1)

        mask = T_m > 0.0
        ratio = np.where(mask, 10.0 * T_m / I, 0.0)    # (180, 360)
        # Thornthwaite formula: PET = 16 * (L/12) * (N/30) * (10T/I)^a  [mm/month]
        pet[m] = np.where(
            mask,
            16.0 * (L_m / 12.0) * (N_m / 30.0) * np.power(np.maximum(ratio, 0.0), a),
            0.0
        )

    return pet.astype(np.float32)   # (12, 180, 360)


# ---- 1. CRUJRA monthly arrays -----------------------------------------------
print("\n[1/3] Extracting CRUJRA monthly arrays (dbar, t_deep, p_ann)...")

# Accumulator arrays: (192, 180, 360)
dbar_all    = np.zeros((N_MONTHS, 180, 360), dtype=np.float32)
t_deep_all  = np.zeros((N_MONTHS, 180, 360), dtype=np.float32)
p_ann_all   = np.zeros((N_MONTHS, 180, 360), dtype=np.float32)
t_air_all   = np.zeros((N_MONTHS, 180, 360), dtype=np.float32)
p_month_all = np.zeros((N_MONTHS, 180, 360), dtype=np.float32)
t_surf_all  = np.zeros((N_MONTHS, 180, 360), dtype=np.float32)

# D_bar (dryness index) accumulation state
dbar_state = np.zeros((180, 360), dtype=np.float32)   # running deficit
RESET_PRECIP_MM = 200.0   # reset threshold (mm/month) — matches Dev's pipeline

idx = 0
for yr in YEARS:
    fpath = CRUJRA / f"CRUJRA_v3.5_climate_{yr}.nc"
    ds = nc.Dataset(fpath)

    temp    = np.array(ds.variables['temperature'][:],    dtype=np.float32)  # (12,360,720) K  N->S
    precip  = np.array(ds.variables['precipitation'][:],  dtype=np.float32)  # (12,360,720) mm/month  N->S
    st3     = np.array(ds.variables['soil_temp3'][:],     dtype=np.float32)  # (12,360,720) K  N->S
    st4     = np.array(ds.variables['soil_temp4'][:],     dtype=np.float32)
    st5     = np.array(ds.variables['soil_temp5'][:],     dtype=np.float32)
    st6     = np.array(ds.variables['soil_temp6'][:],     dtype=np.float32)
    ann_p   = np.array(ds.variables['annual_precipitation'][:], dtype=np.float32)  # (360,720)  N->S
    ds.close()

    # CRUJRA v3.5 is stored N->S (lat[0]=89.75). Flip to S->N to match GFED and TRENDY.
    temp   = temp[:,   ::-1, :]
    precip = precip[:, ::-1, :]
    st3    = st3[:,    ::-1, :]
    st4    = st4[:,    ::-1, :]
    st5    = st5[:,    ::-1, :]
    st6    = st6[:,    ::-1, :]
    ann_p  = ann_p[    ::-1, :]

    # Coarsen spatial dims to 1-deg; convert K -> degC; fill NaN (ocean) with 0
    temp_1_C = np.nan_to_num(coarsen(temp) - 273.15, nan=0.0)   # (12,180,360) degC
    precip_1 = np.nan_to_num(coarsen(precip), nan=0.0)
    t_deep_yr = np.nan_to_num(coarsen((st3 + st4 + st5 + st6) / 4.0) - 273.15, nan=0.0)  # K -> C
    ann_p_1  = np.nan_to_num(coarsen(ann_p), nan=0.0)            # (180,360)

    # Thornthwaite PET for the whole year (12, 180, 360)
    pet_yr = thornthwaite_pet(temp_1_C, _L_MONTHLY)

    for m in range(12):
        # Thornthwaite PET replaces old Hamon T*15 approximation
        pet_m = pet_yr[m]
        deficit = pet_m - precip_1[m]

        # Accumulate; reset where monthly precip > RESET_PRECIP_MM
        dbar_state = np.where(precip_1[m] >= RESET_PRECIP_MM,
                              np.maximum(0, deficit),
                              np.maximum(0, dbar_state + deficit))

        dbar_all[idx]   = dbar_state
        t_deep_all[idx] = t_deep_yr[m]
        p_ann_all[idx]  = ann_p_1          # same annual value broadcast each month
        t_air_all[idx]  = temp_1_C[m]     # monthly air temp (degC)
        p_month_all[idx]= precip_1[m]     # monthly precip (mm/month)
        # soil_temp1 is also N->S — flip before coarsen
        st1_raw = np.array(nc.Dataset(fpath).variables['soil_temp1'][m:m+1],
                           dtype=np.float32)[0]   # (360,720) N->S
        t_surf_all[idx] = np.nan_to_num(
            coarsen(st1_raw[::-1, :]) - 273.15, nan=0.0)
        idx += 1

    print(f"  {yr} done (idx={idx})")

np.save(OUT_CRU / "dbar_monthly.npy",    dbar_all)
np.save(OUT_CRU / "t_deep_monthly.npy",  t_deep_all)
np.save(OUT_CRU / "p_ann_monthly.npy",   p_ann_all)
np.save(OUT_CRU / "t_air_monthly.npy",   t_air_all)
np.save(OUT_CRU / "p_month_monthly.npy", p_month_all)
np.save(OUT_CRU / "t_surf_monthly.npy",  t_surf_all)
print(f"  Saved dbar/t_deep/p_ann/t_air/p_month/t_surf monthly -> {OUT_CRU}")
del dbar_all, t_deep_all, p_ann_all, t_air_all, p_month_all, t_surf_all


# ---- 2. ED static monthly arrays from simulation ----------------------------
print("\n[2/3] Extracting ED static monthly arrays from simulation...")

# Simulation spans 1981-2016 (432 months). 2001 = month index 240.
START_IDX = (2001 - 1981) * 12   # = 240

ds = nc.Dataset(ED_SIM)
h_natr = np.array(ds.variables['mean_height_natr'][START_IDX:START_IDX+N_MONTHS],
                  dtype=np.float32)   # (192, 360, 720)
h_scnd = np.array(ds.variables['mean_height_scnd'][START_IDX:START_IDX+N_MONTHS],
                  dtype=np.float32)
f_natr = np.array(ds.variables['frac_natr'][START_IDX:START_IDX+N_MONTHS],
                  dtype=np.float32)
f_scnd = np.array(ds.variables['frac_scnd'][START_IDX:START_IDX+N_MONTHS],
                  dtype=np.float32)
ds.close()

# ED simulation is also N->S (lat[0]=89.75) — flip before coarsen
h_natr_1 = np.nan_to_num(coarsen(h_natr[:, ::-1, :]), nan=0.0)
h_scnd_1 = np.nan_to_num(coarsen(h_scnd[:, ::-1, :]), nan=0.0)
f_natr_1 = np.nan_to_num(coarsen(f_natr[:, ::-1, :]), nan=0.0)
f_scnd_1 = np.nan_to_num(coarsen(f_scnd[:, ::-1, :]), nan=0.0)

np.save(OUT_STATIC / "h_natr_monthly.npy", h_natr_1)
np.save(OUT_STATIC / "h_scnd_monthly.npy", h_scnd_1)
np.save(OUT_STATIC / "f_natr_monthly.npy", f_natr_1)
np.save(OUT_STATIC / "f_scnd_monthly.npy", f_scnd_1)
print(f"  Saved h_natr/h_scnd/f_natr/f_scnd monthly -> {OUT_STATIC}")
del h_natr, h_scnd, f_natr, f_scnd


# ---- 3. Link/copy TRENDY v14 and GFED data ----------------------------------
print("\n[3/3] Copying TRENDY v14 and GFED data links...")

# TRENDY v14 NC files
for src in TRENDY_V14.glob("*.nc"):
    dst = OUT_TRENDY / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"  copied {src.name}")
    else:
        print(f"  already exists: {src.name}")

# GFED HDF5 files
for src in GFED_RAW.glob("*.hdf5"):
    dst = OUT_GFED / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"  copied {src.name}")
    else:
        print(f"  already exists: {src.name}")

# Also copy any .nc GFED files
for src in GFED_RAW.glob("*.nc"):
    dst = OUT_GFED / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"  copied {src.name}")
    else:
        print(f"  already exists: {src.name}")

print("\n" + "=" * 60)
print("prep_monthly_inputs.py DONE")
print("Next: python scripts/reproduce.py")
print("Then: python scripts/score.py")
print("=" * 60)
