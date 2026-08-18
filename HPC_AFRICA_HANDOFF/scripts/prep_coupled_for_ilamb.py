"""Make Lei's coupled ED burned-area runs scoreable by official ILAMB.

    C:/Users/owusu/miniforge3/envs/edfire/python.exe scripts/prep_coupled_for_ilamb.py

Lei handed over two GCB2026 S3 runs, identical except for the fire scheme: one uses our
Model F, one uses ED's native fire. Scoring both against GFED5 and GFED4.1s answers the
question the whole offline programme rests on, which is whether an offline gain survives
being coupled, where fire cuts the biomass that feeds fire.

The raw files need four things before ILAMB will read them.
  1. They run 1700-2025 at 8 GB. Sliced to each reference window they are a few hundred MB.
  2. Dimensions are latitude/longitude, the references use lat/lon.
  3. Units are "fraction month-1", the references are "%". That is a factor of 100 and it
     is exactly the mistake CLAUDE.md warns about, so the global total is printed for every
     file written and must be read before trusting any score.
  4. No CF bounds, which ILAMB wants.

Writes one model directory per (run, reference) into ilamb/MODELS_COUPLED_<REF>/.
"""
import sys
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

sys.path.insert(0, "scripts")
from reproduce_modelC import add_cf_bounds

REPO = Path(__file__).resolve().parents[1]
R = 6371000.0

RUNS = {"ED-coupled-ModelF":  "D:/GCB2026_coupled_model_F_EDv3_S3_burntArea.nc",
        "ED-coupled-default": "D:/GCB2026_coupled_default_EDv3_S3_burntArea.nc"}

# reference tag -> the years that reference covers. Scoring on the reference's own window
# keeps ILAMB from comparing against months the observation does not have.
#
# COMMON is the third set and it exists for a different reason. The EDv3 burned area sitting
# in the TRENDY leaderboard runs 2001-2016 and burns 2500 Mha/yr, while these coupled runs
# run to 2025 and burn about 175, so putting the three side by side on their own windows
# compares configurations that differ in period as well as in fire scheme. 2001-2016 is the
# overlap of all three and both references, so it is the only window where the three-way
# comparison is controlled.
REFS = {"GFED5": (2001, 2020), "GFED4": (1997, 2016), "COMMON": (2001, 2016)}

# EDv3 as submitted to TRENDY, copied into the COMMON set so it is scored in the same run
EDV3 = REPO / "ilamb" / "MODELS_LEADERBOARD" / "EDv3" / "burntArea.nc"


def cell_area(lat, lon):
    dlon = np.deg2rad(abs(float(lon[1] - lon[0])))
    h = abs(float(lat[1] - lat[0])) / 2.0
    a = (R ** 2) * dlon * (np.sin(np.deg2rad(lat + h)) - np.sin(np.deg2rad(lat - h)))
    return np.abs(a)[:, None] * np.ones((1, len(lon)))


def prep(name, src, ref, y0, yf):
    ds = xr.open_dataset(src)
    yr = ds.time.dt.year.values
    idx = np.where((yr >= y0) & (yr <= yf))[0]
    da = ds["burntArea"].isel(time=idx)

    lat = ds.latitude.values.astype("float64")
    lon = ds.longitude.values.astype("float64")
    a = da.values.astype("float32")

    # fraction month-1 -> percent, matching the reference. Everything downstream, including
    # the ILAMB bias score, depends on this one factor being right.
    pct = a * 100.0

    n_years = yf - y0 + 1
    mha = float((np.nan_to_num(pct).sum(axis=0) / 100.0 / n_years * cell_area(lat, lon)).sum() / 1e10)

    times = [cftime.DatetimeNoLeap(int(t.dt.year), int(t.dt.month), 15) for t in da.time]
    out = xr.Dataset(
        {"burntArea": (("time", "lat", "lon"), pct, {"units": "%", "long_name": "burned area fraction"})},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    out = add_cf_bounds(out)

    d = REPO / "ilamb" / f"MODELS_COUPLED_{ref}" / name
    d.mkdir(parents=True, exist_ok=True)
    for stale in d.glob("*.nc"):          # ILAMB merges every .nc in a model dir
        stale.unlink()
    # time and time_bounds must be written on the SAME units string. Left to itself xarray
    # picks one epoch for the axis and another for the bounds, and ILAMB then builds 193
    # edges for 192 months and dies with a broadcast error. The leaderboard files pin both.
    tunits = f"days since {y0}-01-01"
    out.to_netcdf(d / "burntArea.nc",
                  encoding={"burntArea": {"zlib": True, "complevel": 4},
                            "time": {"units": tunits, "calendar": "noleap"},
                            "time_bounds": {"units": tunits, "calendar": "noleap"}})
    ds.close()
    print(f"  {name:20s} {ref:6s} {y0}-{yf}  {len(idx):4d} months   {mha:6.0f} Mha/yr")


for ref, (y0, yf) in REFS.items():
    print(f"[{ref}]")
    for name, src in RUNS.items():
        prep(name, src, ref, y0, yf)
    if ref == "COMMON":
        import shutil
        d = REPO / "ilamb" / f"MODELS_COUPLED_{ref}" / "EDv3-TRENDY"
        d.mkdir(parents=True, exist_ok=True)
        for stale in d.glob("*.nc"):
            stale.unlink()
        shutil.copy(EDV3, d / "burntArea.nc")
        print(f"  {'EDv3-TRENDY':20s} {ref:6s} {y0}-{yf}   copied from the leaderboard")
print("\nGFED5 observed is 793 Mha/yr over 2001-2016. Check the numbers above before scoring.")
