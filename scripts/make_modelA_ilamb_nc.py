"""
Build burntArea.nc for Model A from canonical 1deg drivers + best params,
uncoarsen to 0.5deg, write CF-compliant NC for ILAMB leaderboard.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import cftime, numpy as np, xarray as xr
import sys
sys.path.insert(0, "scripts")
from refit_modelA_magaware import fire_A, predict_monthly, load_all_drivers

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "ilamb" / "MODELS_LEADERBOARD" / "ED-ModelA-Ours" / "burntArea.nc"
PJSON = REPO / "models" / "A" / "params.magaware-annual.json"

print("Loading drivers + params ...")
d = load_all_drivers()
params = json.load(open(PJSON))["params"]

print("Predicting at 1deg, uncoarsening to 0.5deg ...")
pred_1 = predict_monthly(d, params)   # (192, 180, 360) monthly fraction
pred_hd = np.repeat(np.repeat(pred_1, 2, axis=1), 2, axis=2).astype(np.float32)

# Land-fire mask: same as ILAMB (cells with any GFED fire)
import h5py
gfed = np.zeros((192, 180, 360), dtype=np.float32)
idx = 0
for yr in range(2001, 2017):
    with h5py.File(REPO / "data" / "gfed" / f"GFED4.1s_{yr}.hdf5", "r") as f:
        for m in range(1, 13):
            arr = f[f"burned_area/{m:02d}/burned_fraction"][:][::-1, :]
            gfed[idx] = arr.reshape(180, 4, 360, 4).mean(axis=(1, 3))
            idx += 1
land = (gfed > 0).any(axis=0)
land_hd = np.repeat(np.repeat(land, 2, axis=0), 2, axis=1)
pred_hd = np.where(land_hd[None, :, :], pred_hd, np.nan).astype(np.float32)

print(f"  pred_hd shape {pred_hd.shape}, mean {np.nanmean(pred_hd):.4g}, max {np.nanmax(pred_hd):.4g}")

# Build CF dataset
times = [cftime.DatetimeNoLeap(y, m, 15) for y in range(2001, 2017) for m in range(1, 13)]
lat = np.arange(-89.75, 90.0, 0.5)
lon = np.arange(-179.75, 180.0, 0.5)
ds = xr.Dataset(
    {"burntArea": (("time", "lat", "lon"), pred_hd,
                   {"units": "1", "standard_name": "burnt_area_fraction",
                    "long_name": "Burnt Area Fraction"})},
    coords={"time": ("time", times), "lat": ("lat", lat), "lon": ("lon", lon)},
    attrs={"title": "ED-ModelA-Ours (8-mechanism, magaware-annual)",
           "Conventions": "CF-1.7",
           "params_source": str(PJSON)}
)
# CF bounds
tb = np.empty((len(times), 2), dtype=object)
for i, t in enumerate(times):
    y, m = t.year, t.month
    tb[i, 0] = cftime.DatetimeNoLeap(y, m, 1)
    tb[i, 1] = cftime.DatetimeNoLeap(y + (m == 12), (m % 12) + 1, 1)
ds = ds.assign(time_bounds=(("time", "nb"), tb))
ds.time.attrs.update({"bounds": "time_bounds", "standard_name": "time", "axis": "T"})
ds = ds.assign(lat_bounds=(("lat", "nb"), np.stack([lat - 0.25, lat + 0.25], axis=1)))
ds.lat.attrs.update({"bounds": "lat_bounds", "units": "degrees_north", "standard_name": "latitude", "axis": "Y"})
ds = ds.assign(lon_bounds=(("lon", "nb"), np.stack([lon - 0.25, lon + 0.25], axis=1)))
ds.lon.attrs.update({"bounds": "lon_bounds", "units": "degrees_east", "standard_name": "longitude", "axis": "X"})

OUT.parent.mkdir(parents=True, exist_ok=True)
time_units = "days since 2001-01-01 00:00:00"
enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
       "time": {"units": time_units, "calendar": "noleap", "dtype": "float64"},
       "time_bounds": {"units": time_units, "calendar": "noleap", "dtype": "float64"}}
tmp = OUT.with_suffix(".nc.tmp")
ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
os.replace(tmp, OUT)
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
