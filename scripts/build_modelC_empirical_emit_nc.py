"""Build burntArea.nc for the fFire-tuned Model C variant.

Reads:
  models/C-gfed5-ffire/params.gfed5-ffire.json
  global_baseline_modelC_inputs_1997-2016.nc

Writes:
  ilamb/MODELS_LEADERBOARD/ED-ModelC-fFireTuned/burntArea.nc
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import cftime, numpy as np, xarray as xr

sys.path.insert(0, "scripts")
from refit_modelC_magaware import predict_monthly, load_inputs

REPO = Path(__file__).resolve().parents[1]
PARAMS = REPO / "models" / "C-gfed5-ffire" / "params.gfed5-ffire.json"
INPUT = REPO / "global_baseline_modelC_inputs_1997-2016.nc"
OUT_DIR = REPO / "ilamb" / "MODELS_LEADERBOARD" / "ED-ModelC-fFireTuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "burntArea.nc"

params = json.load(open(PARAMS))["params"]
ds = xr.open_dataset(INPUT)
yr = np.array([d.year for d in ds["time"].values])
m = (yr >= 2001) & (yr <= 2016)
d_in = load_inputs(ds.isel(time=m))
pred = predict_monthly(d_in, params).astype(np.float32)
print(f"pred shape {pred.shape}, mean {np.nanmean(pred):.4g}, max {np.nanmax(pred):.4g}")

t_in = ds["time"].values[m]
lat = ds["lat"].values
lon = ds["lon"].values

tb = np.empty((len(t_in), 2), dtype=object)
for i, t in enumerate(t_in):
    y, mo = t.year, t.month
    tb[i, 0] = cftime.DatetimeNoLeap(y, mo, 1)
    tb[i, 1] = cftime.DatetimeNoLeap(y + (mo == 12), (mo % 12) + 1, 1)

dlat = abs(float(lat[1] - lat[0]))
dlon = abs(float(lon[1] - lon[0]))
lat_b = np.stack([lat - dlat/2, lat + dlat/2], axis=1)
lon_b = np.stack([lon - dlon/2, lon + dlon/2], axis=1)

out = xr.Dataset(
    {"burntArea": (("time", "lat", "lon"), pred,
                    {"units": "1", "long_name": "Burned area fraction",
                     "standard_name": "burned_area_fraction"})},
    coords={"time": t_in, "lat": lat, "lon": lon},
    attrs={"title": "ED-ModelC-fFireTuned (refit against GFED5 fFire)",
           "Conventions": "CF-1.7",
           "params_source": str(PARAMS.relative_to(REPO)),
           "input_source": str(INPUT.name)},
)
out = out.assign(time_bounds=(("time", "nb"), tb),
                 lat_bounds=(("lat", "nb"), lat_b),
                 lon_bounds=(("lon", "nb"), lon_b))
out.time.attrs.update({"bounds": "time_bounds", "standard_name": "time", "axis": "T"})
out.lat.attrs.update({"bounds": "lat_bounds", "units": "degrees_north",
                      "standard_name": "latitude", "axis": "Y"})
out.lon.attrs.update({"bounds": "lon_bounds", "units": "degrees_east",
                      "standard_name": "longitude", "axis": "X"})

t0_year = int(t_in[0].year)
time_units = f"days since {t0_year}-01-01 00:00:00"
enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
       "time":        {"units": time_units, "calendar": "noleap", "dtype": "float64"},
       "time_bounds": {"units": time_units, "calendar": "noleap", "dtype": "float64"}}
tmp = OUT.with_suffix(".nc.tmp")
out.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
os.replace(tmp, OUT)
print(f"wrote {OUT}")
