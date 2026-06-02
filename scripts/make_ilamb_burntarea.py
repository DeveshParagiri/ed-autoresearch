"""
Generate CF-compliant burntArea.nc for ILAMB leaderboard from a params JSON
applied to Lei's hand-off NC.

Usage:
  python scripts/make_ilamb_burntarea.py \
      --params models/C/params.lei-magaware-annual.json \
      --out ilamb/MODELS_LEADERBOARD/ED-ModelC-Ours/burntArea.nc \
      --title "ED-ModelC-Ours (magaware-annual)"
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import cftime
import numpy as np
import xarray as xr
import sys
sys.path.insert(0, "scripts")
from predict_modelC_lei import predict_global

REPO = Path(__file__).resolve().parents[1]


def add_cf_bounds(ds):
    times = ds.time.values
    tb = np.empty((len(times), 2), dtype=object)
    for i, t in enumerate(times):
        y, m = t.year, t.month
        tb[i, 0] = cftime.DatetimeNoLeap(y, m, 1)
        tb[i, 1] = cftime.DatetimeNoLeap(y + (m == 12), (m % 12) + 1, 1)
    ds = ds.assign(time_bounds=(("time", "nb"), tb))
    ds.time.attrs.update({"bounds": "time_bounds", "standard_name": "time", "axis": "T"})
    lat = ds.lat.values; dlat = abs(float(lat[1] - lat[0]))
    ds = ds.assign(lat_bounds=(("lat", "nb"),
                               np.stack([lat - dlat/2, lat + dlat/2], axis=1)))
    ds.lat.attrs.update({"bounds": "lat_bounds", "units": "degrees_north",
                         "standard_name": "latitude", "axis": "Y"})
    lon = ds.lon.values; dlon = abs(float(lon[1] - lon[0]))
    ds = ds.assign(lon_bounds=(("lon", "nb"),
                               np.stack([lon - dlon/2, lon + dlon/2], axis=1)))
    ds.lon.attrs.update({"bounds": "lon_bounds", "units": "degrees_east",
                         "standard_name": "longitude", "axis": "X"})
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="JSON with 'params' or top-level params")
    ap.add_argument("--input", default="global_baseline_modelC_inputs_1997-2016.nc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="ED-ModelC")
    ap.add_argument("--year-start", type=int, default=2001)
    ap.add_argument("--year-end", type=int, default=2016)
    args = ap.parse_args()

    print(f"Loading {args.input}")
    ds = xr.open_dataset(REPO / args.input)
    yrs = np.array([d.year for d in ds["time"].values])
    mask_t = (yrs >= args.year_start) & (yrs <= args.year_end)
    ds_w = ds.isel(time=mask_t)

    pj = json.load(open(args.params))
    params = pj.get("params", pj.get("best_params", pj))
    print(f"Predicting with params: D_low={params['D_low']:.4g} k1={params['k1']:.4g} ...")

    pred, rate = predict_global(ds_w, params)
    print(f"  rate yr_mean={float(np.nanmean(rate)):.4g}  monthly mean={float(np.nanmean(pred)):.4g}")

    # Build CF dataset
    times = [cftime.DatetimeNoLeap(y, m, 15)
             for y in range(args.year_start, args.year_end + 1)
             for m in range(1, 13)]
    out = xr.Dataset(
        {"burntArea": (("time", "lat", "lon"), pred,
                       {"units": "1", "standard_name": "burnt_area_fraction",
                        "long_name": "Burnt Area Fraction"})},
        coords={"time": ("time", times),
                 "lat":  ("lat",  ds_w.lat.values),
                 "lon":  ("lon",  ds_w.lon.values)},
        attrs={"title": args.title,
               "Conventions": "CF-1.7",
               "params_source": str(args.params),
               "input_source": args.input}
    )
    out = add_cf_bounds(out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_units = f"days since {args.year_start}-01-01 00:00:00"
    enc = {"burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time": {"units": time_units, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": time_units, "calendar": "noleap", "dtype": "float64"}}
    tmp = out_path.with_suffix(".nc.tmp")
    out.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
