"""Regenerate burned area for the recovered population-term candidates, for official ILAMB.

    C:/Users/owusu/miniforge3/envs/edfire/python.exe scripts/build_pop_candidates.py

Writes ilamb/MODELS_TOPK_coupledPOP/ED-ModelC-coupledPOP-k{1..5}/burntArea.nc

The optimizer's internal score is its own estimate and it has been wrong before. On the last
coupling model it read 0.607 while official ILAMB read 0.6523, so nothing is believed until
it has been through ilamb-run. Five candidates rather than one, because the internal ranking
does not always survive the official scorer.

This mirrors what the optimizer does at the end of a completed run. That run was interrupted,
so the parameters were recovered from the log and the output has to be rebuilt here.
"""
import json
import sys
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

sys.path.insert(0, "scripts")
from reproduce_modelC import (coarsen, uncoarsen, add_cf_bounds, fire_C, trailing_mean)

REPO = Path(__file__).resolve().parents[1]
SL = slice(48, 240)                      # 2001-2016 out of the 1997-2016 dump
YEARS = list(range(2001, 2017))
FIRE_MAX = 5.0
FUEL_WINDOW = 60

# ---- drivers, exactly as the optimizer built them under DUMP_CLIMATE=1 FUEL_WINDOW=60 ----
ds = xr.open_dataset(REPO / "global_baseline_modelC_inputs_1997-2016.nc")
g = lambda n: np.nan_to_num(ds[n].isel(time=SL).values.astype(np.float32), nan=0.0)
d = {"dbar": coarsen(g("D_bar")), "p_ann": coarsen(g("P_ann")),
     "p_month": coarsen(g("P_month")), "t_air": coarsen(g("T_air"))}
gpp_tot = sum(np.clip(g(f"GPP_month_{t}"), 0, None) * g(f"area_frac_{t}")
              for t in ("ntrl", "scnd", "past")).astype(np.float32)
d["gpp_monthly"] = coarsen(gpp_tot)
full = np.zeros_like(ds["GPP_month_ntrl"].values, dtype=np.float32)
for t in ("ntrl", "scnd", "past"):
    full += (np.clip(np.nan_to_num(ds[f"GPP_month_{t}"].values.astype(np.float32), nan=0.0), 0, None)
             * np.nan_to_num(ds[f"area_frac_{t}"].values.astype(np.float32), nan=0.0))
d["gpp_fuel"] = coarsen(trailing_mean(full, FUEL_WINDOW))[SL]
ds.close()
da = xr.open_dataset(REPO / "global_baseline_modelCfuel_inputs_1997-2016.nc")
d["agb"] = coarsen(np.nan_to_num(da["AGB"].isel(time=SL).values.astype(np.float32), nan=0.0))
da.close()

lpop = np.log10(np.load(REPO / "data_human" / "pop_density_1deg_2001_2016.npy").astype(np.float32) + 0.1)

ref = xr.open_dataset(REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc")
obs = coarsen(np.nan_to_num(ref["burntArea"].isel(time=slice(0, 192)).values.astype(np.float32)) / 100.0)
ref.close()
land = (obs > 0).any(axis=0)

_POP = ("pop_amp", "pop_peak", "pop_sig")
lat = np.arange(-89.75, 90.0, 0.5)
lon = np.arange(-179.75, 180.0, 0.5)
times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]

for k in range(1, 6):
    p = json.load(open(REPO / "models" / "C" / f"params.coupledPOP.k{k}.json"))["params"]
    fc = {a: b for a, b in p.items() if a not in _POP}
    with np.errstate(over="ignore", invalid="ignore"):
        rate = fire_C(d, fc)
    mu = np.log10(p["pop_peak"] + 0.1)
    z = (lpop - mu) / (p["pop_sig"] + 1e-9)
    rate = rate * (1.0 + p["pop_amp"] * np.exp(-0.5 * z * z))
    rate = rate * land[None, :, :]
    frac = 1.0 - np.exp(-np.minimum(rate, FIRE_MAX) / 12.0)     # SEASONAL_TRANSFORM=1

    out = xr.Dataset({"burntArea": (("time", "lat", "lon"),
                                    (uncoarsen(frac) * 100.0).astype(np.float32),
                                    {"units": "%", "long_name": "burned area fraction"})},
                     coords={"time": times, "lat": lat, "lon": lon})
    out = add_cf_bounds(out)
    dirp = REPO / "ilamb" / "MODELS_TOPK_coupledPOP" / f"ED-ModelC-coupledPOP-k{k}"
    dirp.mkdir(parents=True, exist_ok=True)
    for stale in dirp.glob("*.nc"):
        stale.unlink()
    tu = "days since 2001-01-01"
    out.to_netcdf(dirp / "burntArea.nc",
                  encoding={"burntArea": {"zlib": True, "complevel": 4},
                            "time": {"units": tu, "calendar": "noleap"},
                            "time_bounds": {"units": tu, "calendar": "noleap"}})
    R = 6371000.0
    la1 = np.arange(-89.5, 90.0, 1.0)
    area = np.abs((R ** 2) * np.deg2rad(1.0)
                  * (np.sin(np.deg2rad(la1 + 0.5)) - np.sin(np.deg2rad(la1 - 0.5))))[:, None]
    mha = float((frac.reshape(16, 12, 180, 360).sum(1).mean(0) * area).sum() / 1e10)
    print(f"  k{k}  pop_amp {p['pop_amp']:6.3f}  peak {p['pop_peak']:7.2f}  -> {mha:5.0f} Mha/yr")

print("\nGFED5 observed is 793 Mha/yr. Now score with official ILAMB.")
