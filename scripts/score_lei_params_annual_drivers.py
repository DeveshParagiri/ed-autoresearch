"""
Test hypothesis: Lei applies fire_C on ANNUAL-mean drivers (one rate per cell per year),
not on monthly drivers summed afterward.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import xarray as xr
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_modelC_lei import fire_C_per_lu, load_gfed_halfdeg, metrics

REPO = Path(__file__).resolve().parents[1]
ds = xr.open_dataset(REPO / "global_baseline_modelC_inputs_1997-2016.nc")
yrs = np.array([d.year for d in ds["time"].values])
ds_w = ds.isel(time=(yrs >= 2001) & (yrs <= 2016))
years = list(range(2001, 2017))
p = json.load(open(REPO / "from_lei" / "lei_params_unwrapped.json"))["params"]

def to_annual(arr):
    return arr.reshape(16, 12, 360, 720).mean(axis=1)

D = to_annual(ds_w["D_bar"].values)
T = to_annual(ds_w["T_air"].values)
Pa = to_annual(ds_w["P_ann"].values)
Pm = to_annual(ds_w["P_month"].values)

rate_yr = np.zeros_like(D, dtype=np.float32)
for lu in ("ntrl", "scnd", "past"):
    gpp = to_annual(ds_w[f"GPP_month_{lu}"].values)
    frac = to_annual(ds_w[f"area_frac_{lu}"].values)
    r = fire_C_per_lu(D, T, Pa, Pm, gpp, p)
    rate_yr += np.where(np.isfinite(frac), frac, 0.0) * r

rate_capped = np.minimum(rate_yr, 5.0)
pred_annual = (1.0 - np.exp(-rate_capped)) * 100.0   # percent

obs_a = load_gfed_halfdeg(years).reshape(16, 12, 360, 720).sum(axis=1) * 100.0

lat = ds_w["lat"].values
cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
w2 = np.broadcast_to(cos_lat[:, None], (360, 720))

land = (ds_w["area_frac_ntrl"].isel(time=0).values
        + ds_w["area_frac_scnd"].isel(time=0).values
        + ds_w["area_frac_past"].isel(time=0).values) > 0
gfed_active = (obs_a > 0).any(axis=0)

for name, m in [("land only", land), ("GFED-active", gfed_active),
                ("land AND GFED-active", land & gfed_active)]:
    w = (w2 * m)[None, :, :].repeat(16, axis=0)
    M = metrics(pred_annual, obs_a, w)
    print(f"{name:25s} r={M['r']:.4f}  pred%={M['pred_mean']:.4f}  obs%={M['obs_mean']:.4f}  n={M['n']}")
