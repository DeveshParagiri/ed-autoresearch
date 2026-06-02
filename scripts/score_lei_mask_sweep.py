"""
Sweep mask choices on annual scoring to see which one reproduces Lei's
r ~ 0.63 and gfed_mean_pct ~ 3.08, pred_mean_pct ~ 1.69.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import xarray as xr
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_modelC_lei import predict_global, load_gfed_halfdeg, metrics

REPO = Path(__file__).resolve().parents[1]
ds = xr.open_dataset(REPO / "global_baseline_modelC_inputs_1997-2016.nc")
yrs = np.array([d.year for d in ds["time"].values])
ds_w = ds.isel(time=(yrs >= 2001) & (yrs <= 2016))
years = list(range(2001, 2017))

p = json.load(open(REPO / "from_lei" / "lei_params_unwrapped.json"))["params"]
pred_m, _ = predict_global(ds_w, p)
obs_m = load_gfed_halfdeg(years)
pred_a = pred_m.reshape(16, 12, 360, 720).sum(axis=1) * 100
obs_a  = obs_m.reshape(16, 12, 360, 720).sum(axis=1) * 100

lat = ds_w["lat"].values
cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
w2 = np.broadcast_to(cos_lat[:, None], (360, 720)).copy()

# Land mask from any landuse fraction > 0 in Lei's NC
land = (ds_w["area_frac_ntrl"].isel(time=0).values
        + ds_w["area_frac_scnd"].isel(time=0).values
        + ds_w["area_frac_past"].isel(time=0).values) > 0
gfed_active = (obs_m > 0).any(axis=0)
gfed_active_strict = (obs_a > 0).any(axis=0)            # any annual activity
gfed_threshold = (obs_a.mean(axis=0) >= 0.01)           # >=0.01% mean burned

masks = {
    "land only": land,
    "GFED-active (any month nonzero)": gfed_active,
    "GFED-active (any year nonzero)": gfed_active_strict,
    "GFED >= 0.01% mean": gfed_threshold,
    "land AND GFED-any-month": land & gfed_active,
}

print(f"{'mask':40s} {'r':>8s} {'pred%':>8s} {'obs%':>8s} {'ncells':>10s}")
for name, m in masks.items():
    w = (w2 * m)[None, :, :].repeat(16, axis=0)
    M = metrics(pred_a, obs_a, w)
    print(f"{name:40s} {M['r']:8.4f} {M['pred_mean']:8.4f} {M['obs_mean']:8.4f} {M['n']:10d}")
