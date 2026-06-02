"""
Score Lei's params against GFED4.1s on ANNUAL sums (not monthly).
Probe to see whether Lei's reported r ~ 0.63 comes from annual aggregation
rather than monthly cell-level scoring.
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
m = (yrs >= 2001) & (yrs <= 2016)
ds_w = ds.isel(time=m)
years = list(range(2001, 2017))

p = json.load(open(REPO / "from_lei" / "lei_params_unwrapped.json"))["params"]
print(f"Lei params loaded, k1={p['k1']:.4g} D_low={p['D_low']:.4g}")

pred_m, _ = predict_global(ds_w, p)             # monthly fraction
obs_m = load_gfed_halfdeg(years)                # monthly fraction
print(f"shapes pred={pred_m.shape} obs={obs_m.shape}")

# Annual sums per cell, percent units
pred_a = pred_m.reshape(len(years), 12, 360, 720).sum(axis=1) * 100
obs_a  = obs_m.reshape(len(years), 12, 360, 720).sum(axis=1) * 100

lat = ds_w["lat"].values
cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
w2 = np.broadcast_to(cos_lat[:, None], (360, 720))
land_fire = (obs_m > 0).any(axis=0)
w_a = np.broadcast_to((w2 * land_fire)[None, :, :], pred_a.shape)

print("\nAnnual scoring, percent units, cos-lat weighted, GFED-active cells")
M = metrics(pred_a, obs_a, w_a)
for k, v in M.items():
    print(f"  {k:10s} {v}")

# Also try time-mean per cell (16-year average), single map score
pred_mean = pred_a.mean(axis=0)
obs_mean  = obs_a.mean(axis=0)
w_mean = w2 * land_fire
print("\n16-year mean per cell, percent units")
M2 = metrics(pred_mean, obs_mean, w_mean)
for k, v in M2.items():
    print(f"  {k:10s} {v}")
