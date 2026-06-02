"""
Maps comparing GFED truth, Model C (magaware-annual), and Model C-fuel (preliminary).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import sys
sys.path.insert(0, "scripts")
from predict_modelC_lei import predict_global as predict_C, load_gfed_halfdeg
from refit_modelCfuel_magaware import predict_monthly as predict_Cfuel, load_inputs as load_Cfuel_inputs

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "Cfuel"
OUT.mkdir(parents=True, exist_ok=True)

# Inputs
ds_C  = xr.open_dataset(REPO / "global_baseline_modelC_inputs_1997-2016.nc")
ds_Cf = xr.open_dataset(REPO / "global_baseline_modelCfuel_inputs_1997-2016.nc")
yrs = np.array([d.year for d in ds_C["time"].values])
m = (yrs >= 2001) & (yrs <= 2016)
ds_C_w  = ds_C.isel(time=m)
ds_Cf_w = ds_Cf.isel(time=m)
years = list(range(2001, 2017))
lat = ds_C_w["lat"].values; lon = ds_C_w["lon"].values

p_C  = json.load(open(REPO / "models" / "C"      / "params.lei-magaware-annual.json"))["params"]
p_Cf = json.load(open(REPO / "models" / "C-fuel" / "params.lei-magaware-annual.json"))["params"]

print("Predicting Model C ...")
pred_C_m, _ = predict_C(ds_C_w, p_C)
print("Predicting Model C-fuel ...")
d_in = load_Cfuel_inputs(ds_Cf_w)
pred_Cf_m = predict_Cfuel(d_in, p_Cf)
print("Loading GFED ...")
obs_m = load_gfed_halfdeg(years)

def annual_pct(x): return x.reshape(16, 12, 360, 720).sum(1) * 100.0
truth = annual_pct(obs_m).mean(0)
mc    = annual_pct(pred_C_m).mean(0)
mcf   = annual_pct(pred_Cf_m).mean(0)

gfed_active = (obs_m > 0).any(0)


def make_map(data, fname, title, cmap, vmin, vmax, label):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


vmax = 15
make_map(truth, "20_truth.png",
         "GFED4.1s, 16-yr mean annual burned area (%)", "YlOrRd", 0, vmax,
         "% per year")
make_map(mc, "21_modelC.png",
         "Model C (magaware-annual), 16-yr mean (%)", "YlOrRd", 0, vmax,
         "% per year")
make_map(mcf, "22_modelCfuel.png",
         "Model C-fuel (985-trial preliminary), 16-yr mean (%)", "YlOrRd", 0, vmax,
         "% per year")

bvmax = 10
make_map(np.where(gfed_active, mc - truth, np.nan),
         "23_modelC_bias.png",
         "Model C minus GFED (%)", "RdBu_r", -bvmax, bvmax,
         "pred - obs (% per year)")
make_map(np.where(gfed_active, mcf - truth, np.nan),
         "24_modelCfuel_bias.png",
         "Model C-fuel minus GFED (%)", "RdBu_r", -bvmax, bvmax,
         "pred - obs (% per year)")
make_map(np.where(gfed_active, mcf - mc, np.nan),
         "25_Cfuel_minus_C.png",
         "Model C-fuel minus Model C (%)", "RdBu_r", -5, 5,
         "Cfuel - C (% per year)")

# Region scoring
import sys
sys.path.insert(0, "scripts")
from predict_modelC_lei import metrics
cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
w2 = np.broadcast_to(cos_lat[:, None], (360, 720)).copy()
afr = ((lat[:, None] >= -35) & (lat[:, None] <= 38)
       & (lon[None, :] >= -20) & (lon[None, :] <= 52))
land = (ds_Cf_w["area_frac_ntrl"].isel(time=0).values
        + ds_Cf_w["area_frac_scnd"].isel(time=0).values
        + ds_Cf_w["area_frac_past"].isel(time=0).values) > 0

pred_C_a  = annual_pct(pred_C_m)
pred_Cf_a = annual_pct(pred_Cf_m)
obs_a     = annual_pct(obs_m)

print(f"\n{'region':24s} {'r_C':>7s} {'r_Cf':>7s} {'p_C%':>7s} {'p_Cf%':>7s} {'obs%':>7s}")
for name, mask in [("Global, land", land),
                    ("Africa, land", land & afr),
                    ("Non-Africa, land", land & ~afr)]:
    w = (w2 * mask)[None].repeat(16, 0)
    M_C  = metrics(pred_C_a,  obs_a, w)
    M_Cf = metrics(pred_Cf_a, obs_a, w)
    print(f"{name:24s} {M_C['r']:7.4f} {M_Cf['r']:7.4f} "
          f"{M_C['pred_mean']:7.3f} {M_Cf['pred_mean']:7.3f} {M_C['obs_mean']:7.3f}")
print("done")
