"""
Six diagnostic maps comparing Lei's params vs our magaware-annual params
against GFED4.1s, 2001-2016.

Outputs (PNG, 0.5 deg, robinson projection, NEW MAPS/):
  10_truth_annualmean_pct.png
  11_lei_pred_annualmean_pct.png
  12_ours_pred_annualmean_pct.png
  13_lei_bias.png
  14_ours_bias.png
  15_r_diff_ours_minus_lei.png
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
from predict_modelC_lei import predict_global, load_gfed_halfdeg

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS"
OUT.mkdir(exist_ok=True)

ds = xr.open_dataset(REPO / "global_baseline_modelC_inputs_1997-2016.nc")
yrs = np.array([d.year for d in ds["time"].values])
ds_w = ds.isel(time=(yrs >= 2001) & (yrs <= 2016))
years = list(range(2001, 2017))
lat = ds_w["lat"].values; lon = ds_w["lon"].values

p_lei  = json.load(open(REPO / "from_lei" / "lei_params_unwrapped.json"))["params"]
p_ours = json.load(open(REPO / "models" / "C" / "params.lei-magaware-annual.json"))["params"]

print("Predicting Lei ...")
pred_lei_m, _ = predict_global(ds_w, p_lei)
print("Predicting Ours ...")
pred_ours_m, _ = predict_global(ds_w, p_ours)
print("Loading GFED ...")
obs_m = load_gfed_halfdeg(years)

# Annual percent
def annual_pct(x): return x.reshape(16, 12, 360, 720).sum(1) * 100.0
truth = annual_pct(obs_m).mean(0)
lei   = annual_pct(pred_lei_m).mean(0)
ours  = annual_pct(pred_ours_m).mean(0)

# Per-cell time series Pearson r on monthly (192 months)
def per_cell_r(p, o):
    p = p.reshape(192, -1); o = o.reshape(192, -1)
    pm = p.mean(0); om = o.mean(0)
    pa = p - pm; oa = o - om
    num = (pa * oa).sum(0)
    den = np.sqrt((pa**2).sum(0) * (oa**2).sum(0)) + 1e-30
    return (num / den).reshape(360, 720)

r_lei  = per_cell_r(pred_lei_m, obs_m)
r_ours = per_cell_r(pred_ours_m, obs_m)

# Mask cells that never burn so r is meaningful
gfed_active = (obs_m > 0).any(0)
r_diff = np.where(gfed_active, r_ours - r_lei, np.nan)


def draw(ax, data, title, cmap, vmin, vmax, label):
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)


def make_map(data, fname, title, cmap, vmin, vmax, label):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Robinson())
    draw(ax, data, title, cmap, vmin, vmax, label)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


vmax = 15  # percent annual burned, generous
make_map(truth, "10_truth_annualmean_pct.png",
         "GFED4.1s, 16-yr mean annual burned area (%)", "YlOrRd", 0, vmax,
         "% per year")
make_map(lei,   "11_lei_pred_annualmean_pct.png",
         "Lei params, 16-yr mean annual burned area (%)", "YlOrRd", 0, vmax,
         "% per year")
make_map(ours,  "12_ours_pred_annualmean_pct.png",
         "Magaware-annual, 16-yr mean annual burned area (%)", "YlOrRd", 0, vmax,
         "% per year")

bvmax = 10
make_map(np.where(gfed_active, lei  - truth, np.nan),
         "13_lei_bias.png",
         "Lei minus GFED (16-yr annual mean, %)", "RdBu_r", -bvmax, bvmax,
         "pred - obs (% per year)")
make_map(np.where(gfed_active, ours - truth, np.nan),
         "14_ours_bias.png",
         "Magaware-annual minus GFED (16-yr annual mean, %)", "RdBu_r", -bvmax, bvmax,
         "pred - obs (% per year)")

make_map(r_diff, "15_r_diff_ours_minus_lei.png",
         "Per-cell time-series r difference (Ours - Lei)", "RdBu_r", -0.6, 0.6,
         "Pearson r difference")
print("done")
