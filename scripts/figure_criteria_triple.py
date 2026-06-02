"""
Paper figure: Model C under three optimization weightings on GFED5 burned area.
Row 1: GFED5 truth and the three weighting variants, large maps.
Row 2: bias maps for each variant, large.
Bottom: official ILAMB component score table.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeat

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "CRITERIA"
OUT.mkdir(parents=True, exist_ok=True)


def annual_pct_from_fraction(monthly_frac, n_years=16):
    arr = monthly_frac.reshape(n_years, 12, *monthly_frac.shape[1:]).sum(axis=1) * 100.0
    return arr.mean(axis=0)


def annual_pct_from_percent(monthly_pct, n_years=16):
    arr = monthly_pct.reshape(n_years, 12, *monthly_pct.shape[1:]).sum(axis=1)
    return arr.mean(axis=0)


def load_model_pct(name):
    p = REPO / "ilamb" / "MODELS_LEADERBOARD" / name / "burntArea.nc"
    da = xr.open_dataset(p)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    mask = (yrs >= 2001) & (yrs <= 2016)
    return annual_pct_from_fraction(da.values[mask])


gfed5 = xr.open_dataset(REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc")
yrs = np.array([t.year for t in gfed5.time.values])
mask = (yrs >= 2001) & (yrs <= 2016)
truth = annual_pct_from_percent(gfed5["burntArea"].values[mask])
lat = gfed5["lat"].values
lon = gfed5["lon"].values

crit_a = load_model_pct("ED-ModelC-GFED5")
crit_b = load_model_pct("ED-ModelC-GFED5cont")
crit_c = load_model_pct("ED-ModelC-GFED5type")


def map_panel(ax, data, title, vmax=15):
    ax.set_global()
    ax.coastlines(linewidth=0.5, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.25, edgecolor="0.5")
    im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                       cmap="YlOrRd", vmin=0, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=14, pad=4)
    return im


def bias_panel(ax, data, title, vlim=10):
    ax.set_global()
    ax.coastlines(linewidth=0.5, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.25, edgecolor="0.5")
    im = ax.pcolormesh(lon, lat, np.where(truth > 0, data, np.nan),
                       transform=ccrs.PlateCarree(),
                       cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto")
    ax.set_title(title, fontsize=14, pad=4)
    return im


# Pull official ILAMB component scores from the ba_criteria run
ilamb_csv = REPO / "ilamb_out_ba_criteria" / "scalar_database.csv"
ilamb_df = pd.read_csv(ilamb_csv)
def ilamb(model, name):
    sub = ilamb_df[(ilamb_df["Region"] == "global") & (ilamb_df["Model"] == model)]
    r = sub[sub["ScalarName"] == name]["Data"]
    return float(r.iloc[0]) if len(r) else float("nan")


def score_row(model_label, model_name):
    return [model_label,
            f"{ilamb(model_name, 'Overall Score'):.4f}",
            f"{ilamb(model_name, 'Bias Score'):.4f}",
            f"{ilamb(model_name, 'RMSE Score'):.4f}",
            f"{ilamb(model_name, 'Seasonal Cycle Score'):.4f}",
            f"{ilamb(model_name, 'Spatial Distribution Score'):.4f}"]


fig = plt.figure(figsize=(22, 13))
gs = gridspec.GridSpec(5, 4, figure=fig,
                       height_ratios=[1, 0.07, 1, 0.07, 0.30],
                       hspace=0.25, wspace=0.04)

# Row 1: predictions
ax_t = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
ax_a = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
ax_b = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())
ax_c = fig.add_subplot(gs[0, 3], projection=ccrs.Robinson())
im_v = map_panel(ax_t, truth, "GFED5, 2001-2016 mean")
map_panel(ax_a, crit_a, "(a) every fire equal")
map_panel(ax_b, crit_b, "(b) every continent equal")
map_panel(ax_c, crit_c, "(c) every fire type equal")

# Row 2 is the prediction colorbar band (gs[1, :])
cax_v = fig.add_subplot(gs[1, 1:3])
cb_v = fig.colorbar(im_v, cax=cax_v, orientation="horizontal")
cb_v.set_label("annual burned area, % per year", fontsize=12)
cb_v.ax.tick_params(labelsize=11)

# Row 3: bias maps
ax_ba = fig.add_subplot(gs[2, 0], projection=ccrs.Robinson())
ax_bb = fig.add_subplot(gs[2, 1], projection=ccrs.Robinson())
ax_bc = fig.add_subplot(gs[2, 2], projection=ccrs.Robinson())
ax_bd = fig.add_subplot(gs[2, 3], projection=ccrs.Robinson())
im_b = bias_panel(ax_ba, crit_a - truth, "(a) bias")
bias_panel(ax_bb, crit_b - truth, "(b) bias")
bias_panel(ax_bc, crit_c - truth, "(c) bias")
ax_bd.set_visible(False)

# Row 4 is the bias colorbar band (gs[3, :])
cax_b = fig.add_subplot(gs[3, 1:3])
cb_b = fig.colorbar(im_b, cax=cax_b, orientation="horizontal")
cb_b.set_label("predicted minus observed, % per year", fontsize=12)
cb_b.ax.tick_params(labelsize=11)

# Row 5: official ILAMB score table
ax_tab = fig.add_subplot(gs[4, :])
ax_tab.axis("off")
rows = [
    ["Criterion", "Overall", "Bias", "RMSE", "Seasonal", "Spatial"],
    score_row("(a) every fire equal",       "ED-ModelC-GFED5"),
    score_row("(b) every continent equal",  "ED-ModelC-GFED5cont"),
    score_row("(c) every fire type equal",  "ED-ModelC-GFED5type"),
]
tab = ax_tab.table(cellText=rows, loc="center", cellLoc="center",
                   colWidths=[0.22, 0.08, 0.08, 0.08, 0.08, 0.08])
tab.auto_set_font_size(False)
tab.set_fontsize(11)
tab.scale(1.0, 1.3)
for j in range(6):
    tab[(0, j)].set_text_props(weight="bold")
    tab[(0, j)].set_facecolor("#eaeaea")

fig.suptitle("Model C under three optimization weightings on GFED5 burned area",
             fontsize=17, y=0.96)
out_path = OUT / "FIG_criteria_triple.png"
fig.savefig(out_path, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_path}")
print(f"truth mean {np.nanmean(truth):.3f}, (a) {np.nanmean(crit_a):.3f}, "
      f"(b) {np.nanmean(crit_b):.3f}, (c) {np.nanmean(crit_c):.3f}")
