"""
Side-by-side global map: GFED5 vs the continental Model C vs their difference,
annual burned-area fraction (period mean 2001-2016). Answers "how close is the
map to GFED5" directly. Magnitude 1.03x, spatial r 0.70, 1:1 slope 0.65.

Output: NEW MAPS/proto_seasonal/map_continental_vs_gfed.png
"""
from pathlib import Path
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "proto_seasonal"; OUT.mkdir(parents=True, exist_ok=True)


def annual_frac(path):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values]); sel = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[sel].astype(np.float64), nan=0.0)
    if da.attrs.get("units", "1") in ("%", "percent"): arr = arr / 100.0
    ann = arr.reshape(-1, 12, *arr.shape[1:]).sum(axis=1).mean(axis=0)   # annual fraction
    return da.lat.values, da.lon.values, ann * 100.0                      # -> percent/yr


lat, lon, gfed = annual_frac(REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
_, _, cont = annual_frac(REPO / "ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc")
diff = cont - gfed

proj = ccrs.Robinson()
fig, axes = plt.subplots(1, 3, figsize=(19, 4.6), subplot_kw={"projection": proj})
fig.suptitle("Annual burned-area fraction (% / yr), 2001-2016:  "
             "Model C continental vs GFED5   (magnitude 1.03x, spatial r 0.70)",
             fontsize=13, y=1.04)

vmax = 30.0
fire_cmap = LinearSegmentedColormap.from_list(
    "fire", ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"])


def draw(ax, data, title, cmap, vmin, vmax, label):
    ax.set_global(); ax.coastlines(linewidth=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.3)
    m = ax.pcolormesh(lon, lat, np.where(data == 0, np.nan, data),
                      transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax,
                      shading="auto")
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85, extend="both")
    cb.set_label(label, fontsize=9); cb.ax.tick_params(labelsize=8)


draw(axes[0], gfed, "GFED5 (observed)", fire_cmap, 0, vmax, "% burned / yr")
draw(axes[1], cont, "Model C continental", fire_cmap, 0, vmax, "% burned / yr")
draw(axes[2], diff, "Model C - GFED5 (difference)", "RdBu_r", -15, 15, "% / yr  (red=too much, blue=too little)")

fig.tight_layout()
f = OUT / "map_continental_vs_gfed.png"
fig.savefig(f, dpi=140, bbox_inches="tight"); print("wrote", f)
