"""Quick 4-panel BA comparison for the coupling-ready refit.

GFED5 reference | paper Model E (per-continent, look for seams) |
coupling-ready k2 (0.99x, smooth) | coupling-ready k1 (1.26x, top ILAMB).

Period-mean monthly burned fraction, 2001-2016. Output PNG path from argv[1].
"""
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

OUT = sys.argv[1] if len(sys.argv) > 1 else "coupled_ba_maps.png"
N = 192  # 2001-2016

PANELS = [
    ("GFED5 (reference)",
     "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc", 1 / 100.0),
    ("Paper Model E  (per-continent, 0.6646)",
     "ilamb/MODELS_LEADERBOARD/ED-ModelC-E-clean/burntArea.nc", 1.0),
    ("Coupling-ready k2  (single global, 0.99x, 0.6532)",
     "ilamb/MODELS_TOPK_coupledE/ED-ModelC-coupledE-k2/burntArea.nc", 1.0),
    ("Coupling-ready k1  (single global, 1.26x, 0.6710)",
     "ilamb/MODELS_TOPK_coupledE/ED-ModelC-coupledE-k1/burntArea.nc", 1.0),
]


def period_mean(nc, scale):
    ds = xr.open_dataset(nc)
    a = ds["burntArea"]
    v = np.nan_to_num(a.isel(time=slice(0, N)).values.astype(np.float64)) * scale
    m = v.mean(axis=0)  # mean monthly fraction
    lat = a.lat.values; lon = a.lon.values
    ds.close()
    return lat, lon, m


fields = [(t, *period_mean(f, s)) for t, f, s in PANELS]
# shared color scale anchored on GFED5's 99th pct of nonzero cells
g = fields[0][3]
vmax = float(np.percentile(g[g > 0], 99))
cmap = plt.get_cmap("YlOrRd").copy()
cmap.set_under("white")

fig, axes = plt.subplots(2, 2, figsize=(15, 8),
                         subplot_kw={"projection": ccrs.Robinson()})
for ax, (title, lat, lon, m) in zip(axes.ravel(), fields):
    mm = np.ma.masked_less_equal(m, 1e-6)
    pcm = ax.pcolormesh(lon, lat, mm, transform=ccrs.PlateCarree(),
                        cmap=cmap, vmin=1e-6, vmax=vmax, shading="auto")
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="#444444")
    ax.set_global(); ax.set_ylim
    ax.set_title(title, fontsize=11, weight="bold")
cbar = fig.colorbar(pcm, ax=axes, orientation="horizontal",
                    fraction=0.045, pad=0.03, aspect=45)
cbar.set_label("mean monthly burned fraction (2001-2016)", fontsize=10)
fig.suptitle("Coupling-ready refit vs GFED5 and the paper Model E",
             fontsize=13, weight="bold", y=0.99)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}  (vmax={vmax:.4f})")
