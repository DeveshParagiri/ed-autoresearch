"""Burned-area map of the new regional-GDP model (ILAMB 0.6783) vs GFED5.
Annual mean burned fraction (% / yr): GFED5 | model | difference.
"""
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import cartopy.crs as ccrs, cartopy.feature as cfeature

def ann_pct(path, var="burntArea", pct_in=False):
    d = xr.open_dataset(path); a = d[var].values.astype(float); d.close()
    a = np.nan_to_num(a)
    if pct_in: a = a / 100.0                       # GFED5 stored in %
    n = a.shape[0] // 12
    return a[:n*12].reshape(n, 12, a.shape[-2], a.shape[-1]).sum(1).mean(0) * 100.0  # %/yr

gfed = ann_pct("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc", pct_in=True)[:, :]
mod  = ann_pct("ilamb/MODELS_GDP_REGIONAL/ED-ModelC-gdpreg/burntArea.nc")
d = xr.open_dataset("ilamb/MODELS_GDP_REGIONAL/ED-ModelC-gdpreg/burntArea.nc")
lat, lon = d.lat.values, d.lon.values; d.close()
gfed = gfed[:mod.shape[0]]                          # match 2001-2016 vs GFED 2001-2020
land = mod > 0
diff = np.where(land | (gfed > 0), mod - gfed, np.nan)
gfed_m = np.where(gfed > 0, gfed, np.nan); mod_m = np.where(mod > 0, mod, np.nan)

PC = ccrs.PlateCarree(); ext = [-180, 180, -58, 84]
kw = dict(transform=PC, extent=[-180, 180, -90, 90], origin="lower")
fig = plt.figure(figsize=(9.2, 11))
def ax(i):
    a = fig.add_subplot(3, 1, i, projection=PC); a.set_extent(ext, PC)
    a.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="#555"); return a

norm = LogNorm(vmin=0.1, vmax=60)
a1 = ax(1); im1 = a1.imshow(gfed_m, cmap="inferno_r", norm=norm, **kw)
a1.set_title("GFED5 observed burned area", fontsize=12, loc="left")
a2 = ax(2); im2 = a2.imshow(mod_m, cmap="inferno_r", norm=norm, **kw)
a2.set_title("New model (regional-GDP, ILAMB 0.6783)", fontsize=12, loc="left")
cb = fig.colorbar(im2, ax=[a1, a2], orientation="vertical", shrink=0.7, pad=0.01, aspect=30)
cb.set_label("burned fraction (% / yr)")
cb.set_ticks([0.1, 1, 5, 20, 60]); cb.set_ticklabels(["0.1", "1", "5", "20", "60"])

v = 20
a3 = ax(3); im3 = a3.imshow(diff, cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), **kw)
a3.set_title("Model - GFED5   (red = model burns more, blue = less)", fontsize=12, loc="left")
cb3 = fig.colorbar(im3, ax=a3, orientation="vertical", shrink=0.85, pad=0.01, aspect=18)
cb3.set_label("difference (% / yr)")

fig.suptitle("Burned area: new regional-GDP model vs GFED5", fontsize=13, y=0.995)
fig.savefig("ba_map_gdpreg.png", dpi=150, bbox_inches="tight")
print("[fig] ba_map_gdpreg.png")
# quick regional sanity in the caption
print(f"global model {np.nansum(mod_m):.0f} vs GFED {np.nansum(gfed_m):.0f} (sum of %/yr, unweighted)")
