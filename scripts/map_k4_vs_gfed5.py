"""
Side-by-side map: GFED5 vs tropfix2-k4 burned area (with canonical + difference).
Annual-mean burned area, % per year, 2001-2016. GFED5 is already in %; our model
output is fraction, so x100.
"""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "NEW MAPS" / "tropfix2"
OUT.mkdir(parents=True, exist_ok=True)


def annual_pct(path, is_truth):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    m = (yrs >= 2001) & (yrs <= 2016)
    arr = da.values[m]
    n = arr.shape[0] // 12
    annual = arr.reshape(n, 12, *arr.shape[1:]).sum(axis=1).mean(axis=0)
    if not is_truth:          # our output is fraction -> %
        annual = annual * 100.0
    return da.lat.values, da.lon.values, annual


GF = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"
CN = REPO / "ilamb" / "MODELS_TOPK_tropfix2" / "ED-ModelC-canonical" / "burntArea.nc"
K4 = REPO / "ilamb" / "MODELS_TOPK_tropfix2" / "ED-ModelC-tropfix2-k4" / "burntArea.nc"

lat, lon, gfed = annual_pct(GF, True)
_,   _,   cano = annual_pct(CN, False)
_,   _,   k4   = annual_pct(K4, False)

# align k4/canonical onto GFED5 grid for the difference panel
def to_grid(vals, vlat, vlon):
    da = xr.DataArray(vals, coords={"lat": vlat, "lon": vlon}, dims=("lat", "lon"))
    return da.interp(lat=lat, lon=lon, method="nearest").values

clat, clon, _ = annual_pct(K4, False)
k4_g = to_grid(k4, clat, clon)
diff = np.where(gfed > 0, k4_g - gfed, np.nan)

print(f"GFED5     mean={np.nanmean(gfed):.2f}% max={np.nanmax(gfed):.0f}")
print(f"canonical mean={np.nanmean(cano):.2f}% max={np.nanmax(cano):.0f}")
print(f"k4        mean={np.nanmean(k4):.2f}% max={np.nanmax(k4):.0f}")


def panel(ax, plat, plon, data, title, vmin, vmax, cmap, clabel):
    ax.set_global(); ax.coastlines(linewidth=0.4, color="0.3")
    ax.add_feature(cfeat.BORDERS, linewidth=0.2, edgecolor="0.5")
    im = ax.pcolormesh(plon, plat, data, transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.7)
    cb.set_label(clabel, fontsize=8); cb.ax.tick_params(labelsize=7)


fig, ax = plt.subplots(2, 2, figsize=(16, 9), subplot_kw={"projection": ccrs.Robinson()})
fig.suptitle("Burned Area 2001-2016 mean annual (% per year): GFED5 vs tropfix2-k4", fontsize=13)
panel(ax[0, 0], lat, lon, gfed, "GFED5 truth (793 Mha/yr, ILAMB=0.6485 ref)", 0, 15, "YlOrRd", "% per year")
panel(ax[0, 1], clat, clon, k4, "tropfix2-k4 (881 Mha/yr=1.11x, official ILAMB=0.6474)", 0, 15, "YlOrRd", "% per year")
panel(ax[1, 0], clat, clon, cano, "canonical (1001 Mha/yr=1.26x, official ILAMB=0.6485)", 0, 15, "YlOrRd", "% per year")
panel(ax[1, 1], lat, lon, diff, "k4 minus GFED5 (where GFED5>0)", -10, 10, "RdBu_r", "pred minus obs (% per year)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fp = OUT / "BA_k4_vs_gfed5.png"
fig.savefig(fp, dpi=150, bbox_inches="tight")
print(f"wrote {fp}")
