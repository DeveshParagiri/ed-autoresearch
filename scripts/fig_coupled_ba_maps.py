"""Burned-area maps for the coupled comparison Lei asked for.

    C:/Users/owusu/miniforge3/python.exe scripts/fig_coupled_ba_maps.py

Writes paper_gmd/figures/coupled_ba_maps.{png,pdf}.

Six panels on one common scale, every field on the same 2001-2016 window so nothing in
the picture is explained by a difference in period.

    GFED4.1s observed      GFED5 observed        EDv3 as submitted to TRENDY
    coupled default        coupled Model F       Model F minus default

The two observations sit together at the top because they disagree with each other, and
seeing by how much is the honest frame for judging any model against either. The last
panel is where our fire scheme actually changes the answer, which no side-by-side pair
of maps makes visible on its own.

One scale for all five burned-area panels, deliberately. EDv3 burns 2500 Mha/yr and the
coupled runs burn under 180 against 793 observed, so a per-panel scale would hide the
single largest fact in the comparison. Panels saturate, and the Mha in each title carries
the magnitude where the colour runs out.

Run with the BASE env. matplotlib cannot render in edfire on this machine.
"""
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper_gmd" / "figures"
CM = REPO / "ilamb" / "MODELS_COUPLED_COMMON"
REF = REPO / "ilamb_ref_official" / "DATA" / "burntArea"

# sequential, single ramp light to dark, the house fire scale used by the paper figures
FIRE = LinearSegmentedColormap.from_list(
    "fire", ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"])

R = 6371e3
LAT = np.arange(-89.75, 90, 0.5)
AREA = (R ** 2 * np.deg2rad(0.5) ** 2 * np.cos(np.deg2rad(LAT)))[:, None] * np.ones((1, 720))

PANELS = [("GFED4.1s observed",      REF / "GFED4.1S" / "burntArea.nc"),
          ("GFED5 observed",         REF / "GFED5" / "burntArea.nc"),
          ("EDv3, TRENDY submission", CM / "EDv3-TRENDY" / "burntArea.nc"),
          ("Coupled ED, default fire", CM / "ED-coupled-default" / "burntArea.nc"),
          ("Coupled ED, Model F",     CM / "ED-coupled-ModelF" / "burntArea.nc")]


def annual_pct(path):
    """mean annual burned area, percent of cell per year, on 2001-2016, plus the Mha total"""
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    sel = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[sel].astype(np.float64), nan=0.0)
    if da.attrs.get("units", "1") in ("%", "percent"):
        arr = arr / 100.0                       # references are percent, some models fraction
    ann = arr.reshape(int(sel.sum()) // 12, 12, *arr.shape[1:]).sum(1).mean(0)
    return ann * 100.0, float((ann * AREA).sum() / 1e10)


fields = [(t, *annual_pct(p)) for t, p in PANELS]
gfed5 = fields[1][1]
vmax = float(np.percentile(gfed5[gfed5 > 0], 98))   # set by the observation, not by a model

fig, axes = plt.subplots(2, 3, figsize=(11.0, 4.6),
                         subplot_kw={"projection": ccrs.Robinson()})
axes = axes.ravel()


def base(ax):
    ax.set_global()
    ax.coastlines(linewidth=0.28, color="#4a4a4a")
    ax.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.28)
    ax.spines["geo"].set_visible(False)


for ax, (title, data, mha), lab in zip(axes, fields, "abcde"):
    base(ax)
    m = ax.pcolormesh(np.arange(-179.75, 180, 0.5), LAT,
                      np.where(data < 1e-9, np.nan, data), transform=ccrs.PlateCarree(),
                      cmap=FIRE, vmin=0, vmax=vmax, shading="auto")
    ax.set_title(f"({lab}) {title}\n{mha:.0f} Mha yr$^{{-1}}$", fontsize=8.5, pad=3)

# what our fire scheme actually changed, on its own diverging scale
diff = fields[4][1] - fields[3][1]
ax = axes[5]
base(ax)
lim = float(np.percentile(np.abs(diff[np.abs(diff) > 0]), 99))
md = ax.pcolormesh(np.arange(-179.75, 180, 0.5), LAT,
                   np.where(np.abs(diff) < 1e-9, np.nan, diff), transform=ccrs.PlateCarree(),
                   cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
ax.set_title(f"(f) Model F minus default\n{fields[4][2] - fields[3][2]:+.0f} Mha yr$^{{-1}}$",
             fontsize=8.5, pad=3)

# the two bars measure different things, so they are kept apart rather than stacked in one
# margin, where a reader takes the second for a continuation of the first
fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.10, wspace=0.02, hspace=0.28)

cb = fig.colorbar(m, ax=axes[:5], orientation="horizontal",
                  fraction=0.030, pad=0.03, aspect=48, anchor=(0.32, 1.0), shrink=0.55)
cb.set_label("burned area (% of cell per year)", fontsize=8, color="#5a5f66")
cb.ax.tick_params(labelsize=7.5, length=0, colors="#5a5f66")
cb.outline.set_visible(False)

cd = fig.colorbar(md, ax=axes[5], orientation="horizontal",
                  fraction=0.030, pad=0.03, aspect=20, shrink=0.85)
cd.set_label("Model F minus default (% per year)", fontsize=7.5, color="#5a5f66")
cd.ax.tick_params(labelsize=7, length=0, colors="#5a5f66")
cd.outline.set_visible(False)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"coupled_ba_maps.{ext}", dpi=300, bbox_inches="tight")
print("wrote coupled_ba_maps.png / .pdf")
for t, _, mha in fields:
    print(f"  {t:26s} {mha:6.0f} Mha/yr")
