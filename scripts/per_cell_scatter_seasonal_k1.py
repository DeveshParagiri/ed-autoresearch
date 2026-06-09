"""
Per-grid-cell burned-area 1:1 scatter (model vs GFED5), period-mean monthly
burned fraction, for the SEASONAL-TRANSFORM refit winner (seasonal-k1, trial
2438) compared against the current canonical (tropfix2-k4). This is George's
1:1-plot bar: does the band climb the diagonal (dynamic range / spatial sigma)
WITHOUT re-inflating magnitude.

Official ILAMB (same single-model run): canonical k4 Spatial 0.7617 Overall
0.6473; seasonal-k1 Spatial 0.7797 Overall 0.6495; both at magnitude 1.11x GFED5.

Output: NEW MAPS/proto_seasonal/per_cell_scatter_seasonal_k1.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "NEW MAPS" / "proto_seasonal"
OUT.mkdir(parents=True, exist_ok=True)


def period_mean_fraction(path):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    m = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[m].astype(np.float64), nan=0.0)
    if da.attrs.get("units", "1") in ("%", "percent"):
        arr = arr / 100.0
    return arr.mean(axis=0)


k4   = period_mean_fraction(REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc")
k1   = period_mean_fraction(REPO / "ilamb/MODELS_TOPK_seasonal/ED-ModelC-seasonal-k1/burntArea.nc")
gfed = period_mean_fraction(REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")

land = (gfed > 0) | (k4 > 0) | (k1 > 0)
g, a4, a1 = gfed[land], k4[land], k1[land]


def stats(model, ref):
    d = model - ref
    return d.mean(), np.sqrt((d**2).mean()), np.abs(d).mean()


def panel(ax, model, ref, title, lim=0.12):
    hb = ax.hexbin(ref, model, gridsize=80, extent=(0, lim, 0, lim),
                   norm=LogNorm(vmin=1, vmax=1e4), cmap="viridis", mincnt=1)
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="1:1")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("GFED5 burned fraction"); ax.set_ylabel("Model burned fraction")
    md, rmse, _ = stats(model, ref)
    ax.set_title(f"{title}\nmean diff {md:+.4f}   RMSE {rmse:.4f}   "
                 f"sigma_ratio {model.std()/ref.std():.3f}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    return hb


fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
fig.suptitle("Per-grid-cell burned area: seasonal-transform refit (k1) vs canonical vs GFED5",
             fontsize=14, y=1.02)
panel(axes[0], a4, g, "canonical tropfix2-k4 (legacy /12)\nofficial Spatial 0.7617  Overall 0.6473")
hb = panel(axes[1], a1, g, "seasonal-k1  refit (1-exp(-rate/12))\nofficial Spatial 0.7797  Overall 0.6495")
fig.colorbar(hb, ax=axes[1], fraction=0.046, pad=0.02).set_label("cell count (log)")

ax = axes[2]
bins = np.linspace(-0.075, 0.075, 80)
ax.hist(a4 - g, bins=bins, color="indianred", alpha=0.55,
        label=f"canonical k4   |d| {np.abs(a4-g).mean():.4f}")
ax.hist(a1 - g, bins=bins, color="steelblue", alpha=0.55,
        label=f"seasonal-k1    |d| {np.abs(a1-g).mean():.4f}")
ax.axvline(0, color="k", lw=1); ax.set_yscale("log")
ax.set_xlabel("Model - GFED5 (burned fraction)"); ax.set_ylabel("cell count (log)")
ax.set_title("Distribution of per-cell difference", fontsize=11)
ax.legend(loc="upper right", fontsize=9)

fig.tight_layout()
f = OUT / "per_cell_scatter_seasonal_k1.png"
fig.savefig(f, dpi=130, bbox_inches="tight")
print("wrote", f)

print("\nPer-cell summary (burnable cells, n=%d):" % land.sum())
for nm, mod in [("canonical k4", a4), ("seasonal-k1 ", a1)]:
    md, rmse, am = stats(mod, g)
    print(f"  {nm}: mean diff {md:+.5f}  RMSE {rmse:.5f}  |d|mean {am:.5f}  "
          f"max {mod.max():.4f}  sigma_ratio {mod.std()/g.std():.3f}")
print(f"  GFED5 max {g.max():.4f}  GFED5 std {g.std():.5f}  mean {g.mean():.5f}")
