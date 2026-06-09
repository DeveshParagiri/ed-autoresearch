"""
Per-grid-cell burned-area scatter (model vs GFED5), period-mean monthly burned
fraction, for the PROMOTED canonical model (tropfix2-k4) compared against the
previous (PRE-tropfix2) canonical. Mirrors the 2026-06-03 scalar-session figure
(NEW MAPS/Seasonal/2_diff_scatter_scaled.png) but answers "what did the principled
tropical suppression do to the per-cell distribution / the dynamic-range ceiling".

Output: ED-ModelC-tropfix2-k4/figures/per_cell_scatter_k4.png
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
OUT  = REPO / "ED-ModelC-tropfix2-k4" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def period_mean_fraction(path):
    """Mean over all 2001-2016 monthly steps of burned fraction (units '1')."""
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    m = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[m].astype(np.float64), nan=0.0)
    units = da.attrs.get("units", "1")
    if units in ("%", "percent"):
        arr = arr / 100.0          # -> fraction
    return arr.mean(axis=0)        # (lat, lon) period-mean monthly fraction


k4    = period_mean_fraction(REPO / "ED-ModelC-tropfix2-k4" / "model" / "burntArea.nc")
preTF = period_mean_fraction(REPO / "backups_PRE-tropfix2" / "burntArea.canonical.nc")
gfed  = period_mean_fraction(REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc")

# burnable land: any of the three burns somewhere
land = (gfed > 0) | (k4 > 0) | (preTF > 0)
g  = gfed[land]
a4 = k4[land]
ap = preTF[land]


def stats(model, ref):
    d = model - ref
    return d.mean(), np.sqrt((d**2).mean()), np.abs(d).mean()


def panel(ax, model, ref, title, lim=0.12):
    hb = ax.hexbin(ref, model, gridsize=80, extent=(0, lim, 0, lim),
                   norm=LogNorm(vmin=1, vmax=1e4), cmap="viridis", mincnt=1)
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="1:1")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("GFED5 burned fraction")
    ax.set_ylabel("Model burned fraction")
    md, rmse, _ = stats(model, ref)
    ax.set_title(f"{title}\nmean diff {md:+.4f}   RMSE {rmse:.4f}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    return hb


fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
fig.suptitle("Per-grid-cell burned area: PROMOTED model (tropfix2-k4) vs GFED5",
             fontsize=14, y=1.02)

hb1 = panel(axes[0], ap, g, "PRE-tropfix2 canonical (1.26x)")
hb2 = panel(axes[1], a4, g, "tropfix2-k4  PROMOTED (1.11x)")
cb = fig.colorbar(hb2, ax=axes[1], fraction=0.046, pad=0.02)
cb.set_label("cell count (log)")

# Panel 3: per-cell difference distribution
ax = axes[2]
dp = ap - g; d4 = a4 - g
bins = np.linspace(-0.075, 0.075, 80)
ax.hist(dp, bins=bins, color="indianred", alpha=0.55,
        label=f"PRE-tropfix2  |d|mean {np.abs(dp).mean():.4f}")
ax.hist(d4, bins=bins, color="steelblue", alpha=0.55,
        label=f"tropfix2-k4   |d|mean {np.abs(d4).mean():.4f}")
ax.axvline(0, color="k", lw=1)
ax.set_yscale("log")
ax.set_xlabel("Model - GFED5 (burned fraction)")
ax.set_ylabel("cell count (log)")
ax.set_title("Distribution of per-cell difference", fontsize=11)
ax.legend(loc="upper right", fontsize=9)

fig.tight_layout()
f = OUT / "per_cell_scatter_k4.png"
fig.savefig(f, dpi=130, bbox_inches="tight")
print("wrote", f)

# Console summary
print("\nPer-cell summary (burnable cells, n=%d):" % land.sum())
for nm, mod in [("PRE-tropfix2", ap), ("tropfix2-k4 ", a4)]:
    md, rmse, am = stats(mod, g)
    print(f"  {nm}: mean diff {md:+.5f}  RMSE {rmse:.5f}  |d|mean {am:.5f}  "
          f"model max {mod.max():.4f}")
print(f"  GFED5 max {g.max():.4f}  GFED5 mean {g.mean():.5f}")
