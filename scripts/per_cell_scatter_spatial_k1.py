"""
Per-cell 1:1 scatter showing the PROGRESSION on George's bar: canonical k4
(legacy) -> seasonal-k1 (transform, sigma up) -> spatial-k1 (A+B refit: fire_amp
breaks the rate cap, spatial Taylor objective drives sigma->1). Each panel is
model vs GFED5 burned fraction with the 1:1 line; the title reports the per-cell
slope = r*sigma (George's bar is slope -> 1).

The story this figure tells: the band climbs from slope 0.34 -> 0.50 and the
ceiling from 0.039 -> 0.095 (now reaching GFED5's 0.104), but it PLATEAUS at
slope ~ r ~ 0.5 because the spatial correlation r is stuck ~0.5 - reaching the
diagonal needs better WHERE (continent-specific structure), not more amplitude.

Output: NEW MAPS/proto_seasonal/per_cell_scatter_spatial_k1.png
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
OUT = REPO / "NEW MAPS" / "proto_seasonal"
OUT.mkdir(parents=True, exist_ok=True)

PANELS = [
    ("canonical tropfix2-k4 (legacy)",
     REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc"),
    ("seasonal-k1 (transform)",
     REPO / "ilamb/MODELS_TOPK_seasonal/ED-ModelC-seasonal-k1/burntArea.nc"),
    ("spatial-k1 (A+B: fire_amp + spatial obj)",
     REPO / "ilamb/MODELS_TOPK_spatial/ED-ModelC-spatial-k1/burntArea.nc"),
]


def period_mean(path):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    sel = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[sel].astype(np.float64), nan=0.0)
    if da.attrs.get("units", "1") in ("%", "percent"):
        arr = arr / 100.0
    return arr.mean(axis=0)


gfed = period_mean(REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
maps = [(lab, period_mean(p)) for lab, p in PANELS]
# active-fire cells (GFED5 annual > 1%) — where the 1:1 dynamic range matters
mask = np.isfinite(gfed) & ((gfed * 12.0) > 0.01)
g = gfed[mask]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
fig.suptitle("Per-cell burned area vs GFED5 (active-fire cells): the band climbing the 1:1 line",
             fontsize=14, y=1.02)
lim = 0.12
for ax, (lab, mp) in zip(axes, maps):
    m = mp[mask]
    r = np.corrcoef(m, g)[0, 1]
    sigma = m.std() / g.std()
    slope = r * sigma
    hb = ax.hexbin(g, m, gridsize=80, extent=(0, lim, 0, lim),
                   norm=LogNorm(vmin=1, vmax=1e4), cmap="viridis", mincnt=1)
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="1:1 (target)")
    # fitted slope line through origin
    ax.plot([0, lim], [0, lim * slope], "w-", lw=1.4, alpha=0.8,
            label=f"fit slope {slope:.2f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("GFED5 burned fraction"); ax.set_ylabel("Model burned fraction")
    ax.set_title(f"{lab}\nr {r:.2f}   sigma {sigma:.2f}   slope {slope:.2f}   "
                 f"max {m.max():.3f}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
fig.colorbar(hb, ax=axes[2], fraction=0.046, pad=0.02).set_label("cell count (log)")
fig.tight_layout()
f = OUT / "per_cell_scatter_spatial_k1.png"
fig.savefig(f, dpi=130, bbox_inches="tight")
print("wrote", f)
print(f"GFED5 active-fire cells n={int(mask.sum())}, max {g.max():.4f}, std {g.std():.5f}")
