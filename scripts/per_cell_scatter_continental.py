"""
The money figure for George's 1:1 bar: the per-cell band climbing the diagonal
across the whole arc - canonical (slope 0.34) -> spatial-k1, global amplitude
(0.50) -> continental, per-continent + fuel-Africa form (0.65). Each panel is
model vs GFED5 burned fraction on active-fire cells with the 1:1 line and the
fitted slope (= r*sigma).

Output: NEW MAPS/proto_seasonal/per_cell_scatter_continental.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "proto_seasonal"; OUT.mkdir(parents=True, exist_ok=True)
PANELS = [
    ("canonical tropfix2-k4 (global, legacy)", REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc"),
    ("spatial-k1 (global amplitude fix)",
     REPO / "ilamb/MODELS_TOPK_spatial/ED-ModelC-spatial-k1/burntArea.nc"),
    ("continental (per-continent + fuel-Africa)",
     REPO / "ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc"),
]


def period_mean(path):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values]); sel = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[sel].astype(np.float64), nan=0.0)
    if da.attrs.get("units", "1") in ("%", "percent"): arr = arr / 100.0
    return arr.mean(axis=0)


gfed = period_mean(REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
mask = np.isfinite(gfed) & ((gfed * 12.0) > 0.01)
g = gfed[mask]
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
fig.suptitle("Per-cell burned area vs GFED5 (active-fire cells): the band climbing the 1:1 line",
             fontsize=14, y=1.02)
lim = 0.12
for ax, (lab, p) in zip(axes, PANELS):
    m = period_mean(p)[mask]
    r = np.corrcoef(m, g)[0, 1]; sigma = m.std() / g.std(); slope = r * sigma
    hb = ax.hexbin(g, m, gridsize=80, extent=(0, lim, 0, lim),
                   norm=LogNorm(vmin=1, vmax=1e4), cmap="viridis", mincnt=1)
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="1:1 (target)")
    ax.plot([0, lim], [0, lim * slope], "w-", lw=1.4, alpha=0.85, label=f"fit slope {slope:.2f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("GFED5 burned fraction"); ax.set_ylabel("Model burned fraction")
    ax.set_title(f"{lab}\nr {r:.2f}   sigma {sigma:.2f}   slope {slope:.2f}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
fig.colorbar(hb, ax=axes[2], fraction=0.046, pad=0.02).set_label("cell count (log)")
fig.tight_layout()
f = OUT / "per_cell_scatter_continental.png"
fig.savefig(f, dpi=130, bbox_inches="tight"); print("wrote", f)
