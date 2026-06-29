"""
Taylor diagram for Model C burned area vs GFED5, on active-fire cells (GFED5>0).
Each model is one point: angle = arccos(correlation r), radius = std ratio sigma.
GFED5 is the reference at (r=1, sigma=1). Shows the arc from the old global model
(low r, low sigma) -> gain term (sigma fixed) -> continental (r lifted).

Output: NEW MAPS/continental_model/taylor_diagram.png
"""
from pathlib import Path
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "NEW MAPS" / "continental_model"; OUT.mkdir(parents=True, exist_ok=True)

REF = REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc"
MODELS = [
    ("Old global model",          REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc",                       "#d62728"),
    ("+ gain term (global)",      REPO / "ilamb/MODELS_TOPK_spatial/ED-ModelC-spatial-k1/burntArea.nc",     "#ff7f0e"),
    ("+ per-continent (now)",     REPO / "ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc",     "#2ca02c"),
]


def period_mean(path):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values]); sel = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[sel].astype(np.float64), nan=0.0)
    if da.attrs.get("units", "1") in ("%", "percent"): arr = arr / 100.0
    return arr.mean(axis=0)


gfed = period_mean(REF)
mask = np.isfinite(gfed) & ((gfed * 12.0) > 0.01)
g = gfed[mask]; gstd = g.std()

stats = []
print(f"{'model':24s} {'r':>6s} {'sigma':>6s} {'cRMSD_norm':>11s} {'RMSE_frac':>10s}")
for name, p, c in MODELS:
    m = period_mean(p)[mask]
    r = float(np.corrcoef(m, g)[0, 1])
    sigma = float(m.std() / gstd)
    crmsd = float(np.sqrt(sigma**2 + 1.0 - 2.0 * sigma * r))   # normalized centered RMS diff
    rmse = float(np.sqrt(((m - g) ** 2).mean()))               # raw RMSE in burned-fraction units
    stats.append((name, r, sigma, crmsd, rmse, c))
    print(f"{name:24s} {r:6.3f} {sigma:6.3f} {crmsd:11.3f} {rmse:10.5f}")

# --- Taylor diagram (first quadrant polar) ---
smax = 1.6
fig = plt.figure(figsize=(8, 7.2))
ax = fig.add_subplot(111, projection="polar")
ax.set_thetamin(0); ax.set_thetamax(90)
ax.set_rlim(0, smax)

# correlation grid (angular)
corr_ticks = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
ax.set_thetagrids(np.degrees(np.arccos(corr_ticks)), labels=[f"{c:g}" for c in corr_ticks])
ax.set_xlabel("")
ax.text(np.radians(45), smax * 1.18, "Correlation", ha="center", va="center",
        fontsize=11, rotation=-45)

# sigma (radial) grid + the reference unit arc
ax.set_rgrids([0.5, 1.0, 1.5], labels=["0.5", "1.0", "1.5"], angle=90, fontsize=9)
th = np.linspace(0, np.pi / 2, 100)
ax.plot(th, np.ones_like(th), color="k", ls="--", lw=1, alpha=0.6)
ax.set_ylabel("Standard deviation ratio (model / GFED5)", labelpad=30, fontsize=11)

# Centered-RMS-difference arcs centered on the reference (r=1, sigma=1).
# Distance from any model point to the reference = its (normalized) centered RMS diff.
ref_x = 1.0
for rms in [0.25, 0.5, 0.75, 1.0, 1.25]:
    ang = np.linspace(0, np.pi, 200)
    xs = ref_x + rms * np.cos(ang)
    ys = rms * np.sin(ang)
    rr = np.sqrt(xs**2 + ys**2); tt = np.arctan2(ys, xs)
    keep = (rr <= smax) & (tt >= 0) & (tt <= np.pi / 2)
    ax.plot(tt[keep], rr[keep], color="#3a7d3a", ls=":", lw=0.9, alpha=0.7)
    # label each arc where it crosses sigma = 1 (straight left of the reference)
    lx = ref_x - rms
    if lx > 0:
        ax.text(np.arctan2(0.0, lx), abs(lx), f"{rms:g}", color="#2d662d",
                fontsize=8, ha="center", va="bottom", alpha=0.9)

# reference point (GFED5)
ax.plot(0, 1.0, "k*", ms=18, label="GFED5 (reference)")

# model points
for name, r, sigma, crmsd, rmse, c in stats:
    ax.plot(np.arccos(r), sigma, "o", ms=12, color=c,
            label=f"{name}  (r={r:.2f}, sd={sigma:.2f}, cRMS={crmsd:.2f})")

# note explaining the green dotted arcs
ax.text(np.radians(2), smax * 0.30, "green dotted arcs =\ncentered RMS diff\n(norm. to GFED5 sd)",
        color="#2d662d", fontsize=8, ha="left", va="center")

ax.set_title("Taylor diagram: Model C burned area vs GFED5\n(active-fire cells, 2001-2016 mean)",
             fontsize=12, pad=24)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=9, frameon=True)
fig.tight_layout()
f = OUT / "taylor_diagram.png"
fig.savefig(f, dpi=160, bbox_inches="tight"); print("wrote", f)
