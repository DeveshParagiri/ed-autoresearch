"""
B: spatial-pattern goodness-of-fit module — the meeting's alternative optimization
criteria. The point (from the June-09 fire meeting): ILAMB Overall barely rewards
the spatial pattern, so the per-cell 1:1 band stays flat. George's bar is that the
band sits ON the diagonal. This module scores that directly, on the cells that
ACTUALLY BURN (GFED5 > threshold), using the annual-mean burned-fraction map.

Metrics (all on the burning-cell mask, pixel level):
  r        pixel-level spatial Pearson correlation (pattern alignment)
  sigma    std(model)/std(GFED5)                    (dynamic-range match; 1.0 = ideal)
  slope    OLS slope of model ~ GFED5 = r * sigma   (George's "on the 1:1 line"; 1.0 = ideal)
  bias     mean(model) - mean(GFED5)                (signed)
  magx     mean(model)/mean(GFED5)                  (global-mean / magnitude ratio)
  rmse     root mean square per-cell error
  taylor   Taylor skill score in [0,1] rewarding r->1 AND sigma->1 (single scalar
           for optimization: 4(1+r) / [(sigma+1/sigma)^2 (1+r0)], r0=1)

`spatial_metrics()` is importable so the optimizer (step 2, the A+B refit) can use
the same scorer as the objective. The CLI prints a comparison table across models.

Usage:
  python scripts/score_spatial.py                 # built-in model set vs GFED5
  python scripts/score_spatial.py a.nc b.nc ...   # compare given burntArea.nc files
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
GFED_NC = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"

# Built-in comparison set: (label, path). Lei-shipped = the lab-official model.
DEFAULT_MODELS = [
    ("Lei-shipped (PRE-tropfix2)", REPO / "backups_PRE-tropfix2" / "burntArea.canonical.nc"),
    ("tropfix2-k4 (repo canonical)", REPO / "ilamb" / "MODELS" / "ED-ModelC-final" / "burntArea.nc"),
    ("seasonal-k1 (transform refit)",
     REPO / "ilamb" / "MODELS_TOPK_seasonal" / "ED-ModelC-seasonal-k1" / "burntArea.nc"),
]


def period_mean_fraction(path):
    """Period-mean (2001-2016) of the monthly burned-area fraction, units '1'."""
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    sel = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[sel].astype(np.float64), nan=0.0)
    if da.attrs.get("units", "1") in ("%", "percent"):
        arr = arr / 100.0
    da.close() if hasattr(da, "close") else None
    return arr.mean(axis=0)            # (lat, lon) period-mean monthly fraction


def spatial_metrics(model_pm, gfed_pm, mask):
    """Pixel-level spatial-pattern metrics on the boolean `mask` (burning cells).

    model_pm / gfed_pm: 2-D period-mean burned-fraction maps on the same grid.
    Returns a dict. Scale-invariant terms (r, sigma, slope, taylor) are unaffected
    by monthly-vs-annual scaling; bias/magx/rmse are in the input (monthly) units.
    """
    m = model_pm[mask].astype(np.float64)
    g = gfed_pm[mask].astype(np.float64)
    n = m.size
    mm, gm = m.mean(), g.mean()
    ms, gs = m.std(), g.std()
    cov = ((m - mm) * (g - gm)).mean()
    r = cov / (ms * gs + 1e-30)
    sigma = ms / (gs + 1e-30)
    slope = r * sigma                       # OLS slope of model on GFED5
    rmse = float(np.sqrt(((m - g) ** 2).mean()))
    r0 = 1.0
    taylor = 4.0 * (1.0 + r) / (((sigma + 1.0 / (sigma + 1e-30)) ** 2) * (1.0 + r0))
    return {"n": n, "r": float(r), "sigma": float(sigma), "slope": float(slope),
            "bias": float(mm - gm), "magx": float(mm / (gm + 1e-30)),
            "rmse": rmse, "taylor": float(taylor), "cell_max": float(m.max())}


def burning_masks(gfed_pm, land):
    """Two burning-cell masks: any fire (annual>0.1%) and active fire (annual>1%).
    Thresholds are on the ANNUAL fraction = 12 * monthly period-mean."""
    annual = gfed_pm * 12.0
    return {
        "GFED>0.1%/yr (any fire)":   land & (annual > 0.001),
        "GFED>1%/yr (active fire)":  land & (annual > 0.01),
    }


def main(argv):
    gfed_pm = period_mean_fraction(GFED_NC)
    # land = anywhere GFED5 is defined (the reference fill is NaN over ocean)
    graw = xr.open_dataset(GFED_NC)["burntArea"]
    land = np.isfinite(graw.values[0])
    graw.close()

    if len(argv) > 1:
        models = [(Path(p).stem, Path(p)) for p in argv[1:]]
    else:
        models = [(lab, p) for lab, p in DEFAULT_MODELS if p.exists()]

    masks = burning_masks(gfed_pm, land)
    print(f"GFED5 reference: {GFED_NC.relative_to(REPO)}")
    print(f"GFED5 per-cell max {np.nanmax(gfed_pm):.4f}  "
          f"(annual {np.nanmax(gfed_pm)*12:.3f}); the 1:1 target is slope=1, r=1, sigma=1.\n")

    for mlabel, mask in masks.items():
        print(f"=== mask: {mlabel}   (n={int(mask.sum())} cells) ===")
        print(f"{'model':32s} {'r':>6s} {'sigma':>6s} {'slope':>6s} "
              f"{'magx':>6s} {'rmse':>8s} {'taylor':>7s} {'cellmax':>8s}")
        for label, path in models:
            mp = period_mean_fraction(path)
            s = spatial_metrics(mp, gfed_pm, mask)
            print(f"{label:32s} {s['r']:6.3f} {s['sigma']:6.3f} {s['slope']:6.3f} "
                  f"{s['magx']:6.2f} {s['rmse']:8.5f} {s['taylor']:7.4f} {s['cell_max']:8.4f}")
        print()

    print("Reading: slope = r*sigma is the per-cell 1:1 line slope. George's bar is "
          "slope -> 1 (band on the diagonal). taylor in [0,1] is the single scalar that "
          "rewards BOTH alignment (r) and dynamic range (sigma); use it as the B objective.")


if __name__ == "__main__":
    main(sys.argv)
