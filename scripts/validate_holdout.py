"""
Held-out-years validation: is the per-continent + fuel-form skill GENUINE or
overfit? A model fit on 2001-2012 (TRAIN) is scored on 2013-2016 (TEST, unseen).
If the spatial pattern skill (r, Taylor) on TEST ~ TRAIN, the structure
generalizes (genuine science). A big TEST drop = the fit memorized the training
period (overfit).

Computes spatial metrics (r, sigma, slope, Taylor) on the active-fire mask for the
TRAIN and TEST year windows separately, for each model nc passed (default: the
held-out-trained continental model). The mask uses the FULL-period GFED5 active
cells so train/test see the same cells.

Run: python scripts/validate_holdout.py [model.nc ...]
"""
import sys
from pathlib import Path
import numpy as np, xarray as xr

REPO = Path(__file__).resolve().parents[1]
GFED = REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc"
TRAIN = (2001, 2012)
TEST = (2013, 2016)


def period_mean(path, y0, y1):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    sel = (yrs >= y0) & (yrs <= y1)
    arr = np.nan_to_num(da.values[sel].astype(np.float64), nan=0.0)
    if da.attrs.get("units", "1") in ("%", "percent"):
        arr = arr / 100.0
    return arr.mean(axis=0)


def metrics(model_pm, gfed_pm, mask):
    m = model_pm[mask]; g = gfed_pm[mask]
    ms, gs = m.std(), g.std()
    r = float(((m - m.mean()) * (g - g.mean())).mean() / (ms * gs + 1e-30))
    sigma = float(ms / (gs + 1e-30))
    taylor = 4.0 * (1.0 + r) / (((sigma + 1.0 / (sigma + 1e-30)) ** 2) * 2.0)
    return r, sigma, r * sigma, taylor


def main(argv):
    models = argv[1:] or [
        "ilamb/MODELS/paper/Model-E-ho/burntArea.nc",
        "ilamb/MODELS_CONTINENTAL_HO/ED-ModelC-continental-ho/burntArea.nc",
    ]
    models = [m for m in models if (REPO / m).is_file()] or models
    # active-fire mask from full-period GFED5 (same cells for train and test)
    gfull = period_mean(GFED, 2001, 2016)
    land = np.isfinite(xr.open_dataset(GFED)["burntArea"].values[0])
    mask = land & ((gfull * 12.0) > 0.01)

    g_tr = period_mean(GFED, *TRAIN); g_te = period_mean(GFED, *TEST)
    print(f"active-fire cells n={int(mask.sum())}   TRAIN {TRAIN}  TEST {TEST}\n")
    print(f"{'model':40s} {'split':6s} {'r':>6s} {'sigma':>6s} {'slope':>6s} {'taylor':>7s}")
    for mp in models:
        name = Path(mp).parent.name
        tr = metrics(period_mean(mp, *TRAIN), g_tr, mask)
        te = metrics(period_mean(mp, *TEST), g_te, mask)
        print(f"{name:40s} {'TRAIN':6s} {tr[0]:6.3f} {tr[1]:6.3f} {tr[2]:6.3f} {tr[3]:7.4f}")
        print(f"{name:40s} {'TEST':6s} {te[0]:6.3f} {te[1]:6.3f} {te[2]:6.3f} {te[3]:7.4f}")
        print(f"{'  -> r drop train->test':40s} {te[0]-tr[0]:+.3f}\n")
    print("Reading: TEST r close to TRAIN r => the spatial structure generalizes to "
          "unseen years (genuine). A large negative r drop => overfit to the training period.")


if __name__ == "__main__":
    main(sys.argv)
