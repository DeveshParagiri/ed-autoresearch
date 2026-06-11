"""
Held-out-CELLS validation (blocked spatial cross-validation): is the per-continent
+ fuel-form skill genuine or did the regional parameters memorise the map? A model
fit on a checkerboard of 10x10-deg TRAIN tiles is scored on the complementary TEST
tiles (cells it never saw). Blocked tiles (not per-cell) so neighbouring-cell spatial
autocorrelation cannot leak the answer.

If spatial r on TEST tiles ~ TRAIN tiles, the structure generalises to unseen cells
(genuine). A big TEST drop = the regional params overfit the training cells.

Run: python scripts/validate_holdout_cells.py [model.nc ...]
"""
import sys
from pathlib import Path
import numpy as np, xarray as xr

REPO = Path(__file__).resolve().parents[1]
GFED = REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc"


def period_mean(path):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values]); sel = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[sel].astype(np.float64), nan=0.0)
    if da.attrs.get("units", "1") in ("%", "percent"): arr = arr / 100.0
    return arr.mean(axis=0)


def metrics(m_pm, g_pm, mask):
    m = m_pm[mask]; g = g_pm[mask]
    ms, gs = m.std(), g.std()
    r = float(((m - m.mean()) * (g - g.mean())).mean() / (ms * gs + 1e-30))
    sigma = float(ms / (gs + 1e-30))
    taylor = 4.0 * (1.0 + r) / (((sigma + 1.0 / (sigma + 1e-30)) ** 2) * 2.0)
    return r, sigma, r * sigma, taylor


def main(argv):
    models = argv[1:] or [
        "ilamb/MODELS_CONTINENTAL_CELL/ED-ModelC-continental-cell/burntArea.nc"]
    g = period_mean(GFED)
    land = np.isfinite(xr.open_dataset(GFED)["burntArea"].values[0])
    active = land & ((g * 12.0) > 0.01)
    # same 10x10-deg checkerboard tiles as the optimizer's CELL_HOLDOUT
    ti = (np.arange(180)[:, None] // 10); tj = (np.arange(360)[None, :] // 10)
    train_tiles = ((ti + tj) % 2 == 0)
    m_train = active & train_tiles
    m_test = active & ~train_tiles
    print(f"active cells: train tiles n={int(m_train.sum())}  test tiles n={int(m_test.sum())}\n")
    print(f"{'model':38s} {'tiles':6s} {'r':>6s} {'sigma':>6s} {'slope':>6s} {'taylor':>7s}")
    for mp in models:
        name = Path(mp).parent.name
        mpm = period_mean(mp)
        tr = metrics(mpm, g, m_train); te = metrics(mpm, g, m_test)
        print(f"{name:38s} {'TRAIN':6s} {tr[0]:6.3f} {tr[1]:6.3f} {tr[2]:6.3f} {tr[3]:7.4f}")
        print(f"{name:38s} {'TEST':6s} {te[0]:6.3f} {te[1]:6.3f} {te[2]:6.3f} {te[3]:7.4f}")
        print(f"{'  -> r drop train->test':38s} {te[0]-tr[0]:+.3f}\n")
    print("Reading: TEST r close to TRAIN r => the regional params generalise to unseen "
          "cells (genuine structure). A large negative drop => the per-continent fit "
          "overfit the training cells.")


if __name__ == "__main__":
    main(sys.argv)
