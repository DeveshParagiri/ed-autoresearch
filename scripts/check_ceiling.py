from pathlib import Path
import numpy as np, xarray as xr

REPO = Path(__file__).resolve().parents[1]

MODELS = {
    "GFED5 (ref)":      REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc",
    "canonical k4":     REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc",
    "spatial-k1":       REPO / "ilamb/MODELS_TOPK_spatial/ED-ModelC-spatial-k1/burntArea.nc",
    "continental":      REPO / "ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc",
}


def period_mean(path):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values]); sel = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[sel].astype(np.float64), nan=0.0)
    if da.attrs.get("units", "1") in ("%", "percent"): arr = arr / 100.0
    return arr.mean(axis=0)


gfed = period_mean(MODELS["GFED5 (ref)"])
mask = np.isfinite(gfed) & ((gfed * 12.0) > 0.01)
g = gfed[mask]

print(f"{'model':<16}{'max':>9}{'p99.9':>9}{'p99':>9}{'sigma':>8}{'slope':>8}{'r':>7}")
for name, p in MODELS.items():
    if not p.exists():
        print(f"{name:<16} MISSING {p}")
        continue
    m = period_mean(p)[mask]
    r = np.corrcoef(m, g)[0, 1]
    sigma = m.std() / g.std()
    slope = r * sigma
    print(f"{name:<16}{m.max():>9.4f}{np.percentile(m,99.9):>9.4f}"
          f"{np.percentile(m,99):>9.4f}{sigma:>8.3f}{slope:>8.3f}{r:>7.3f}")
