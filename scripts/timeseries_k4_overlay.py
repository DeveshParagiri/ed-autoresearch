"""Global monthly burned-area timeseries: GFED5 vs canonical vs tropfix2-k4.
Same loader + area-weighting as maps_seasonal.py figure 1, with k4 overlaid."""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "NEW MAPS" / "tropfix2"
OUT.mkdir(parents=True, exist_ok=True)


def load_ba(path, to_pct=True):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    m = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[m].astype(np.float64), nan=0.0)
    units = da.attrs.get("units", "1")
    if to_pct and units in ("1", "fraction", ""):
        arr = arr * 100.0
    return arr, da.lat.values, da.lon.values


CANON = REPO / "ilamb" / "MODELS" / "ED-ModelC-final" / "burntArea.nc"
K4    = REPO / "ilamb" / "MODELS_TOPK_tropfix2" / "ED-ModelC-tropfix2-k4" / "burntArea.nc"
GFED  = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc"

cano, lat, lon = load_ba(CANON)
k4,   _,   _   = load_ba(K4)
truth,_,   _   = load_ba(GFED)

R = 6.371e6; dlon = np.deg2rad(0.5)
area_lat = (R**2) * dlon * (np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25)))
area2d = np.abs(np.broadcast_to(area_lat[:, None], (len(lat), len(lon))))


def ts_Mha(monthly_pct):
    return ((monthly_pct / 100.0) * area2d[None, :, :] / 1e10).sum(axis=(1, 2))


truth_ts, cano_ts, k4_ts = ts_Mha(truth), ts_Mha(cano), ts_Mha(k4)
t = np.arange(192) / 12.0 + 2001

print(f"GFED5     mean {truth_ts.mean():.1f} Mha/month")
print(f"canonical mean {cano_ts.mean():.1f} Mha/month ({cano_ts.mean()/truth_ts.mean():.2f}x)")
print(f"k4        mean {k4_ts.mean():.1f} Mha/month ({k4_ts.mean()/truth_ts.mean():.2f}x)")

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(t, truth_ts, color="k", lw=1.3, label="GFED5")
ax.plot(t, cano_ts, color="firebrick", lw=1.1, alpha=0.55,
        label=f"ED-ModelC canonical ({cano_ts.mean()/truth_ts.mean():.2f}x)")
ax.plot(t, k4_ts, color="tab:blue", lw=1.3,
        label=f"ED-ModelC tropfix2-k4 ({k4_ts.mean()/truth_ts.mean():.2f}x)")
ax.set_xlabel("Year"); ax.set_ylabel("Global burned area (Mha / month)")
ax.set_title("Global monthly burned area, 2001-2016: canonical vs tropfix2-k4")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fp = OUT / "1_global_timeseries_k4.png"
fig.savefig(fp, dpi=150, bbox_inches="tight")
print(f"wrote {fp}")
