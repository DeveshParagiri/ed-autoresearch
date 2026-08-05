"""Fire-presence F1 for ED-stock vs Model E against GFED5.

Units gotcha (CLAUDE.md): GFED5 burntArea is in PERCENT, model output is FRACTION.
Convert GFED5 /100 before any comparison.
"""
import numpy as np
import xarray as xr

REF = r"D:\FIRE_OFFLINE\ilamb_ref_official\DATA\burntArea\GFED5\burntArea.nc"
STOCK = r"D:\FIRE_OFFLINE\paper_gmd\models\ED-stock\burntArea.nc"
E = r"D:\FIRE_OFFLINE\paper_gmd\models\E-clean\burntArea.nc"


def load(p, scale=1.0):
    ds = xr.open_dataset(p, decode_times=False)
    a = ds["burntArea"].values.astype("float64") * scale
    return a, ds["lat"].values, ds["lon"].values


ref, rlat, rlon = load(REF, 1.0 / 100.0)   # % -> fraction
stk, slat, slon = load(STOCK)
mdl, elat, elon = load(E)

assert np.allclose(rlat, slat) and np.allclose(rlat, elat), "lat mismatch"
assert np.allclose(rlon, slon) and np.allclose(rlon, elon), "lon mismatch"

n = min(ref.shape[0], stk.shape[0], mdl.shape[0])
ref, stk, mdl = ref[:n], stk[:n], mdl[:n]
print(f"months compared: {n} (2001-{2001 + n // 12 - 1})")

# land / valid mask: cells where all three have data at least somewhere in time
valid = np.isfinite(ref).any(0) & np.isfinite(stk).any(0) & np.isfinite(mdl).any(0)
print("valid cells:", int(valid.sum()))

ref = np.nan_to_num(ref)
stk = np.nan_to_num(stk)
mdl = np.nan_to_num(mdl)


def f1(obs_bin, pred_bin):
    tp = np.sum(obs_bin & pred_bin)
    fp = np.sum(~obs_bin & pred_bin)
    fn = np.sum(obs_bin & ~pred_bin)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f, prec, rec, int(tp), int(fp), int(fn)


# ---- (a) annual-climatology presence (spatial pattern of where fire occurs) ----
print("\n=== ANNUAL-MEAN (climatological) fire presence, per grid cell ===")
ref_a = ref.reshape(n // 12, 12, *ref.shape[1:]).sum(1).mean(0)   # fraction burned per year
stk_a = stk.reshape(n // 12, 12, *stk.shape[1:]).sum(1).mean(0)
mdl_a = mdl.reshape(n // 12, 12, *mdl.shape[1:]).sum(1).mean(0)
print(f"{'thresh(/yr)':>12} {'ED-stock F1':>12} {'ModelE F1':>10} "
      f"{'obs cells':>10} {'stock P/R':>16} {'E P/R':>16}")
for thr in (1e-4, 5e-4, 1e-3, 5e-3, 1e-2):
    o = (ref_a > thr) & valid
    s = (stk_a > thr) & valid
    m = (mdl_a > thr) & valid
    fs, ps, rs, *_ = f1(o, s)
    fm, pm, rm, *_ = f1(o, m)
    print(f"{thr:>12.0e} {fs:>12.3f} {fm:>10.3f} {int(o.sum()):>10} "
          f"{ps:>7.3f}/{rs:<8.3f} {pm:>7.3f}/{rm:<8.3f}")

# ---- (b) cell-month presence (space and time) ----
print("\n=== MONTHLY cell-month fire presence ===")
v3 = np.broadcast_to(valid, ref.shape)
print(f"{'thresh(/mo)':>12} {'ED-stock F1':>12} {'ModelE F1':>10} {'obs cell-months':>16}")
for thr in (1e-5, 1e-4, 1e-3, 1e-2):
    o = (ref > thr) & v3
    s = (stk > thr) & v3
    m = (mdl > thr) & v3
    fs, *_ = f1(o, s)
    fm, *_ = f1(o, m)
    print(f"{thr:>12.0e} {fs:>12.3f} {fm:>10.3f} {int(o.sum()):>16}")

# sanity: global totals (Mha/yr), area-weighted
R = 6371000.0
lat_edges = np.deg2rad(np.linspace(-90, 90, ref.shape[1] + 1))
dlon = np.deg2rad(360.0 / ref.shape[2])
cell = (R ** 2 * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])))[:, None]
area = np.broadcast_to(cell, ref_a.shape)
tomha = 1e-10
print("\nsanity, area-weighted annual burned area (Mha/yr):")
for nm, a in (("GFED5", ref_a), ("ED-stock", stk_a), ("Model E", mdl_a)):
    print(f"  {nm:>9}: {np.sum(a * area * valid) * tomha:8.1f}")
