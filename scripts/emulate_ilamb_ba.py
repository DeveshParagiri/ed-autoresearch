"""
EMULATED ILAMB burned-area Collier-2018 score, computed on the 0.5deg grid
directly from burntArea.nc files vs the GFED5 reference. Mirrors the validated
score_BA() in optimize_modelC_coupled.py (bias/RMSE exp-scores, seasonal phase,
Taylor spatial; Overall = (Bias + 2*RMSE + Seas + Spatial)/5).

NOT official ILAMB (no regridding/region machinery), but on the same 0.5deg grid
as official so the DELTA between models is a reliable directional read. Use to
compare candidates; re-score the winner with official ilamb-run before promotion.
"""
import sys
from pathlib import Path
import numpy as np, xarray as xr

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc"


def load_monthly(path, as_fraction=True):
    da = xr.open_dataset(path)["burntArea"]
    yrs = np.array([t.year for t in da.time.values])
    m = (yrs >= 2001) & (yrs <= 2016)
    arr = np.nan_to_num(da.values[m].astype(np.float64), nan=0.0)
    if as_fraction and da.attrs.get("units") in ("%", "percent"):
        arr = arr / 100.0
    return arr, da.lat.values


obs, lat = load_monthly(REF)
cos_lat = np.cos(np.deg2rad(lat))
w2 = np.broadcast_to(cos_lat[:, None], obs.shape[1:]).astype(np.float64)
land_mask = (obs > 0).any(axis=0)
w2_burn = w2 * land_mask
gfed_tm = obs.mean(axis=0)
gfed_std = obs.std(axis=0).clip(1e-12)
gfed_cyc = obs.reshape(16, 12, *obs.shape[1:]).mean(axis=0)
gfed_peak = np.argmax(gfed_cyc, axis=0).astype(np.float32)
mass_w = w2 * gfed_tm
mass_w_burn = mass_w * land_mask
obs_anom = obs - gfed_tm[None]


def score(pred):
    pred = pred * land_mask[None]            # match optimizer land masking
    pred_tm = pred.mean(axis=0)
    bias_s = np.exp(-np.abs(pred_tm - gfed_tm) / gfed_std)
    bias = (bias_s * mass_w).sum() / (mass_w.sum() + 1e-12)
    pred_anom = pred - pred_tm[None]
    crmse = np.sqrt(((pred_anom - obs_anom) ** 2).mean(axis=0))
    rmse = (np.exp(-crmse / gfed_std) * mass_w).sum() / (mass_w.sum() + 1e-12)
    pcyc = pred.reshape(16, 12, *pred.shape[1:]).mean(axis=0)
    ppeak = np.argmax(pcyc, axis=0).astype(np.float32)
    shift = ppeak - gfed_peak
    shift = np.where(shift > 6, shift - 12, shift)
    shift = np.where(shift < -6, shift + 12, shift)
    seas = (((1 + np.cos(np.abs(shift) / 12 * 2 * np.pi)) * 0.5) * mass_w_burn).sum() / (mass_w_burn.sum() + 1e-12)
    of, pf, pw = gfed_tm[land_mask], pred_tm[land_mask], w2_burn[land_mask]
    ow = (of * pw).sum() / pw.sum(); pwm = (pf * pw).sum() / pw.sum()
    oa, pa = of - ow, pf - pwm
    std0 = max(float(np.sqrt(((oa**2)*pw).sum()/pw.sum())), 1e-12)
    std = max(float(np.sqrt(((pa**2)*pw).sum()/pw.sum())), 1e-12)
    denom = float(np.sqrt(((pa**2)*pw).sum() * ((oa**2)*pw).sum()))
    rho = float((pa*oa*pw).sum()/(denom+1e-12))
    sigma = std/std0
    spatial = 2*(1+rho)/((sigma + 1/max(sigma,1e-12))**2)
    overall = (bias + 2*rmse + seas + spatial)/5
    return dict(bias=bias, rmse=rmse, seas=seas, spatial=spatial,
                sigma=sigma, rho=rho, overall=overall)


MODELS = {
    "canonical k4 (current /12)": REPO / "ilamb/MODELS/ED-ModelC-final/burntArea.nc",
    "PRE-tropfix2 (old 1.26x)":   REPO / "backups_PRE-tropfix2/burntArea.canonical.nc",
    "PROTOTYPE 1-exp(-rate/12)":  REPO / "ilamb/MODELS_SEASONAL_PROTO/ED-ModelC-seasonal/burntArea.nc",
}
print("EMULATED ILAMB burned-area scores (0.5deg, vs GFED5) — NOT official\n")
hdr = f"{'model':28s} {'Overall':>8s} {'Bias':>7s} {'RMSE':>7s} {'Seas':>7s} {'Spatial':>8s} {'sigma':>6s} {'rho':>6s}"
print(hdr); print("-"*len(hdr))
for name, p in MODELS.items():
    if not p.exists():
        print(f"{name:28s}  MISSING {p}"); continue
    arr, la = load_monthly(p)
    s = score(arr)
    print(f"{name:28s} {s['overall']:8.4f} {s['bias']:7.4f} {s['rmse']:7.4f} "
          f"{s['seas']:7.4f} {s['spatial']:8.4f} {s['sigma']:6.3f} {s['rho']:6.3f}")
print("\nReminder: emulated absolute values differ from official; trust the DELTAS.")
print("sigma = model/ref spatial std ratio (1.0 = ideal). The ceiling problem shows as sigma<1.")
