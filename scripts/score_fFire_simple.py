"""
Simple fire-carbon-emission scoring per Hurtt's "start simple" directive
from the 2026-05-12 meeting:

  1. Compute per-cell emission factor from GFED4: EF = fFire_GFED / BA_GFED
  2. Apply: pred_fFire(cell, t) = our_BA(cell, t) × EF(cell)
  3. Score against GFED4 fFire reference via ILAMB

This is the no-survivorship (S=0) approach: every cell has a fixed emission
factor inferred empirically from the reference dataset itself.

For each model variant in MODELS_LEADERBOARD, this script reads its burntArea.nc
and writes a derived fFire.nc.

Usage:
    python scripts/score_fFire_simple.py --models ED-ModelC-GFED5 ED-ModelC-ILAMB
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import cftime
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
REF_NC = REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED4.1S" / "fFire.nc"
GFED_BA = REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED4.1S" / "burntArea.nc"
SEC_PER_MONTH = (365.25 / 12) * 86400.0


def compute_per_cell_EF():
    """EF(lat, lon) = mean over time of [fFire_kg_per_m2_per_s / BA_fraction_per_month]
    where BA_fraction is converted from % to fraction and from monthly to per-second.
    Result has units kg C m^-2 s^-1 per (unit burned-fraction per second).
    Practically: EF * (BA_frac per second) = fFire kg/m2/s.
    """
    ff = xr.open_dataset(REF_NC)["fFire"].values  # kg/m^2/s, monthly mean
    ba_pct = xr.open_dataset(GFED_BA)["burntArea"].values  # % per month
    # Align time windows: both 2001-2016 already
    if ff.shape[0] != ba_pct.shape[0]:
        # Trim to overlap
        n = min(ff.shape[0], ba_pct.shape[0])
        ff = ff[:n]; ba_pct = ba_pct[:n]
    # BA fraction per second = (BA_pct / 100) / sec_per_month
    ba_frac_per_sec = (ba_pct / 100.0) / SEC_PER_MONTH
    # Sum over time for numerator and denominator (more stable than per-month ratio)
    num = np.nansum(ff, axis=0)                       # kg/m^2/s summed
    den = np.nansum(ba_frac_per_sec, axis=0)          # 1/s summed
    with np.errstate(divide="ignore", invalid="ignore"):
        EF = num / den                                # kg C / m^2 per unit burned-fraction
    EF = np.where(np.isfinite(EF) & (den > 0), EF, 0.0).astype(np.float32)
    return EF


def make_ffire_for_model(model_name, EF, target_lat, target_lon):
    """Read model burntArea.nc, regrid to 0.5deg if needed, multiply by EF, write fFire.nc."""
    p = REPO / "ilamb" / "MODELS_LEADERBOARD" / model_name / "burntArea.nc"
    ds = xr.open_dataset(p)
    ba_da = ds["burntArea"]
    units = ba_da.attrs.get("units", "1")
    # Regrid to 0.5deg if shape doesn't match
    if ba_da.shape[-2:] != (360, 720):
        print(f"  regridding from {ba_da.shape[-2:]} to (360, 720)")
        ba_da = ba_da.interp(lat=target_lat, lon=target_lon, method="linear")
    ba = ba_da.values
    # Convert to BA_frac per second
    if units in ("1", "fraction", ""):
        ba_frac = ba.astype(np.float64)
    elif units == "%":
        ba_frac = ba.astype(np.float64) / 100.0
    else:
        ba_frac = ba.astype(np.float64)
    # Clean NaN before multiplying
    ba_frac = np.nan_to_num(ba_frac, nan=0.0, posinf=0.0, neginf=0.0)
    ba_frac_per_sec = ba_frac / SEC_PER_MONTH
    # Apply EF
    pred_ffire = (ba_frac_per_sec * EF[None, :, :]).astype(np.float32)

    # Write fFire NC in CF-compliant format matching the reference
    out_dir = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "fFire.nc"

    out_ds = xr.Dataset(
        {"fFire": (("time", "lat", "lon"), pred_ffire,
                    {"units": "kg m-2 s-1",
                     "standard_name": "fire_carbon_flux",
                     "long_name": "Fire carbon emission flux"})},
        coords={"time": ds.time, "lat": target_lat, "lon": target_lon},
        attrs={"title": f"{model_name} fFire (S=0, GFED4-derived per-cell EF)",
               "method": "fFire = burnedArea * EF, where EF = sum_t(GFED4_fFire) / sum_t(GFED4_BA_per_sec)",
               "reference_for_EF": "GFED4.1S",
               "Conventions": "CF-1.7"})

    tb = np.empty((len(ds.time), 2), dtype=object)
    for i, t in enumerate(ds.time.values):
        y, m = t.year, t.month
        tb[i, 0] = cftime.DatetimeNoLeap(y, m, 1)
        tb[i, 1] = cftime.DatetimeNoLeap(y + (m == 12), (m % 12) + 1, 1)
    out_ds = out_ds.assign(time_bounds=(("time", "nb"), tb))
    out_ds.time.attrs.update({"bounds": "time_bounds", "standard_name": "time", "axis": "T"})

    time_units = f"days since {int(ds.time.values[0].year)}-01-01 00:00:00"
    enc = {"fFire": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time": {"units": time_units, "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": time_units, "calendar": "noleap", "dtype": "float64"}}
    tmp = out_p.with_suffix(".nc.tmp")
    out_ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, out_p)
    print(f"  wrote {out_p}  mean={float(np.nanmean(pred_ffire)):.4g}, max={float(np.nanmax(pred_ffire)):.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                     help="Models to process. Default: all leaderboard models.")
    args = ap.parse_args()

    print("Computing per-cell emission factor from GFED4...")
    EF = compute_per_cell_EF()
    print(f"  EF range [{EF.min():.4g}, {EF.max():.4g}]  mean={EF.mean():.4g}")
    print(f"  Non-zero cells: {(EF > 0).sum()} / {EF.size}")

    # Target grid is 0.5deg from the GFED reference
    ref = xr.open_dataset(REF_NC)
    target_lat = ref.lat.values
    target_lon = ref.lon.values
    ref.close()

    if args.models is None:
        models = sorted([p.name for p in (REPO / "ilamb" / "MODELS_LEADERBOARD").iterdir()
                         if p.is_dir() and (p / "burntArea.nc").exists()])
    else:
        models = args.models

    print(f"\nProcessing {len(models)} models...")
    for m in models:
        print(f"\n[{m}]")
        try:
            make_ffire_for_model(m, EF, target_lat, target_lon)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
