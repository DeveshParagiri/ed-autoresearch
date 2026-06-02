"""
dump_model_c_terms.py
=====================
Produces a NetCDF with every intermediate term of the Model C formula exactly
as computed in the offline optimizer, so the integrated-in-ED run can be
diffed term-by-term against it.

Requested by Lei (2026-04-23) after the first integrated run differed from
the offline burntArea by "several orders of magnitude."

Model C formula (from scripts/reproduce_v2.py :: fire_C, kept in sync):

    onset    = sig (dbar,      k1,  D_low)
    suppress = supp(dbar,      k2,  D_high)
    p_floor  = p_ann / (p_ann + P_half)
    p_damp   = 1 / (1 + p_month / pre_dampen_half)
    gpp_mod  = hump(gpp_af * gpp_monthly, gpp_b, gpp_d)
    ign_mod  = sig (t_air,     ign_k, ign_c)
    product  = onset * suppress * p_floor * p_damp * gpp_mod * ign_mod
    raw      = clip(product, 0, inf) ** fire_exp
    final    = raw * (GFED_land_mean / raw_land_mean)   <-- OFFLINE POST-HOC RESCALE

    sig(x,k,c)  = 1 / (1 + exp(-k*(x-c)))
    supp(x,k,c) = 1 / (1 + exp( k*(x-c)))
    hump(x,b,d) = (1 - exp(-x/b)) * exp(-x/d)

IMPORTANT for the integrated run:
    The offline file models/ED-ModelC-final/burntArea.nc has the GFED
    land-mean rescale baked in (factor `rescale_k` below). The integrated
    ED run will NOT have this rescale, so its `raw` term is what should be
    compared to `raw` here — not to the published burntArea.nc.

Grid: 1-deg regular lat/lon (180 x 360), S->N, -180..180 lon, monthly
      2001-01 through 2016-12 (192 time steps).

Outputs:
    out_terms/ed_model_c_terms.nc        — every term (10 vars, 192 x 180 x 360)
    out_terms/ed_model_c_terms_meta.json — params + rescale factor + driver paths

Usage:
    python scripts/dump_model_c_terms.py
    python scripts/dump_model_c_terms.py --site 45.5 -120.5   # print table at one cell
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
from pathlib import Path

import cftime
import h5py
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from reproduce_v2 import (
    sig, supp, hump, coarsen, load_drivers, load_gfed_1deg, _add_time_bounds,
)

YEARS      = list(range(2001, 2017))
N_MONTHS   = 192
OUT_DIR    = REPO / "out_terms"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_FP  = REPO / "models" / "C" / "params.json"
TIME_UNITS = "days since 2001-01-01 00:00:00"
TIMES      = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]


def compute_terms(d, p):
    """Return dict of (term_name -> (192,180,360) float32) for Model C."""
    onset    = sig (d["dbar"],    p["k1"],  p["D_low"]).astype(np.float32)
    suppress = supp(d["dbar"],    p["k2"],  p["D_high"]).astype(np.float32)
    p_floor  = (d["p_ann"] / (d["p_ann"] + p["P_half"] + 1e-12)).astype(np.float32)
    p_damp   = (1.0 / (1.0 + d["p_month"] / (p["pre_dampen_half"] + 1e-12))).astype(np.float32)
    gpp_scaled = (p["gpp_af"] * d["gpp_monthly"]).astype(np.float32)
    gpp_mod  = hump(gpp_scaled, p["gpp_b"], p["gpp_d"]).astype(np.float32)
    ign_mod  = sig (d["t_air"],   p["ign_k"], p["ign_c"]).astype(np.float32)
    product  = (onset * suppress * p_floor * p_damp * gpp_mod * ign_mod).astype(np.float32)
    raw      = np.power(np.clip(product, 0, None), p["fire_exp"]).astype(np.float32)
    return {
        "dbar":        d["dbar"].astype(np.float32),
        "p_ann":       d["p_ann"].astype(np.float32),
        "p_month":     d["p_month"].astype(np.float32),
        "t_air":       d["t_air"].astype(np.float32),
        "gpp_monthly": d["gpp_monthly"].astype(np.float32),
        "onset":       onset,
        "suppress":    suppress,
        "p_floor":     p_floor,
        "p_damp":      p_damp,
        "gpp_mod":     gpp_mod,
        "ign_mod":     ign_mod,
        "product":     product,
        "burntArea_raw": raw,
    }


TERM_META = {
    # name           : (units,      long_name)
    "dbar":            ("mm",       "CRUJRA cumulative precipitation deficit (D-bar)"),
    "p_ann":           ("mm/yr",    "CRUJRA annual precipitation (rolling sum)"),
    "p_month":         ("mm/month", "CRUJRA monthly precipitation"),
    "t_air":           ("degC",     "CRUJRA 2-m monthly air temperature"),
    "gpp_monthly":     ("kgC/m2/yr","TRENDY v14 EDv3_S3 GPP, monthly, annualized units"),
    "onset":           ("1",        "sig(dbar, k1, D_low)"),
    "suppress":        ("1",        "supp(dbar, k2, D_high)"),
    "p_floor":         ("1",        "p_ann / (p_ann + P_half)"),
    "p_damp":          ("1",        "1 / (1 + p_month / pre_dampen_half)"),
    "gpp_mod":         ("1",        "hump(gpp_af * gpp_monthly, gpp_b, gpp_d)"),
    "ign_mod":         ("1",        "sig(t_air, ign_k, ign_c)"),
    "product":         ("1",        "onset * suppress * p_floor * p_damp * gpp_mod * ign_mod"),
    "burntArea_raw":   ("1",        "clip(product,0,inf)^fire_exp (BEFORE GFED rescale)"),
}


def write_nc(terms, rescale_k, params):
    lat = np.arange(-89.5, 90.0, 1.0).astype(np.float32)
    lon = np.arange(-179.5, 180.0, 1.0).astype(np.float32)

    data_vars = {}
    for k, v in terms.items():
        units, long_name = TERM_META[k]
        data_vars[k] = (("time", "lat", "lon"), v,
                        {"units": units, "long_name": long_name})
    ds = xr.Dataset(
        data_vars,
        coords={"time": ("time", TIMES),
                "lat":  ("lat", lat),
                "lon":  ("lon", lon)},
        attrs={
            "title": "ED Model C — intermediate terms (offline, 1-deg)",
            "description": ("Per-term dump of the offline Model C burntArea "
                            "formula. Variable 'burntArea_raw' is the output "
                            "BEFORE the GFED land-mean rescale applied to the "
                            "published burntArea.nc."),
            "rescale_factor_applied_to_published_burntArea": float(rescale_k),
            "rescale_note": ("published burntArea = burntArea_raw * "
                             "rescale_factor (applied only over GFED land mask)"),
            "formula": ("burntArea_raw = (onset*suppress*p_floor*p_damp*"
                        "gpp_mod*ign_mod)^fire_exp"),
            "reference_script": "scripts/reproduce_v2.py :: fire_C",
            "params": json.dumps(params),
            "Conventions": "CF-1.7",
        })
    ds = _add_time_bounds(ds)

    enc = {k: {"zlib": True, "complevel": 4, "_FillValue": 1e20, "dtype": "float32"}
           for k in terms}
    enc["time"]        = {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"}
    enc["time_bounds"] = {"units": TIME_UNITS, "calendar": "noleap", "dtype": "float64"}

    out = OUT_DIR / "ed_model_c_terms.nc"
    tmp = out.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    import os
    os.replace(tmp, out)
    print(f"[write] {out}   ({out.stat().st_size/1e6:.1f} MB)")
    return out


def print_site_table(terms, lat_q, lon_q):
    i = int(np.round(lat_q + 89.5))
    j = int(np.round(lon_q + 179.5))
    i = np.clip(i, 0, 179); j = np.clip(j, 0, 359)
    print(f"\n--- Site dump at grid cell lat={lat_q}, lon={lon_q}  "
          f"(i={i}, j={j}) ---")
    hdr = ["yyyy-mm"] + list(terms.keys())
    print("  ".join(f"{h:>14}" for h in hdr))
    for t_idx in range(N_MONTHS):
        y, m = YEARS[t_idx // 12], (t_idx % 12) + 1
        row = [f"{y}-{m:02d}"]
        for k in terms:
            row.append(f"{float(terms[k][t_idx, i, j]):.6g}")
        print("  ".join(f"{c:>14}" for c in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", nargs=2, type=float, metavar=("LAT", "LON"),
                    help="Also print per-month table at this grid cell")
    args = ap.parse_args()

    print(f"[load] Model C params:  {PARAMS_FP}")
    params = json.load(open(PARAMS_FP))["params"]
    for k, v in params.items():
        print(f"    {k:>20s} = {v:.6g}")

    print("[load] drivers (same as offline optimizer) ...")
    drivers = load_drivers()

    print("[compute] Model C intermediate terms ...")
    terms = compute_terms(drivers, params)

    # Compute the same land-mean rescale that rescale_and_write applies so
    # Lei knows exactly what factor separates burntArea_raw from the published
    # burntArea.nc.
    print("[load] GFED 1-deg reference (for rescale factor) ...")
    obs = load_gfed_1deg()
    lat_1d = np.arange(-89.5, 90.0, 1.0)
    cos_lat = np.cos(np.deg2rad(lat_1d)).astype(np.float32)
    w3 = np.broadcast_to(cos_lat[None, :, None], (N_MONTHS, 180, 360))
    land_mask = (obs > 0).any(axis=0)
    w3_land = w3 * land_mask[None, :, :]
    raw = terms["burntArea_raw"]
    raw_land_mean  = float((raw * w3_land).sum() / (w3_land.sum() + 1e-12))
    gfed_land_mean = float((obs * w3_land).sum() / (w3_land.sum() + 1e-12))
    rescale_k = gfed_land_mean / raw_land_mean if raw_land_mean > 0 else 0.0
    print(f"[diag] burntArea_raw land-mean  = {raw_land_mean:.6g}")
    print(f"[diag] GFED          land-mean  = {gfed_land_mean:.6g}")
    print(f"[diag] offline rescale factor   = {rescale_k:.4g}  "
          f"(published burntArea = raw * this, only where GFED land mask is True)")
    print(f"[diag] land cells                = {int(land_mask.sum())} / {land_mask.size} "
          f"({100*land_mask.mean():.1f}%)")

    out = write_nc(terms, rescale_k, params)

    meta = {
        "model": "ED Model C",
        "params": params,
        "rescale_factor_published_over_raw": rescale_k,
        "gfed_land_mean": gfed_land_mean,
        "raw_land_mean": raw_land_mean,
        "land_cells": int(land_mask.sum()),
        "grid": "1-deg regular, 180x360, S->N, -180..180",
        "time": f"{YEARS[0]}-01 to {YEARS[-1]}-12 monthly",
        "driver_sources": {
            "dbar,p_ann,p_month,t_air":  "data/crujra/*_monthly.npy",
            "gpp_monthly":               "data/trendy_v14/EDv3_S3_gpp.nc (slice 3612:3804, coarsened 0.5->1 deg)",
        },
        "output_nc":  str(out),
        "formula":    "burntArea_raw = (onset*suppress*p_floor*p_damp*gpp_mod*ign_mod)^fire_exp",
        "reference":  "scripts/reproduce_v2.py :: fire_C (authoritative)",
        "note_for_integrated_run": (
            "Compare integrated-ED terms to THESE offline terms one-by-one. "
            "Do NOT compare integrated output to models/ED-ModelC-final/burntArea.nc — "
            "that file has the GFED land-mean rescale baked in. Compare to "
            "burntArea_raw instead, or un-apply the rescale factor above."
        ),
    }
    meta_fp = OUT_DIR / "ed_model_c_terms_meta.json"
    meta_fp.write_text(json.dumps(meta, indent=2, default=float))
    print(f"[write] {meta_fp}")

    if args.site:
        print_site_table(terms, args.site[0], args.site[1])

    print("\n[done]")


if __name__ == "__main__":
    main()
