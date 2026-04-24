"""Dump Model C's 6 intermediate terms + 5 input drivers to a single NetCDF.

Useful for debugging / cross-comparison against a coupled ED run. For each site
(lat, lon), you can pull every input + every mechanism term month by month and
diff them against the coupled-ED equivalents.

Variables (all 1 deg, 192 months, 2001-2016):
  Inputs        : dbar, t_air, p_ann, p_month, gpp_monthly
  Intermediate  : onset, suppress, p_floor, p_damp, gpp_mod, ign_mod
  Combined      : product, burntArea_raw  (raw = product ^ fire_exp, pre rescale)

Also supports `--site LAT LON` to print a per-month table at one grid cell.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

from reproduce_modelC import (
    sig, supp, hump, load_drivers, add_cf_bounds,
    REPO, YEARS, N_MONTHS,
)


OUT_DIR = REPO / "out_terms"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_terms(d, p):
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
        "dbar":          d["dbar"].astype(np.float32),
        "p_ann":         d["p_ann"].astype(np.float32),
        "p_month":       d["p_month"].astype(np.float32),
        "t_air":         d["t_air"].astype(np.float32),
        "gpp_monthly":   d["gpp_monthly"].astype(np.float32),
        "onset":         onset,
        "suppress":      suppress,
        "p_floor":       p_floor,
        "p_damp":        p_damp,
        "gpp_mod":       gpp_mod,
        "ign_mod":       ign_mod,
        "product":       product,
        "burntArea_raw": raw,
    }


TERM_META = {
    "dbar":          ("mm",         "Accumulated precipitation deficit (Thornthwaite + daylength)"),
    "p_ann":         ("mm/yr",      "CRUJRA annual precipitation"),
    "p_month":       ("mm/month",   "CRUJRA monthly precipitation"),
    "t_air":         ("degC",       "CRUJRA monthly 2-m air temperature"),
    "gpp_monthly":   ("kg/m2/yr",   "TRENDY v14 EDv3 S3 GPP"),
    "onset":         ("1",          "sig(dbar, k1, D_low)"),
    "suppress":      ("1",          "supp(dbar, k2, D_high)"),
    "p_floor":       ("1",          "p_ann / (p_ann + P_half)"),
    "p_damp":        ("1",          "1 / (1 + p_month / pre_dampen_half)"),
    "gpp_mod":       ("1",          "hump(gpp_af * gpp_monthly, gpp_b, gpp_d)"),
    "ign_mod":       ("1",          "sig(t_air, ign_k, ign_c)"),
    "product":       ("1",          "onset * suppress * p_floor * p_damp * gpp_mod * ign_mod"),
    "burntArea_raw": ("1",          "product ^ fire_exp (BEFORE GFED land-mean rescale)"),
}


def print_site_table(terms, lat_q, lon_q):
    # 1 deg grid: S->N starting at -89.5, -179.5..179.5
    i = int(np.round(lat_q + 89.5))
    j = int(np.round(lon_q + 179.5))
    i = int(np.clip(i, 0, 179))
    j = int(np.clip(j, 0, 359))
    print(f"\n--- Site dump at grid cell lat={lat_q}, lon={lon_q}  (i={i}, j={j}) ---")
    keys = list(terms.keys())
    hdr = ["yyyy-mm"] + keys
    print("  ".join(f"{h:>14}" for h in hdr))
    for t_idx in range(N_MONTHS):
        y, m = YEARS[t_idx // 12], (t_idx % 12) + 1
        row = [f"{y}-{m:02d}"]
        for k in keys:
            row.append(f"{float(terms[k][t_idx, i, j]):.6g}")
        print("  ".join(f"{c:>14}" for c in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", nargs=2, type=float, metavar=("LAT", "LON"),
                    help="Also print per-month table at this grid cell")
    args = ap.parse_args()

    params = json.load(open(REPO / "models" / "C" / "params.json"))["params"]
    print(f"Using params: {params}")

    print("Loading drivers ...")
    d = load_drivers()

    print("Computing terms ...")
    terms = compute_terms(d, params)

    # Build dataset
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in YEARS for m in range(1, 13)]
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(-179.5, 180.0, 1.0)

    data_vars = {}
    for k, arr in terms.items():
        u, ln = TERM_META[k]
        data_vars[k] = (("time", "lat", "lon"), arr, {"units": u, "long_name": ln})

    ds = xr.Dataset(
        data_vars,
        coords={"time": ("time", times), "lat": ("lat", lat), "lon": ("lon", lon)},
        attrs={
            "title": "ED Model C — per-term dump for coupled-ED cross-comparison",
            "formula": "burntArea_raw = (onset*suppress*p_floor*p_damp*gpp_mod*ign_mod)^fire_exp",
            "reference_script": "scripts/reproduce_modelC.py :: fire_C",
            "note_for_integrated_run": (
                "Compare integrated-ED terms to these offline terms one-by-one. "
                "Do NOT compare integrated output to the published ilamb/MODELS/"
                "ED-ModelC-final/burntArea.nc -- that file has the GFED land-mean "
                "rescale baked in. Compare to burntArea_raw instead, and apply the "
                "scale factor from the reproduce_modelC.py stdout if needed."
            ),
            "params": json.dumps(params),
            "Conventions": "CF-1.7",
        },
    )
    ds = add_cf_bounds(ds)

    enc = {k: {"zlib": True, "complevel": 4, "_FillValue": 1e20, "dtype": "float32"}
           for k in terms}
    time_units = "days since 2001-01-01 00:00:00"
    enc["time"]        = {"units": time_units, "calendar": "noleap", "dtype": "float64"}
    enc["time_bounds"] = {"units": time_units, "calendar": "noleap", "dtype": "float64"}

    dst = OUT_DIR / "modelC_terms.nc"
    tmp = dst.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, dst)
    print(f"wrote {dst} ({dst.stat().st_size/1e6:.1f} MB)")

    if args.site:
        print_site_table(terms, args.site[0], args.site[1])


if __name__ == "__main__":
    main()
