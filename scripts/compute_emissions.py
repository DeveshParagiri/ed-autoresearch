"""Process-based fFire using the Hurtt/Lei formula on local inputs.

    fFire = burned_fraction × Σ_k (C_k × β_k(θ)) / Δt

Fuel pools:
    C_leaf  = 0.05 · AGB
    C_fine  = 0.10 · AGB
    C_coarse= 0.85 · AGB
    C_litter= 0.15 · cSoil

β_k(θ) = β_k,max · clip(D_bar / D_REF, 0, 1)
β_max  = {leaf:0.90, fine:0.80, coarse:0.35, litter:0.80}  (CLM5 defaults)
D_REF  = 150 mm

Inputs (local):
  global_baseline_modelCfuel_inputs_1997-2016.nc  → AGB (240, 360, 720)
  data/trendy_v14/EDv3_S3_cSoil.nc                → cSoil (annual, 360, 720)
  data/crujra/dbar_monthly.npy                    → D-bar at 1° (192, 180, 360)
  ilamb/MODELS_LEADERBOARD/<MODEL>/burntArea.nc   → BA (192, 360, 720)

Output:
  ilamb/MODELS_LEADERBOARD_FFIRE_GFED5/<MODEL>-Hurtt/fFire.nc

Usage:
  python scripts/compute_ffire_hurtt.py --model ED-ModelC-GFED5
  python scripts/compute_ffire_hurtt.py --model ED-ModelC-GFED5 --d-ref 200
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import cftime, numpy as np, xarray as xr

REPO = Path(__file__).resolve().parents[1]
SEC_PER_MONTH = (365.25 / 12) * 86400.0

POOL_FRAC = {"leaf": 0.05, "fine": 0.10, "coarse": 0.85, "litter": 0.15}
BETA_MAX = {"leaf": 0.90, "fine": 0.80, "coarse": 0.35, "litter": 0.80}


def load_agb_2001_2016():
    """AGB at 0.5° from fuel inputs NC. Returns (192, 360, 720) kgC/m2."""
    ds = xr.open_dataset(REPO / "global_baseline_modelCfuel_inputs_1997-2016.nc")
    yr = np.array([t.year for t in ds["time"].values])
    m = (yr >= 2001) & (yr <= 2016)
    arr = ds["AGB"].values[m].astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def load_csoil_2001_2016(lat_grid, lon_grid):
    """cSoil at 0.5°, broadcast from annual to monthly. Returns (192, 360, 720)."""
    ds = xr.open_dataset(REPO / "data" / "trendy_v14" / "EDv3_S3_cSoil.nc")
    yr = np.array([t.astype("datetime64[Y]").astype(int) + 1970 for t in ds["time"].values])
    mask = (yr >= 2001) & (yr <= 2016)
    annual = ds["cSoil"].values[mask].astype(np.float32)  # (16, 360, 720)
    csoil_lat = ds["latitude"].values
    if csoil_lat[0] > csoil_lat[-1]:  # N→S, flip to S→N
        annual = annual[:, ::-1, :]
    monthly = np.repeat(annual, 12, axis=0)  # (192, 360, 720)
    return np.nan_to_num(monthly, nan=0.0, posinf=0.0, neginf=0.0)


def load_dbar_2001_2016():
    """D-bar at 1°, upsampled to 0.5°. Returns (192, 360, 720) mm."""
    dbar_1 = np.load(REPO / "data" / "crujra" / "dbar_monthly.npy")  # (192, 180, 360)
    return np.repeat(np.repeat(dbar_1, 2, axis=-2), 2, axis=-1).astype(np.float32)


def load_model_ba(model_name, ba_path=None):
    p = Path(ba_path) if ba_path else REPO / "ilamb" / "MODELS_LEADERBOARD" / model_name / "burntArea.nc"
    ds = xr.open_dataset(p)
    yr = np.array([t.year for t in ds["time"].values])
    m = (yr >= 2001) & (yr <= 2016)
    ba = ds["burntArea"].values[m].astype(np.float32)
    return ba, ds["lat"].values, ds["lon"].values, ds["time"].values[m]


def compute_ffire(ba_frac, agb, csoil, dbar, d_ref=150.0):
    """Returns fFire in kgC/m2/s with shape (192, 360, 720)."""
    m_theta = np.clip(dbar / d_ref, 0.0, 1.0).astype(np.float32)
    beta = {k: (BETA_MAX[k] * m_theta) for k in BETA_MAX}
    C_pool = {
        "leaf":   POOL_FRAC["leaf"]   * agb,
        "fine":   POOL_FRAC["fine"]   * agb,
        "coarse": POOL_FRAC["coarse"] * agb,
        "litter": POOL_FRAC["litter"] * csoil,
    }
    C_combust = np.zeros_like(agb, dtype=np.float32)
    for k in C_pool:
        C_combust += C_pool[k] * beta[k]
    fuel_consumed = ba_frac * C_combust  # kgC/m2/month
    return (fuel_consumed / SEC_PER_MONTH).astype(np.float32), C_combust


def write_ffire(out_path, ffire, lat, lon, times, src_model, d_ref):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tb = np.empty((len(times), 2), dtype=object)
    for i, t in enumerate(times):
        y, m = t.year, t.month
        tb[i, 0] = cftime.DatetimeNoLeap(y, m, 1)
        tb[i, 1] = cftime.DatetimeNoLeap(y + (m == 12), (m % 12) + 1, 1)
    out = xr.Dataset(
        {"fFire": (("time", "lat", "lon"), ffire,
                    {"units": "kg m-2 s-1", "long_name": "Fire Carbon Flux",
                     "standard_name": "fire_carbon_flux"})},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"title": f"{src_model}-Hurtt (process-based fFire, SPITFIRE 4-pool)",
               "method": "fFire = BA × Σ_k (POOL_FRAC_k · C_src_k · β_k(θ)) / Δt",
               "POOL_FRAC": str(POOL_FRAC), "BETA_MAX": str(BETA_MAX),
               "D_REF_mm": d_ref, "burned_area_source": src_model,
               "Conventions": "CF-1.7"})
    out = out.assign(time_bounds=(("time", "nb"), tb))
    out.time.attrs.update({"bounds": "time_bounds", "standard_name": "time", "axis": "T"})
    t0 = int(times[0].year)
    enc = {"fFire": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
           "time": {"units": f"days since {t0}-01-01 00:00:00", "calendar": "noleap", "dtype": "float64"},
           "time_bounds": {"units": f"days since {t0}-01-01 00:00:00", "calendar": "noleap", "dtype": "float64"}}
    tmp = out_path.with_suffix(".nc.tmp")
    out.to_netcdf(tmp, encoding=enc, format="NETCDF4_CLASSIC")
    os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ED-ModelC-GFED5",
                    help="Source model under ilamb/MODELS_LEADERBOARD/ providing burntArea.nc")
    ap.add_argument("--d-ref", type=float, default=150.0)
    ap.add_argument("--out-suffix", default="-Hurtt")
    ap.add_argument("--betas-json", default=None,
                    help="Optional JSON with tuned beta_max overrides")
    ap.add_argument("--ba-path", default=None,
                    help="Read burntArea.nc from this path instead of MODELS_LEADERBOARD/<model>/")
    ap.add_argument("--out-dir", default=None,
                    help="Write fFire.nc here instead of MODELS_LEADERBOARD_FFIRE_GFED5/<model><suffix>/")
    args = ap.parse_args()
    if args.betas_json:
        import json as _json
        d = _json.load(open(args.betas_json))["best_params"]
        for k in BETA_MAX:
            key = f"beta_{k}"
            if key in d:
                BETA_MAX[k] = float(d[key])
        if "D_REF" in d:
            args.d_ref = float(d["D_REF"])
        print(f"  loaded tuned BETA_MAX={BETA_MAX}, D_REF={args.d_ref}")

    print(f"Loading inputs for {args.model} (D_REF={args.d_ref}) ...")
    ba, lat, lon, times = load_model_ba(args.model, args.ba_path)
    agb = load_agb_2001_2016()
    csoil = load_csoil_2001_2016(lat, lon)
    dbar = load_dbar_2001_2016()

    print(f"  BA    shape {ba.shape} mean {np.nanmean(ba):.4g} max {np.nanmax(ba):.3g}")
    print(f"  AGB   shape {agb.shape} mean {np.nanmean(agb):.3g} max {np.nanmax(agb):.2f} kgC/m2")
    print(f"  cSoil shape {csoil.shape} mean {np.nanmean(csoil):.3g} max {np.nanmax(csoil):.2f} kgC/m2")
    print(f"  Dbar  shape {dbar.shape} mean {np.nanmean(dbar):.2f} max {np.nanmax(dbar):.1f} mm")

    ffire, c_combust = compute_ffire(ba, agb, csoil, dbar, d_ref=args.d_ref)
    print(f"  C_combust mean {np.nanmean(c_combust):.3f}  95th {np.nanquantile(c_combust, 0.95):.3f} kgC/m2")
    print(f"  fFire     mean {np.nanmean(ffire):.4g}  max {np.nanmax(ffire):.3g} kg/m2/s")
    ba_clean = np.nan_to_num(ba, nan=0.0)

    # Diagnostic: annual global total in PgC/yr
    R = 6.371e6; dlon = np.deg2rad(0.5)
    area_lat = (R**2) * dlon * (np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25)))
    area_2d = np.broadcast_to(np.abs(area_lat)[:, None], (len(lat), len(lon))).astype(np.float64)
    total_PgC_per_yr = float((ba_clean * c_combust * area_2d[None, :, :]).sum()) / 1e12 / 16
    print(f"  global total ≈ {total_PgC_per_yr:.2f} PgC/yr "
          f"(GFED5 in THIS benchmark integrates to ~3.4 PgC/yr with this area method, "
          f"NOT the literature ~2 -- match the reference, not 2.0)")

    out_dir = (Path(args.out_dir) if args.out_dir else
               REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / (args.model + args.out_suffix))
    out_p = out_dir / "fFire.nc"
    write_ffire(out_p, ffire, lat, lon, times, args.model, args.d_ref)
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
