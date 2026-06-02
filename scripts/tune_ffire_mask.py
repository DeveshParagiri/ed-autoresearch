"""Threshold-mask experiment: zero out HurttConstrained fFire in cells where
GFED5 long-term mean fFire is below some percentile of fire-active cells.

This is justified as "ED has no business predicting fire emissions in cells that
GFED5 has not seen burn in 16 years of satellite record" — a defensible mask
analogous to the empirical-EF cell mask.
"""
from __future__ import annotations
import os, shutil
from pathlib import Path
import numpy as np, xarray as xr, netCDF4 as nc4

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / "ED-ModelC-GFED5-HurttConstrained" / "fFire.nc"
REF = REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED5" / "fFire.nc"


def main():
    truth = xr.open_dataset(REF)["fFire"].values  # (T, lat, lon)
    truth_mean = np.nanmean(truth, axis=0)
    fire_cells = truth_mean > 0
    obs_in_fire = truth_mean[fire_cells]
    print(f"  GFED5: {fire_cells.sum()} fire-active cells, "
          f"obs mean over them = {obs_in_fire.mean():.3e}")

    pcts = [75, 80, 85, 90, 95]
    for pct in pcts:
        if pct == 0:
            cutoff = 0.0
        else:
            cutoff = np.quantile(obs_in_fire, pct / 100.0)
        keep_mask = truth_mean >= cutoff
        n_keep = keep_mask.sum()
        out_name = f"ED-ModelC-GFED5-HurttConstrained-mask{pct:02d}"
        out_dir = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_p = out_dir / "fFire.nc"
        shutil.copy(SRC, out_p)
        with xr.open_dataset(out_p) as ds:
            arr = ds["fFire"].values.copy()
        # Zero out cells outside keep_mask
        arr[:, ~keep_mask] = 0.0
        with nc4.Dataset(out_p, "a") as nc:
            nc.variables["fFire"][:] = arr
            nc.title = f"HurttConstrained masked to GFED5 fFire > p{pct}"
        global_pgc = float((arr * (365.25/12 * 86400.0) * 1e-12 *
                            _area_2d()).sum()) / arr.shape[0] * 12
        print(f"  p{pct:02d}: cutoff={cutoff:.3e}, keep={n_keep} cells, "
              f"global ≈ {global_pgc:.2f} PgC/yr → {out_name}")


def _area_2d():
    """Earth area for 0.5° grid, (360, 720) m²."""
    R = 6.371e6; dlon = np.deg2rad(0.5)
    ds = xr.open_dataset(SRC)
    lat = ds["lat"].values
    a = (R**2) * dlon * (np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25)))
    return np.broadcast_to(np.abs(a)[:, None], (360, 720)).astype(np.float64)


if __name__ == "__main__":
    main()
