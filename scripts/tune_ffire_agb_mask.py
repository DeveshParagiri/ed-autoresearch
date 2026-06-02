"""AGB-based threshold mask. Zero out HurttConstrained fFire in cells where
the long-term mean AGB is below a threshold.

This is more defensible than the truth-based mask because it uses ED's own
biomass field, not GFED5 ground truth. Physical justification: ED has no fuel
to burn in cells where AGB is essentially zero, so emissions in those cells
are an artifact of the BA model painting fire on bare ground.
"""
from __future__ import annotations
import os, shutil
from pathlib import Path
import numpy as np, xarray as xr, netCDF4 as nc4

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / "ED-ModelC-GFED5-HurttConstrained" / "fFire.nc"
FUEL = REPO / "global_baseline_modelCfuel_inputs_1997-2016.nc"


def main():
    ds_fuel = xr.open_dataset(FUEL)
    yr = np.array([t.year for t in ds_fuel["time"].values])
    m = (yr >= 2001) & (yr <= 2016)
    agb_mean = np.nanmean(ds_fuel["AGB"].values[m], axis=0)  # (360, 720) kgC/m²
    has_agb = agb_mean > 0
    obs_in_agb = agb_mean[has_agb]
    print(f"  AGB: {has_agb.sum()} cells with AGB>0, mean = {obs_in_agb.mean():.2f} kgC/m²")
    for p in (10, 25, 50, 75):
        print(f"  AGB p{p}: {np.quantile(obs_in_agb, p/100):.3f} kgC/m²")

    thresholds = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
    for thr in thresholds:
        keep = agb_mean >= thr
        n_keep = keep.sum()
        out_name = f"ED-ModelC-GFED5-HurttConstrained-agb{int(thr*100):03d}"
        out_dir = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_p = out_dir / "fFire.nc"
        shutil.copy(SRC, out_p)
        with xr.open_dataset(out_p) as ds:
            arr = ds["fFire"].values.copy()
        arr[:, ~keep] = 0.0
        with nc4.Dataset(out_p, "a") as nc:
            nc.variables["fFire"][:] = arr
            nc.title = f"HurttConstrained masked to AGB>={thr} kgC/m²"
        print(f"  AGB>={thr:.2f}: {n_keep} cells kept → {out_name}")


if __name__ == "__main__":
    main()
