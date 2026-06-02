"""Apply a uniform scalar to HurttConstrained fFire and find the best multiplier."""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np, xarray as xr, cftime
sys.path.insert(0, "scripts")

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / "ED-ModelC-GFED5-HurttConstrained" / "fFire.nc"


def write_scaled(scalar, out_name):
    import shutil
    out_dir = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "fFire.nc"
    shutil.copy(SRC, out_p)
    with xr.open_dataset(out_p) as ds:
        scaled = (ds["fFire"].values * scalar).astype(np.float32)
    # Reopen in append mode and overwrite variable
    import netCDF4 as nc4
    with nc4.Dataset(out_p, "a") as nc:
        nc.variables["fFire"][:] = scaled
        nc.title = f"HurttConstrained × {scalar:.4f}"
    return out_p


if __name__ == "__main__":
    scalars = [0.50, 0.55, 0.60, 0.625, 0.65, 0.70, 0.80]
    for s in scalars:
        name = f"ED-ModelC-GFED5-HurttConstrained-x{int(s*1000):03d}"
        p = write_scaled(s, name)
        print(f"wrote {p}")
