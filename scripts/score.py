"""Score the three models against GFED4.1s monthly burned area.

Uses the same per-cell scoring formulas as ILAMB's ConfBurntArea:
  Bias Score     = exp(-|pred_mean - obs_mean| / |obs_mean|)
  RMSE Score     = exp(-rmse / obs_std)
  Seasonal Score = (1 + per-cell 12-month correlation) / 2
  Spatial Score  = Taylor: (1 + ρ)/2 × min(σ_ratio, 1/σ_ratio)
  Overall        = mean of the 4

This is NOT a full ilamb-run; it's a standalone scorer so you can evaluate
without installing ILAMB. Matches the 4-metric arithmetic.

Usage:
    python scripts/score.py            # score all 3 models
    python scripts/score.py A          # score just one
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out"
N_MONTHS = 12 * 16


def load_gfed_monthly_1deg():
    """Stub: load GFED4.1s monthly burned fraction at 1-deg."""
    # User must implement; path depends on their GFED4.1s setup.
    # Expected shape: (192, 180, 360) float32, 0.0-1.0 fraction.
    import h5py
    out = np.zeros((192, 180, 360), dtype=np.float32)
    idx = 0
    for yr in range(2001, 2017):
        fp = REPO / f"data/gfed/GFED4.1s_{yr}.hdf5"
        if not fp.exists():
            raise FileNotFoundError(f"Expected {fp}")
        with h5py.File(fp, "r") as f:
            for m in range(1, 13):
                arr = f[f"burned_area/{m:02d}/burned_fraction"][:]  # (720, 1440) N->S
                arr = arr[::-1, :]  # flip to S->N
                arr_1 = arr.reshape(180, 4, 360, 4).mean(axis=(1, 3))
                out[idx] = np.nan_to_num(arr_1, nan=0.0).astype(np.float32)
                idx += 1
    return out


def score_model(name):
    mp = OUT / f"ED-Model{name}" / "burntArea.nc"
    if not mp.exists():
        print(f"Model {name}: missing {mp}")
        return None

    ds = xr.open_dataset(mp, decode_times=False)
    pred = ds["burntArea"].values  # (192, 360, 720)
    # coarsen to 1-deg to match GFED
    pred_1 = pred.reshape(192, 180, 2, 360, 2).mean(axis=(2, 4)).astype(np.float32)

    obs = load_gfed_monthly_1deg()

    # cos-lat weights
    lat = np.arange(-89.5, 90.0, 1.0)
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    w2 = np.broadcast_to(cos_lat[:, None], obs.shape[1:])
    w3 = np.broadcast_to(cos_lat[None, :, None], obs.shape)

    obs_mean = float((obs * w3).sum() / w3.sum())
    pred_mean = float((pred_1 * w3).sum() / w3.sum())

    # Bias
    obs_tm = obs.mean(axis=0)
    pred_tm = pred_1.mean(axis=0)
    rel = np.abs(pred_tm - obs_tm) / (np.abs(obs_tm) + 1e-9)
    bias = float((np.exp(-rel) * w2).sum() / w2.sum())

    # RMSE
    obs_std = obs.std(axis=0)
    rmse_field = np.sqrt(((pred_1 - obs) ** 2).mean(axis=0))
    rmse = float((np.exp(-rmse_field / (obs_std + 1e-9)) * w2).sum() / w2.sum())

    # Seasonal: per-cell 12-month correlation
    obs_cycle = obs.reshape(16, 12, 180, 360).mean(axis=0)
    pred_cycle = pred_1.reshape(16, 12, 180, 360).mean(axis=0)
    obs_anom = obs_cycle - obs_cycle.mean(axis=0, keepdims=True)
    pred_anom = pred_cycle - pred_cycle.mean(axis=0, keepdims=True)
    num = (pred_anom * obs_anom).sum(axis=0)
    den = np.sqrt((pred_anom ** 2).sum(axis=0) * (obs_anom ** 2).sum(axis=0)) + 1e-9
    corr = np.clip(num / den, -1.0, 1.0)
    seas = float(((1 + corr) / 2 * w2).sum() / w2.sum())

    # Spatial distribution (Taylor)
    pw = w2.flatten()
    pp = (pred_tm - pred_mean).flatten()
    oo = (obs_tm - obs_mean).flatten()
    rho = float(((pp * oo) * pw).sum() / (np.sqrt(((pp ** 2) * pw).sum() * ((oo ** 2) * pw).sum()) + 1e-9))
    sp_ratio = float(np.sqrt(((pp ** 2) * pw).sum() / pw.sum()) / (np.sqrt(((oo ** 2) * pw).sum() / pw.sum()) + 1e-9))
    spatial = float(((1 + max(rho, 0.0)) / 2) * min(sp_ratio, 1.0 / sp_ratio))

    overall = np.mean([bias, rmse, seas, spatial])
    return {"bias": bias, "rmse": rmse, "seasonal": seas, "spatial": spatial, "overall": float(overall)}


def main():
    if len(sys.argv) > 1:
        names = [sys.argv[1].upper()]
    else:
        names = ["A", "B", "C"]

    for name in names:
        r = score_model(name)
        if r is None:
            continue
        print(f"Model {name}:  Bias={r['bias']:.3f}  RMSE={r['rmse']:.3f}  "
              f"Seas={r['seasonal']:.3f}  Spatial={r['spatial']:.3f}  "
              f"Overall={r['overall']:.3f}")


if __name__ == "__main__":
    main()
