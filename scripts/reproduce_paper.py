#!/usr/bin/env python3
"""End-to-end reproduction of the offline fire paper ladder + figures.

Paper versions (see paper/paper.pdf, Downloads draft is the source of truth):
  Model C  global formula, aggregate ILAMB fit
  Model D  same form, spatial / active-fire objective
  Model E  per-continent params + fuel amplitude + seasonal transform
  fFire E  combustion with per-continent betas on Model E BA

Usage:
  python scripts/reproduce_paper.py              # BA C/D/E + fFire + figures
  python scripts/reproduce_paper.py --skip-ffire # BA + figures only
  python scripts/reproduce_paper.py --figures-only
  python scripts/reproduce_paper.py --holdout    # also held-out year/cell assemblies

Requires:
  - conda env with xarray, numpy, netCDF4, matplotlib, cartopy, cftime, h5py
  - data/crujra/*_monthly.npy
  - global_baseline_modelC_inputs_1997-2016.nc (GPP)
  - global_baseline_modelCfuel_inputs_1997-2016.nc (AGB, for E fuel/fFire)
  - data/gfed/4.1/GFED4.1s_*.hdf5 (land mask via burned cells; GFED5 monthly in data/gfed/5/)
  - ilamb_ref_official/DATA/... GFED5 BA (+ fFire for emissions figures)
  - data/trendy_v14/EDv3_S3_cSoil.nc (for fFire)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd, env=None):
    print("\n==>", " ".join(cmd) if isinstance(cmd, list) else cmd)
    e = os.environ.copy()
    if env:
        e.update(env)
    subprocess.check_call(cmd, cwd=REPO, env=e)


def check_drivers():
    need = [
        REPO / "data/crujra/dbar_monthly.npy",
        REPO / "data/crujra/p_ann_monthly.npy",
        REPO / "data/crujra/p_month_monthly.npy",
        REPO / "data/crujra/t_air_monthly.npy",
        REPO / "global_baseline_modelC_inputs_1997-2016.nc",
        REPO / "models/paper/C.json",
        REPO / "models/paper/D.json",
        REPO / "models/paper/E/assembly.json",
    ]
    missing = [str(p.relative_to(REPO) if p.is_relative_to(REPO) else p) for p in need if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing required inputs:\n  - "
            + "\n  - ".join(missing)
            + "\n\nSee README.md (Data) and scripts/download_inputs.sh."
        )


def main():
    ap = argparse.ArgumentParser(description="Reproduce paper Models C/D/E + figures")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--skip-ffire", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--holdout", action="store_true", help="Also build held-out year/cell E variants")
    args = ap.parse_args()

    if not args.figures_only:
        check_drivers()

        # Model C: aggregate fit, legacy annual->monthly transform
        run(
            [
                PY,
                "scripts/reproduce_modelC.py",
                "--params",
                "models/paper/C.json",
                "--out",
                "ilamb/MODELS/paper/Model-C",
                "--seasonal-transform",
                "0",
                "--title",
                "Model C (global, aggregate ILAMB fit)",
            ]
        )
        # Also write legacy path some diagnostics still expect
        run(
            [
                PY,
                "scripts/reproduce_modelC.py",
                "--params",
                "models/paper/C.json",
                "--out",
                "ilamb/MODELS/ED-ModelC-final",
                "--seasonal-transform",
                "0",
                "--title",
                "Model C (legacy path)",
            ]
        )

        # Model D: spatial objective, seasonal transform (as stored in D.json metadata)
        run(
            [
                PY,
                "scripts/reproduce_modelC.py",
                "--params",
                "models/paper/D.json",
                "--out",
                "ilamb/MODELS/paper/Model-D",
                "--seasonal-transform",
                "1",
                "--title",
                "Model D (spatial / active-fire objective)",
            ]
        )

        # Model E: continental assembly
        run([PY, "scripts/assemble_continental.py"], env={"SEASONAL_TRANSFORM": "1", "ASSEMBLY": "best"})

        if args.holdout:
            run([PY, "scripts/assemble_continental.py"], env={"SEASONAL_TRANSFORM": "1", "ASSEMBLY": "ho"})
            run([PY, "scripts/assemble_continental.py"], env={"SEASONAL_TRANSFORM": "1", "ASSEMBLY": "cell"})
            if (REPO / "scripts/validate_holdout.py").is_file():
                run([PY, "scripts/validate_holdout.py"])
            if (REPO / "scripts/validate_holdout_cells.py").is_file():
                run([PY, "scripts/validate_holdout_cells.py"])

        if not args.skip_ffire:
            fuel = REPO / "global_baseline_modelCfuel_inputs_1997-2016.nc"
            csoil = REPO / "data/trendy_v14/EDv3_S3_cSoil.nc"
            if not fuel.is_file() or not csoil.is_file():
                print(
                    "[warn] skipping fFire: need global_baseline_modelCfuel_inputs_1997-2016.nc "
                    "and data/trendy_v14/EDv3_S3_cSoil.nc"
                )
            else:
                run([PY, "scripts/assemble_combustion_continental.py"], env={"BA_MODEL": "Model-E"})

    if not args.skip_figures:
        run([PY, "scripts/paper_figures.py"])

    print("\nDone.")
    print("  BA:    ilamb/MODELS/paper/Model-{C,D,E}/burntArea.nc")
    print("  fFire: ilamb/MODELS/paper/Model-E-fFire/fFire.nc")
    print("  figs:  figures/paper/")
    print("  draft: paper/paper.pdf")


if __name__ == "__main__":
    main()
