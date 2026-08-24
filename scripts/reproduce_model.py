"""Rebuild ANY fitted version of the fire model from its params file.

    python scripts/reproduce_model.py --params models/C/params.H.json --out ilamb/MODELS_REBUILD/H
    python scripts/reproduce_model.py --params models/C/params.json   --out /tmp/C --check

WHY THIS EXISTS, AND WHY reproduce_modelC.py IS NOT ENOUGH. That script hardcodes its input
(models/C/params.json), hardcodes its output (the canonical ilamb/MODELS/ED-ModelC-final/ folder,
which is also where stale .nc files break ILAMB with MonotonicityError), and implements Model C and
nothing else. It has no GDP term, no population term, no land use, no curing. Point it at Model H's
parameters and it reads gdp_gamma out of the file, prints it in its own log line, and never uses it,
producing Model C driven by H's numbers. That scores 0.5582 against H's true 0.6819 with no error
raised. Dev did exactly this on 2026-08-22 and reported our recorded scores as wrong. They were not.
The model simply could not be rebuilt.

This script takes both paths as arguments and reads the mechanisms from the params file's
`environment` stamp, so a params file describes a model rather than a point in parameter space. All
the model code is imported from model_mechanisms and reproduce_modelC, never re-implemented here, so
a rebuild cannot drift away from what the optimizer fitted.

IT REFUSES TO RUN ON AN UNSTAMPED FILE unless you pass --env-from or --assume-model-c. Guessing
would reintroduce precisely the silent failure this exists to stop. Files fitted before 2026-08-24
have no stamp, so use --env-from to name the run script or point it at another stamped params file:

    python scripts/reproduce_model.py --params models/C/params.H.json \
        --out ilamb/MODELS_REBUILD/H --env-from GDP_TERM=1,TROP_MASK=0

--check scores the rebuild with the internal ILAMB-aligned scorer and compares it against the
scores_internal block in the params file. That catches a wrong environment in about a minute,
without waiting for an official ILAMB run. It is not a substitute for official ILAMB, which remains
the number that goes in the paper.

Run with the edfire env.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_modelC import fire_C, add_cf_bounds, uncoarsen, coarsen, sig, trailing_mean
import model_mechanisms as mech

REPO = Path(__file__).resolve().parents[1]


def parse_env_from(spec):
    """FLAG=value,FLAG=value -> a mechanism environment, defaults for everything unnamed."""
    env = {name: default for name, (_, default) in mech.MECH_FLAGS.items()}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--env-from item {item!r} is not FLAG=value")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in mech.MECH_FLAGS:
            raise SystemExit(f"--env-from names {k!r}, which is not a mechanism flag. "
                             f"Known flags: {', '.join(mech.MECH_FLAGS)}")
        env[k] = mech._PARSE[mech.MECH_FLAGS[k][0]](v.strip())
    return env


def score_internal(pred, obs, land_mask, n_years):
    """The optimizer's ILAMB-aligned scorer, enough to verify a rebuild.

    Transcribed from optimize_modelC_coupled.score_BA, which cannot be imported because that module
    runs an optuna study at import time. This is a CHECK, not the model, so a copy here is not the
    drift risk the mechanism code was. Official ILAMB remains the number of record.
    """
    lat_1 = np.arange(-89.5, 90.0, 1.0).astype(np.float32)
    cos_lat = np.cos(np.deg2rad(lat_1)).astype(np.float32)
    w2 = np.broadcast_to(cos_lat[:, None], (180, 360)).astype(np.float64)

    gfed_tm = obs.mean(axis=0).astype(np.float64)
    gfed_std = obs.std(axis=0).clip(1e-12).astype(np.float64)
    mass_w = (w2 * gfed_tm).astype(np.float64)
    mass_w_burn = (mass_w * land_mask).astype(np.float64)
    w2_burn = (w2 * land_mask).astype(np.float64)
    obs_anom = (obs - gfed_tm[None, :, :]).astype(np.float64)

    pred_tm = pred.mean(axis=0).astype(np.float64)
    bias_s = np.exp(-np.abs(pred_tm - gfed_tm) / gfed_std)
    bias = float((bias_s * mass_w).sum() / (mass_w.sum() + 1e-12))

    pred_anom = pred.astype(np.float64) - pred_tm[None, :, :]
    crmse = np.sqrt(((pred_anom - obs_anom) ** 2).mean(axis=0))
    rmse = float((np.exp(-crmse / gfed_std) * mass_w).sum() / (mass_w.sum() + 1e-12))

    gfed_cyc = obs.reshape(n_years, 12, 180, 360).mean(axis=0)
    gfed_peak = np.argmax(gfed_cyc, axis=0).astype(np.float32)
    pred_cyc = pred.reshape(n_years, 12, 180, 360).mean(axis=0)
    pred_peak = np.argmax(pred_cyc, axis=0).astype(np.float32)
    shift = pred_peak - gfed_peak
    shift = np.where(shift > 6, shift - 12, shift)
    shift = np.where(shift < -6, shift + 12, shift)
    seas_c = (1.0 + np.cos(np.abs(shift) / 12.0 * 2.0 * np.pi)) * 0.5
    seas = float((seas_c * mass_w_burn).sum() / (mass_w_burn.sum() + 1e-12))

    m = land_mask
    of, pf, pw = gfed_tm[m], pred_tm[m], w2_burn[m]
    if pw.sum() > 0:
        oa = of - (of * pw).sum() / pw.sum()
        pa = pf - (pf * pw).sum() / pw.sum()
        std0 = max(float(np.sqrt(((oa ** 2) * pw).sum() / pw.sum())), 1e-12)
        std = max(float(np.sqrt(((pa ** 2) * pw).sum() / pw.sum())), 1e-12)
        denom = float(np.sqrt(((pa ** 2) * pw).sum() * ((oa ** 2) * pw).sum()))
        rho = float((pa * oa * pw).sum() / (denom + 1e-12))
        sigma = std / std0
        spatial = 2.0 * (1.0 + rho) / ((sigma + 1.0 / max(sigma, 1e-12)) ** 2)
    else:
        spatial = 0.0

    overall = float((bias + 2.0 * rmse + seas + spatial) / 5.0)
    return dict(bias=bias, rmse=rmse, seas=seas, spatial=float(spatial), overall=overall)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", required=True, help="path to a params JSON")
    ap.add_argument("--out", required=True,
                    help="directory to write burntArea.nc into (an ILAMB model folder)")
    ap.add_argument("--env-from", default=None,
                    help="mechanism flags for an unstamped file, e.g. GDP_TERM=1,TROP_MASK=0")
    ap.add_argument("--assume-model-c", action="store_true",
                    help="treat an unstamped file as plain Model C, no mechanism terms")
    ap.add_argument("--check", action="store_true",
                    help="score the rebuild internally and compare against scores_internal")
    a = ap.parse_args()

    pj_path = Path(a.params)
    if not pj_path.is_absolute():
        pj_path = REPO / pj_path
    pj = json.loads(pj_path.read_text())
    params = pj["params"]

    env = mech.env_of(pj)
    source = "the params file's own environment stamp"
    if a.env_from is not None:
        env = parse_env_from(a.env_from)
        source = "--env-from on the command line"
        if mech.env_of(pj) is not None:
            print("[warn] this file HAS a stamp and --env-from is overriding it")
    elif env is None and a.assume_model_c:
        env = {name: d for name, (_, d) in mech.MECH_FLAGS.items()}
        source = "--assume-model-c"
    elif env is None:
        raise SystemExit(
            f"{pj_path.name} has no `environment` stamp, so which model it is cannot be known.\n"
            f"It was fitted before the stamp existed. Rebuilding it as plain Model C would drop any\n"
            f"mechanism term it needs and score it wrongly, silently, which is the exact failure\n"
            f"this script exists to prevent (see the module docstring).\n\n"
            f"Recover the flags from its run script or optimizer log and pass them, e.g.\n"
            f"  --env-from GDP_TERM=1,TROP_MASK=0\n"
            f"or, if you are certain it is plain Model C, pass --assume-model-c.")

    print(f"[env] from {source}")
    print(f"[env] {mech.describe(env)}")

    drivers = mech.load_drivers(env, coarsen, trailing_mean)
    obs = mech.load_gfed5_1deg(coarsen)

    y0, yf = env["FIT_Y0"], env["FIT_YF"]
    n_years = yf - y0 + 1
    if (y0, yf) != (2001, 2016):
        m0, m1 = (y0 - 2001) * 12, (yf - 2001 + 1) * 12
        obs = obs[m0:m1]
        for k in list(drivers):
            drivers[k] = drivers[k][m0:m1]
        print(f"[setup] fit window {y0}-{yf} ({m1-m0} months)")

    land_mask = (obs > 0).any(axis=0)
    print(f"[setup] land cells (GFED5 burnable): {int(land_mask.sum())} / {land_mask.size}")

    gdp_mult = mech.make_gdp_mult(land_mask, env["GDP_MLO"], env["GDP_MHI"]) if env["GDP_TERM"] else None
    pop_mult = mech.make_pop_mult() if env["POP_TERM"] else None
    landuse_mult = mech.make_landuse_mult(drivers) if env["LANDUSE_TERM"] else None
    curing_term = mech.make_curing_term(drivers, sig) if env["CURING"] else None

    fc = {k: v for k, v in params.items() if k not in mech.MECH_KEYS}
    with np.errstate(over="ignore", invalid="ignore"):
        rate = fire_C(drivers, fc)
    rate = mech.apply_mechanisms(rate, params, env, land_mask, gdp_mult=gdp_mult,
                                 pop_mult=pop_mult, landuse_mult=landuse_mult,
                                 curing_term=curing_term)
    pred = mech.ed_transform(rate, env["FIRE_MAX_RATE"], env["SEASONAL_TRANSFORM"])

    out_dir = Path(a.out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    path = mech.write_ba_nc(pred, out_dir / "burntArea.nc", land_mask, env, uncoarsen,
                            add_cf_bounds, list(range(y0, yf + 1)))
    print(f"[write] {path}  ({path.stat().st_size/1e6:.1f} MB)")

    # area-weighted annual total, the physical magnitude. The unweighted mean-of-percent
    # over-counts small high-latitude cells and reads about six times too high.
    lat = np.arange(-89.5, 90.0, 1.0)
    cell_mha = (111.32 ** 2) * np.cos(np.deg2rad(lat))[:, None] * 1e-4   # km2 -> ha -> Mha
    annual = pred.reshape(n_years, 12, 180, 360).sum(axis=1).mean(axis=0)
    print(f"[diag] area-weighted burned area: {float((annual * cell_mha).sum()):.0f} Mha/yr "
          f"(GFED5 793)")

    if a.check:
        got = score_internal(pred, obs, land_mask, n_years)
        want = pj.get("scores_internal")
        print(f"[check] rebuilt internal Overall = {got['overall']:.4f}  "
              f"(bias {got['bias']:.4f} rmse {got['rmse']:.4f} "
              f"seas {got['seas']:.4f} spatial {got['spatial']:.4f})")
        if want:
            d = got["overall"] - want["overall"]
            verdict = "MATCH" if abs(d) < 5e-4 else "MISMATCH"
            print(f"[check] recorded internal Overall = {want['overall']:.4f}   "
                  f"difference {d:+.4f}  {verdict}")
            if verdict == "MISMATCH":
                print("[check] the environment is wrong, or the params file and its scores "
                      "disagree. Do NOT score this rebuild and report it as the model.")
                return 1
        else:
            print("[check] this params file records no scores_internal, nothing to compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
