"""Per-continent combustion-beta tuner (the emissions analogue of workstream C).

Same idea that fixed BA: the continental BA redistributed fire onto low-biomass
savanna and cut the Amazon over-burn, so a SINGLE global beta set no longer maps
that BA onto GFED5's fFire pattern well (global betas give internal spatial 0.517
on the continental BA vs 0.670 on the old Hybrid BA). This fits beta_{leaf,fine,
coarse,litter} + D_REF SEPARATELY per continent, restricting the four_scores
objective to that continent's box (cos-lat weighted, fire-active cells), exactly
like optimize_modelC_coupled.py's REGION lever does for BA.

Writes models/combustion/continental/betas.<region>.json. The assembler
(assemble_combustion_continental.py) stitches them into one fFire field and only
keeps a region's betas if they beat the global betas on that region (keep-best).

Run:  REGION=Africa python scripts/tune_combustion_continental.py
      REGION="" ...  -> global (reproduces tune_combustion_params.py on this BA)
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np, optuna
import sys
sys.path.insert(0, "scripts")
from tune_combustion_params import load_inputs, fFire_from_betas, FF_SCALE
from scores import four_scores

REPO = Path(__file__).resolve().parents[1]

# Continent boxes (lon0, lon1, lat0, lat1) — MUST match assemble_continental.py.
REGION_BOX = {
    "Africa": (-20, 52, -36, 18), "S.America": (-82, -34, -56, 14),
    "N.America": (-168, -52, 14, 74), "Boreal": (40, 180, 48, 78),
    "SEAsia": (60, 150, -11, 30), "Australia": (112, 154, -44, -10),
    "Europe": (-12, 40, 36, 72),
}


def region_mask(lat, lon, region):
    if not region:
        return np.ones((len(lat), len(lon)), bool)
    b = REGION_BOX[region]
    LON, LAT = np.meshgrid(lon, lat)
    return (LON >= b[0]) & (LON <= b[1]) & (LAT >= b[2]) & (LAT <= b[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ba-model", default="Model-E")
    ap.add_argument("--region", default=os.environ.get("REGION", ""))
    ap.add_argument("--n-trials", type=int, default=int(os.environ.get("N_TRIALS", 4000)))
    ap.add_argument("--timeout-h", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "combustion" / "continental"))
    ap.add_argument("--seas-w", type=float, default=float(os.environ.get("SEAS_W", 0.0)),
                    help="Blend weight on the seasonal-cycle score: objective = "
                         "(1-w)*overall + w*seas. 0 = pure overall (original).")
    args = ap.parse_args()
    region = args.region
    seas_w = args.seas_w
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading inputs (BA={args.ba_model}, region={region or 'GLOBAL'}, train 2001-2010) ...")
    ba, agb, csoil, dbar, obs, lat = load_inputs(args.ba_model)
    nlon = ba.shape[-1]
    lon = -179.75 + np.arange(nlon) * 0.5  # 0.5deg grid centers
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire = ((obs / FF_SCALE) > 0).any(axis=0) | (ba > 0).any(axis=0)
    rmask = region_mask(lat, lon, region)
    w2 = (cos_lat[:, None] * land_fire * rmask).astype(np.float32)
    print(f"  active cells in objective: {int((w2 > 0).sum())}")

    def overall_of(betas, d_ref):
        pred = fFire_from_betas(ba, agb, csoil, dbar, betas, d_ref)
        b, r, s, sp = four_scores(pred, obs, w2)
        return (2 * b + 2 * r + s + sp) / 6.0, (b, r, s, sp)

    # Seed from the existing GLOBAL continental betas (warm start).
    seed_path = out_dir / "betas.gfed5.json"
    if not seed_path.is_file():
        seed_path = REPO / "models" / "combustion" / "continental" / "betas.gfed5.json"
    gp = json.load(open(seed_path))["best_params"]
    seed_betas = {"leaf": gp["beta_leaf"], "fine": gp["beta_fine"],
                  "coarse": gp["beta_coarse"], "litter": gp["beta_litter"]}
    seed_overall, sc = overall_of(seed_betas, gp["D_REF"])
    print(f"  global-betas baseline on this mask: overall={seed_overall:.4f} "
          f"(b={sc[0]:.3f} r={sc[1]:.3f} s={sc[2]:.3f} sp={sc[3]:.3f})")

    def objective(trial):
        betas = {
            "leaf":   trial.suggest_float("beta_leaf",   0.0, 1.0),
            "fine":   trial.suggest_float("beta_fine",   0.0, 1.0),
            "coarse": trial.suggest_float("beta_coarse", 0.0, 1.0),
            "litter": trial.suggest_float("beta_litter", 0.0, 1.0),
        }
        d_ref = trial.suggest_float("D_REF", 100.0, 30000.0, log=True)
        overall, (b, r, s, sp) = overall_of(betas, d_ref)
        for k, v in zip(("bias", "rmse", "seas", "spat", "overall"), (b, r, s, sp, overall)):
            trial.set_user_attr(k, v)
        blended = (1.0 - seas_w) * overall + seas_w * s
        return float(1.0 - blended)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True,
                                         n_startup_trials=200)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial({"beta_leaf": seed_betas["leaf"], "beta_fine": seed_betas["fine"],
                         "beta_coarse": seed_betas["coarse"], "beta_litter": seed_betas["litter"],
                         "D_REF": gp["D_REF"]})

    print(f"\nTPE n_trials={args.n_trials} timeout={args.timeout_h}h")
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_h * 3600,
                   show_progress_bar=False, gc_after_trial=True)
    best = study.best_trial
    print(f"\ndone, {len(study.trials)} trials in {(time.time()-t0)/60:.1f} min")
    print(f"Best #{best.number}: overall={best.user_attrs['overall']:.4f} "
          f"(b={best.user_attrs['bias']:.3f} r={best.user_attrs['rmse']:.3f} "
          f"s={best.user_attrs['seas']:.3f} sp={best.user_attrs['spat']:.3f})")
    print(f"  vs global-betas baseline {seed_overall:.4f}  "
          f"(delta {best.user_attrs['overall']-seed_overall:+.4f})")
    print(f"  params: {best.params}")

    tag = (region or "global").replace(".", "")
    json.dump({"ba_model": args.ba_model, "region": region,
               "baseline_overall_globalbetas": float(seed_overall),
               "best_overall": best.user_attrs["overall"], "best_params": best.params,
               "scores": {k: best.user_attrs[k] for k in ("bias", "rmse", "seas", "spat", "overall")}},
              open(out_dir / f"betas.{tag}.json", "w"), indent=2)
    print(f"wrote {out_dir / f'betas.{tag}.json'}")


if __name__ == "__main__":
    main()
