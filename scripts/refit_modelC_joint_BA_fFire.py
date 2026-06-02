"""Joint refit: Model C BA params + Hurtt formula β,D_REF against GFED5 BA and fFire.

Loss is the average of the two ILAMB-style Overalls:
    L = 1 - (Overall_BA + Overall_fFire) / 2

So the optimizer balances burned area against fire carbon emissions equally.

17 parameters total: 12 Model C BA params + 4 betas + D_REF.
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np, optuna, xarray as xr
sys.path.insert(0, "scripts")
from refit_modelC_magaware import predict_monthly, load_inputs as load_modelC_inputs
from refit_modelA_multiobj import four_scores
from tune_hurtt_betas import fFire_from_betas, FF_SCALE, SEC_PER_MONTH

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="global_baseline_modelCfuel_inputs_1997-2016.nc")
    ap.add_argument("--n-trials", type=int, default=8000)
    ap.add_argument("--timeout-h", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "models" / "C-joint-BA-fFire"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print("Joint refit Model C BA + Hurtt fFire, GFED5 targets")

    ds_in = xr.open_dataset(REPO / args.input)
    yr_all = np.array([d.year for d in ds_in["time"].values])
    train_mask = (yr_all >= 2001) & (yr_all <= 2010)
    d_tr = load_modelC_inputs(ds_in.isel(time=train_mask))

    ba_ref = xr.open_dataset(REPO / "ilamb_ref_official" / "DATA" / "burntArea" / "GFED5" / "burntArea.nc")
    yba = np.array([t.year for t in ba_ref["time"].values])
    mba = (yba >= 2001) & (yba <= 2010)
    obs_ba = (ba_ref["burntArea"].values[mba] / 100.0).astype(np.float32)

    ff_ref = xr.open_dataset(REPO / "ilamb_ref_official" / "DATA" / "fFire" / "GFED5" / "fFire.nc")
    yff = np.array([t.year for t in ff_ref["time"].values])
    mff = (yff >= 2001) & (yff <= 2010)
    obs_ff = (ff_ref["fFire"].values[mff].astype(np.float32)) * FF_SCALE

    agb = np.nan_to_num(ds_in["AGB"].values[train_mask].astype(np.float32))

    csoil_ds = xr.open_dataset(REPO / "data" / "trendy_v14" / "EDv3_S3_cSoil.nc")
    yc = np.array([t.astype("datetime64[Y]").astype(int) + 1970 for t in csoil_ds["time"].values])
    mc = (yc >= 2001) & (yc <= 2010)
    annual = csoil_ds["cSoil"].values[mc].astype(np.float32)
    if csoil_ds["latitude"].values[0] > csoil_ds["latitude"].values[-1]:
        annual = annual[:, ::-1, :]
    csoil = np.nan_to_num(np.repeat(annual, 12, axis=0))

    dbar_1 = np.load(REPO / "data" / "crujra" / "dbar_monthly.npy")[:120]
    dbar = np.repeat(np.repeat(dbar_1, 2, axis=-2), 2, axis=-1).astype(np.float32)

    lat = ds_in["lat"].values
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
    land_fire_ba = (obs_ba > 0).any(axis=0)
    land_fire_ff = ((obs_ff / FF_SCALE) > 0).any(axis=0)
    w2_ba = (cos_lat[:, None] * land_fire_ba).astype(np.float32)
    w2_ff = (cos_lat[:, None] * (land_fire_ba | land_fire_ff)).astype(np.float32)

    seed_ba = json.load(open(REPO / "models" / "C-gfed5" / "params.gfed5.json"))["params"]
    seed_betas = json.load(open(REPO / "models" / "hurtt-betas" / "betas.gfed5.json"))["best_params"]

    def evaluate(params):
        ba_pred = predict_monthly(d_tr, params)
        betas = {"leaf": params["beta_leaf"], "fine": params["beta_fine"],
                 "coarse": params["beta_coarse"], "litter": params["beta_litter"]}
        ff_pred = fFire_from_betas(ba_pred, agb, csoil, dbar, betas, params["D_REF"])
        bb, br, bs, bsp = four_scores(ba_pred, obs_ba, w2_ba)
        ba_overall = (2 * bb + 2 * br + bs + bsp) / 6.0
        fb, fr, fs, fsp = four_scores(ff_pred, obs_ff, w2_ff)
        ff_overall = (2 * fb + 2 * fr + fs + fsp) / 6.0
        return ba_overall, ff_overall, (bb, br, bs, bsp), (fb, fr, fs, fsp)

    seed_p = dict(seed_ba)
    for k in ("beta_leaf", "beta_fine", "beta_coarse", "beta_litter", "D_REF"):
        seed_p[k] = seed_betas[k]
    ba_ov, ff_ov, _, _ = evaluate(seed_p)
    print(f"  seed: BA={ba_ov:.4f} fFire={ff_ov:.4f} avg={(ba_ov+ff_ov)/2:.4f}")

    def objective(trial):
        p = dict(
            k1     = trial.suggest_float("k1",     1e-6, 1e-2, log=True),
            D_low  = trial.suggest_float("D_low",  1e2,  1e5,  log=True),
            k2     = trial.suggest_float("k2",     1e-7, 1e-3, log=True),
            D_high = trial.suggest_float("D_high", 1e4,  5e6,  log=True),
            fire_exp = trial.suggest_float("fire_exp", 0.1, 2.0),
            P_half          = trial.suggest_float("P_half",          1e0, 5e3, log=True),
            pre_dampen_half = trial.suggest_float("pre_dampen_half", 1e-1, 1e2, log=True),
            gpp_af = trial.suggest_float("gpp_af", 1e-2, 1e1, log=True),
            gpp_b  = trial.suggest_float("gpp_b",  1e-3, 1e1, log=True),
            gpp_d  = trial.suggest_float("gpp_d",  1e-1, 1e3, log=True),
            ign_k  = trial.suggest_float("ign_k",  1e-2, 1e1, log=True),
            ign_c  = trial.suggest_float("ign_c", -10.0, 35.0),
            beta_leaf   = trial.suggest_float("beta_leaf",   0.0, 1.0),
            beta_fine   = trial.suggest_float("beta_fine",   0.0, 1.0),
            beta_coarse = trial.suggest_float("beta_coarse", 0.0, 1.0),
            beta_litter = trial.suggest_float("beta_litter", 0.0, 1.0),
            D_REF       = trial.suggest_float("D_REF", 100.0, 30000.0, log=True),
        )
        if p["D_high"] <= p["D_low"] * 1.5:
            return 1.0
        ba_ov, ff_ov, _, _ = evaluate(p)
        combined = (ba_ov + ff_ov) / 2.0
        trial.set_user_attr("ba_overall", ba_ov)
        trial.set_user_attr("ff_overall", ff_ov)
        trial.set_user_attr("combined", combined)
        return float(1.0 - combined)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True,
                                          n_startup_trials=300)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(seed_p)

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_h*3600,
                   show_progress_bar=False, gc_after_trial=True)
    print(f"done, {len(study.trials)} trials in {(time.time()-t0)/60:.1f} min")
    best = study.best_trial
    print(f"Best #{best.number}: BA={best.user_attrs['ba_overall']:.4f} "
          f"fFire={best.user_attrs['ff_overall']:.4f} combined={best.user_attrs['combined']:.4f}")
    json.dump({"loss": "1 - (BA_overall + fFire_overall)/2",
               "train_window": [2001, 2010],
               "user_attrs": dict(best.user_attrs),
               "params": best.params},
              open(out_dir / "params.joint.json", "w"), indent=2)
    print(f"wrote {out_dir}/params.joint.json")


if __name__ == "__main__":
    main()
