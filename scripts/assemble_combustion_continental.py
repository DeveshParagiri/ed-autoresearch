"""Stitch per-continent combustion betas into ONE fFire field (emissions analogue
of assemble_continental.py). Each cell uses its continent's betas+D_REF; cells in
no fitted region (or where a region's fit did NOT beat the global betas on its own
box) fall back to the GLOBAL continental betas (models/combustion-params-continental/
betas.gfed5.json). Keep-best-per-region, exactly like the BA assembly.

Writes ilamb/MODELS_LEADERBOARD_FFIRE_GFED5/ED-ModelC-continental-percont/fFire.nc
(scoreable with official ILAMB) and prints the internal global four_scores so we can
see the spatial-pattern gain before the official run.

Run:  python scripts/assemble_combustion_continental.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, "scripts")
from tune_combustion_params import load_inputs, fFire_from_betas, FF_SCALE
from tune_combustion_continental import REGION_BOX, region_mask
from compute_emissions import load_model_ba, load_agb_2001_2016, load_csoil_2001_2016, \
    load_dbar_2001_2016, write_ffire, SEC_PER_MONTH
from refit_modelA_multiobj import four_scores

import os
REPO = Path(__file__).resolve().parents[1]
BA_MODEL = os.environ.get("BA_MODEL", "ED-ModelC-continental")
CDIR = REPO / "models" / "combustion-params-continental"
SEASDIR = REPO / "models" / "combustion-params-continental-seas"
# CAND env: "overall" (default) picks per region the beta set with the highest TRUE
# region overall across {global, original, seasonal}. "seas" FORCES the seasonal-aware
# betas per region (keep-best only vs global) to test the seasonal lever's official effect.
CAND = os.environ.get("CAND", "overall")
CAND_DIRS = [CDIR, SEASDIR] if CAND == "overall" else [SEASDIR]
OUT_NAME = os.environ.get("OUT_NAME",
                          "ED-ModelC-continental-percont" if CAND == "overall"
                          else "ED-ModelC-continental-percont-seas")
REGIONS = ["Africa", "S.America", "N.America", "Boreal", "SEAsia", "Australia", "Europe"]


def betas_of(d):
    p = d["best_params"]
    return ({"leaf": p["beta_leaf"], "fine": p["beta_fine"],
             "coarse": p["beta_coarse"], "litter": p["beta_litter"]}, p["D_REF"])


# --- decide which regions to adopt (keep-best on the train objective) -------------
gjson = json.load(open(CDIR / "betas.gfed5.json"))
gbetas, gdref = betas_of(gjson)

ba_t, agb_t, csoil_t, dbar_t, obs_t, lat_t = load_inputs(BA_MODEL)
nlon = ba_t.shape[-1]
lon_t = -179.75 + np.arange(nlon) * 0.5
cos_lat = np.cos(np.deg2rad(lat_t)).astype(np.float32)
land_fire = ((obs_t / FF_SCALE) > 0).any(axis=0) | (ba_t > 0).any(axis=0)


def overall_on(mask, betas, dref):
    w2 = (cos_lat[:, None] * land_fire * mask).astype(np.float32)
    b, r, s, sp = four_scores(fFire_from_betas(ba_t, agb_t, csoil_t, dbar_t, betas, dref), obs_t, w2)
    return (2 * b + 2 * r + s + sp) / 6.0


adopted = {}
for reg in REGIONS:
    tag = reg.replace(".", "")
    m = region_mask(lat_t, lon_t, reg)
    o_glob = overall_on(m, gbetas, gdref)
    best = (o_glob, None, "global")   # (score, (betas,dref), label)
    for cd in CAND_DIRS:
        f = cd / f"betas.{tag}.json"
        if not f.exists():
            continue
        rb, rd = betas_of(json.load(open(f)))
        o = overall_on(m, rb, rd)
        if o > best[0] + 1e-4:
            best = (o, (rb, rd), cd.name)
    if best[1] is None:
        print(f"[keep-global] {reg}: global {o_glob:.4f} best")
    else:
        adopted[reg] = best[1]
        print(f"[adopt] {reg}: {best[2]} {best[0]:.4f} > global {o_glob:.4f} (+{best[0]-o_glob:.4f})")

# --- build the stitched fFire on the full 0.5deg grid -----------------------------
ba, lat, lon, times = load_model_ba(BA_MODEL)
agb = load_agb_2001_2016()
csoil = load_csoil_2001_2016(lat, lon)
dbar = load_dbar_2001_2016()


def ffire_field(betas, dref):
    m_theta = np.clip(dbar / dref, 0.0, 1.0).astype(np.float32)
    C = (0.05 * agb * betas["leaf"] + 0.10 * agb * betas["fine"]
         + 0.85 * agb * betas["coarse"] + 0.15 * csoil * betas["litter"]).astype(np.float32) * m_theta
    return (np.nan_to_num(ba) * C / SEC_PER_MONTH).astype(np.float32)


ffire = ffire_field(gbetas, gdref)   # global betas everywhere
assigned = np.zeros((len(lat), len(lon)), bool)
for reg, (rb, rd) in adopted.items():
    box = region_mask(lat, lon, reg) & ~assigned
    fr = ffire_field(rb, rd)
    ffire[:, box] = fr[:, box]
    assigned |= box
    print(f"[set] {reg}: {int(box.sum())} cells")

out_dir = REPO / "ilamb" / "MODELS_LEADERBOARD_FFIRE_GFED5" / OUT_NAME
write_ffire(out_dir / "fFire.nc", ffire, lat, lon, times, BA_MODEL, gdref)
print(f"wrote {(out_dir / 'fFire.nc').relative_to(REPO).as_posix()}")

# global magnitude diagnostic (PgC/yr)
R = 6.371e6; dlon = np.deg2rad(0.5)
area_lat = (R**2) * dlon * (np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25)))
area_2d = np.abs(area_lat)[:, None]
tot = float((ffire * SEC_PER_MONTH * area_2d[None]).sum()) / 1e12 / (len(times) / 12)
print(f"global total ~ {tot:.2f} PgC/yr (GFED5 3.40)")
