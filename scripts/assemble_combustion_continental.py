"""Stitch per-continent combustion betas into one fFire field for paper Model E.

Reads betas from models/combustion/continental/ (legacy path also accepted).
Reads BA from paper Model-E (or legacy ED-ModelC-continental).
Writes:
  ilamb/MODELS/paper/Model-E-fFire/fFire.nc
  ilamb/MODELS_LEADERBOARD_FFIRE_GFED5/ED-ModelC-continental-percont/fFire.nc

Run:  python scripts/assemble_combustion_continental.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_emissions import (  # noqa: E402
    SEC_PER_MONTH,
    load_agb_2001_2016,
    load_csoil_2001_2016,
    load_dbar_2001_2016,
    load_model_ba,
    write_ffire,
)
from scores import four_scores  # noqa: E402
from tune_combustion_continental import REGION_BOX, region_mask  # noqa: E402
from tune_combustion_params import FF_SCALE, fFire_from_betas, load_inputs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BA_MODEL = os.environ.get("BA_MODEL", "Model-E")
CDIR_CANDIDATES = [
    REPO / "models" / "combustion" / "continental",
    REPO / "models" / "combustion-params-continental",
]
SEASDIR = REPO / "models" / "combustion-params-continental-seas"
CAND = os.environ.get("CAND", "overall")
OUT_NAME = os.environ.get(
    "OUT_NAME",
    "Model-E-fFire" if CAND == "overall" else "Model-E-fFire-seas",
)
REGIONS = ["Africa", "S.America", "N.America", "Boreal", "SEAsia", "Australia", "Europe"]


def find_cdir() -> Path:
    for d in CDIR_CANDIDATES:
        if (d / "betas.gfed5.json").is_file():
            return d
    raise FileNotFoundError("models/combustion/continental/betas.gfed5.json not found")


def betas_of(d):
    p = d["best_params"]
    return (
        {
            "leaf": p["beta_leaf"],
            "fine": p["beta_fine"],
            "coarse": p["beta_coarse"],
            "litter": p["beta_litter"],
        },
        p["D_REF"],
    )


def main():
    cdir = find_cdir()
    cand_dirs = [cdir]
    if CAND == "overall" and SEASDIR.is_dir():
        cand_dirs.append(SEASDIR)
    elif CAND == "seas":
        cand_dirs = [SEASDIR]

    gjson = json.load(open(cdir / "betas.gfed5.json"))
    gbetas, gdref = betas_of(gjson)

    # Prefer Model-E; fall back to legacy name inside load_inputs via resolve_ba_path
    ba_name = BA_MODEL
    try:
        ba_t, agb_t, csoil_t, dbar_t, obs_t, lat_t = load_inputs(ba_name)
    except FileNotFoundError:
        ba_name = "ED-ModelC-continental"
        ba_t, agb_t, csoil_t, dbar_t, obs_t, lat_t = load_inputs(ba_name)

    nlon = ba_t.shape[-1]
    lon_t = -179.75 + np.arange(nlon) * 0.5
    cos_lat = np.cos(np.deg2rad(lat_t)).astype(np.float32)
    land_fire = ((obs_t / FF_SCALE) > 0).any(axis=0) | (ba_t > 0).any(axis=0)

    def overall_on(mask, betas, dref):
        w2 = (cos_lat[:, None] * land_fire * mask).astype(np.float32)
        b, r, s, sp = four_scores(
            fFire_from_betas(ba_t, agb_t, csoil_t, dbar_t, betas, dref), obs_t, w2
        )
        return (2 * b + 2 * r + s + sp) / 6.0

    adopted = {}
    for reg in REGIONS:
        tag = reg.replace(".", "")
        m = region_mask(lat_t, lon_t, reg)
        o_glob = overall_on(m, gbetas, gdref)
        best = (o_glob, None, "global")
        for cd in cand_dirs:
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
            print(
                f"[adopt] {reg}: {best[2]} {best[0]:.4f} > global {o_glob:.4f} "
                f"(+{best[0] - o_glob:.4f})"
            )

    ba, lat, lon, times = load_model_ba(ba_name)
    agb = load_agb_2001_2016()
    csoil = load_csoil_2001_2016(lat, lon)
    dbar = load_dbar_2001_2016()

    def ffire_field(betas, dref):
        m_theta = np.clip(dbar / dref, 0.0, 1.0).astype(np.float32)
        C = (
            0.05 * agb * betas["leaf"]
            + 0.10 * agb * betas["fine"]
            + 0.85 * agb * betas["coarse"]
            + 0.15 * csoil * betas["litter"]
        ).astype(np.float32) * m_theta
        return (np.nan_to_num(ba) * C / SEC_PER_MONTH).astype(np.float32)

    ffire = ffire_field(gbetas, gdref)
    assigned = np.zeros((len(lat), len(lon)), bool)
    for reg, (rb, rd) in adopted.items():
        box = region_mask(lat, lon, reg) & ~assigned
        fr = ffire_field(rb, rd)
        ffire[:, box] = fr[:, box]
        assigned |= box
        print(f"[set] {reg}: {int(box.sum())} cells")

    out_dirs = [
        REPO / "ilamb" / "MODELS" / "paper" / OUT_NAME,
        REPO
        / "ilamb"
        / "MODELS_LEADERBOARD_FFIRE_GFED5"
        / "ED-ModelC-continental-percont",
    ]
    for out_dir in out_dirs:
        write_ffire(out_dir / "fFire.nc", ffire, lat, lon, times, ba_name, gdref)
        print(f"wrote {(out_dir / 'fFire.nc').relative_to(REPO).as_posix()}")

    R = 6.371e6
    dlon = np.deg2rad(0.5)
    area_lat = (R**2) * dlon * (
        np.sin(np.deg2rad(lat + 0.25)) - np.sin(np.deg2rad(lat - 0.25))
    )
    area_2d = np.abs(area_lat)[:, None]
    tot = float((ffire * SEC_PER_MONTH * area_2d[None]).sum()) / 1e12 / (len(times) / 12)
    print(f"global total ~ {tot:.2f} PgC/yr (GFED5 3.40)")


if __name__ == "__main__":
    main()
