#!/usr/bin/env python3
"""Replay the historical ED-Fire model ladder without admitting it to research.

This script reconstructs the archived A/B/C GFED4.1s study and the C-I GFED5
study. Those studies read benchmark-derived masks, and Model F additionally
pins its global magnitude to GFED5. Their outputs are provenance evidence, not
valid candidates for the active autoresearch contract.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cftime
import h5py
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
PARAM_ABC = ROOT / "model" / "other-models" / "parameters" / "abc-gfed4.1s"
PARAM_GF5 = ROOT / "model" / "other-models" / "parameters" / "coupled-gfed5"
DEFAULT_OUTPUT = ROOT / "model" / "other-models" / "reproduced"
CONTRACT = ROOT / "evals" / "contracts" / "burned-area-eval-v1.json"
YEARS = list(range(2001, 2017))
N_MONTHS = len(YEARS) * 12
FIRE_MAX_RATE = 5.0

ABC_MODELS = ("A-legacy", "B-legacy", "C-legacy")
GFED5_MODELS = ("C", "D", "E", "F", "G", "G6", "G7", "H", "I", "Ibest")
ALL_MODELS = ABC_MODELS + GFED5_MODELS

SCORE_NAMES = {
    "Bias Score": "bias_score",
    "RMSE Score": "rmse_score",
    "Seasonal Cycle Score": "seasonal_cycle_score",
    "Spatial Distribution Score": "spatial_distribution_score",
    "Overall Score": "overall_score",
}

REPLAY_EXPECTED = {
    "A-legacy": {
        "bias_score": 0.711123,
        "rmse_score": 0.490945,
        "seasonal_cycle_score": 0.800552,
        "spatial_distribution_score": 0.766962,
        "overall_score": 0.652105,
    },
    "B-legacy": {
        "bias_score": 0.706665,
        "rmse_score": 0.477999,
        "seasonal_cycle_score": 0.833706,
        "spatial_distribution_score": 0.757963,
        "overall_score": 0.650866,
    },
    "C-legacy": {
        "bias_score": 0.718782,
        "rmse_score": 0.512102,
        "seasonal_cycle_score": 0.839311,
        "spatial_distribution_score": 0.776872,
        "overall_score": 0.671834,
    },
    "C": {
        "bias_score": 0.697665,
        "rmse_score": 0.475449,
        "seasonal_cycle_score": 0.824616,
        "spatial_distribution_score": 0.769089,
        "overall_score": 0.648453,
        "mha_per_year": 1000.984,
    },
    "D": {
        "bias_score": 0.695095,
        "rmse_score": 0.466179,
        "seasonal_cycle_score": 0.791369,
        "spatial_distribution_score": 0.786354,
        "overall_score": 0.641035,
        "mha_per_year": 1218.699,
    },
    "E": {
        "bias_score": 0.751445,
        "rmse_score": 0.475271,
        "seasonal_cycle_score": 0.745543,
        "spatial_distribution_score": 0.875564,
        "overall_score": 0.664619,
        "mha_per_year": 815.606,
    },
    "F": {
        "bias_score": 0.753049,
        "rmse_score": 0.511981,
        "seasonal_cycle_score": 0.774484,
        "spatial_distribution_score": 0.840027,
        "overall_score": 0.678304,
        "mha_per_year": 785.227,
    },
    "G": {
        "bias_score": 0.742971,
        "rmse_score": 0.488513,
        "seasonal_cycle_score": 0.841920,
        "spatial_distribution_score": 0.847104,
        "overall_score": 0.681804,
        "mha_per_year": 1002.424,
    },
    "G6": {
        "bias_score": 0.747339,
        "rmse_score": 0.491287,
        "seasonal_cycle_score": 0.843674,
        "spatial_distribution_score": 0.850320,
        "overall_score": 0.684781,
        "mha_per_year": 943.213,
    },
    "G7": {
        "bias_score": 0.748123,
        "rmse_score": 0.493611,
        "seasonal_cycle_score": 0.845441,
        "spatial_distribution_score": 0.850220,
        "overall_score": 0.686201,
        "mha_per_year": 945.842,
    },
    "H": {
        "bias_score": 0.725840,
        "rmse_score": 0.520087,
        "seasonal_cycle_score": 0.835265,
        "spatial_distribution_score": 0.808126,
        "overall_score": 0.681881,
        "mha_per_year": 829.571,
    },
    "I": {
        "bias_score": 0.755177,
        "rmse_score": 0.479849,
        "seasonal_cycle_score": 0.820001,
        "spatial_distribution_score": 0.847834,
        "overall_score": 0.676542,
        "mha_per_year": 794.284,
    },
    "Ibest": {
        "bias_score": 0.756125,
        "rmse_score": 0.479257,
        "seasonal_cycle_score": 0.821985,
        "spatial_distribution_score": 0.846391,
        "overall_score": 0.676603,
        "mha_per_year": 805.814,
    },
}

ARCHIVED_REPORTED = {
    "A-legacy": {"overall_score": 0.6574},
    "B-legacy": {"overall_score": 0.6506},
    "C-legacy": {"overall_score": 0.6733},
    "C": {"overall_score": 0.6485, "mha_per_year": 1001.0},
    "D": {"overall_score": 0.6411, "mha_per_year": 1219.0},
    "E": {"overall_score": 0.6646, "mha_per_year": 816.0},
    "F": {"overall_score": 0.6783, "mha_per_year": 785.0},
    "G": {"overall_score": 0.6818, "mha_per_year": 1002.0},
    "G6": {"overall_score": 0.6848, "mha_per_year": 943.0},
    "G7": {"overall_score": 0.6862, "mha_per_year": 946.0},
    "H": {"overall_score": 0.6819, "mha_per_year": 830.0},
    "I": {"overall_score": 0.6765, "mha_per_year": 794.0},
    "Ibest": {"overall_score": 0.6766, "mha_per_year": 806.0},
}

REGION_BOX = {
    "Africa": (-20, 52, -36, 18),
    "S.America": (-82, -34, -56, 14),
    "N.America": (-168, -52, 14, 74),
    "Boreal": (40, 180, 48, 78),
    "SEAsia": (60, 150, -11, 30),
    "Australia": (112, 154, -44, -10),
    "Europe": (-12, 40, 36, 72),
}

ASSEMBLIES = {
    "E": {
        "fallback": "params.spatial.k1.json",
        "seasonal": True,
        "regions": {
            "Africa": "params.africafuel.json",
            "Boreal": "params.boreal.json",
            "S.America": "params.samerica.json",
            "SEAsia": "params.seasia.json",
            "Europe": "params.europe.json",
        },
    },
    "G": {
        "fallback": "params.nsga2.json",
        "seasonal": False,
        "regions": {
            "Africa": "params.G_Africa.json",
            "Boreal": "params.G_Boreal.json",
            "S.America": "params.G_SAmerica.json",
            "SEAsia": "params.G_SEAsia.json",
            "Europe": "params.G_Europe.json",
        },
    },
    "G6": {
        "fallback": "params.nsga2.json",
        "seasonal": False,
        "regions": {
            "Africa": "params.G_Africa.json",
            "Boreal": "params.G_Boreal.json",
            "S.America": "params.G_SAmerica.json",
            "SEAsia": "params.G_SEAsia.json",
            "Europe": "params.G_Europe.json",
            "N.America": "params.G_NAmerica.json",
        },
    },
    "G7": {
        "fallback": "params.nsga2.json",
        "seasonal": False,
        "regions": {
            "Africa": "params.G_Africa.json",
            "Boreal": "params.G_Boreal.json",
            "S.America": "params.G_SAmerica.json",
            "SEAsia": "params.G_SEAsia.json",
            "Europe": "params.G_Europe.json",
            "N.America": "params.G_NAmerica.json",
            "Australia": "params.G_Australia.json",
        },
    },
    "I": {
        "fallback": "params.nsga2.json",
        "seasonal": False,
        "regions": {
            "Africa": "params.Gtrop_Africa.json",
            "Boreal": "params.Gtrop_Boreal.json",
            "S.America": "params.Gtrop_SAmerica.json",
            "SEAsia": "params.Gtrop_SEAsia.json",
            "Europe": "params.Gtrop_Europe.json",
            "N.America": "params.Gtrop_NAmerica.json",
            "Australia": "params.Gtrop_Australia.json",
        },
    },
    "Ibest": {
        "fallback": "params.nsga2.json",
        "seasonal": False,
        "regions": {
            "Africa": "params.Gtrop_Africa.json",
            "Boreal": "params.G_Boreal.json",
            "S.America": "params.Gtrop_SAmerica.json",
            "SEAsia": "params.Gtrop_SEAsia.json",
            "Europe": "params.G_Europe.json",
            "N.America": "params.G_NAmerica.json",
            "Australia": "params.Gtrop_Australia.json",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("cftime", "h5py", "netCDF4", "numpy", "scipy", "xarray"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    ilamb = json.loads(CONTRACT.read_text())["evaluation"]["ilamb"]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "ilamb": ilamb,
        "invocation": [sys.executable, *sys.argv],
        "replay_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def protected_benchmark_hashes() -> dict[str, str]:
    contract = json.loads(CONTRACT.read_text())
    protected: dict[str, str] = {}
    for item in contract["protected_files"]:
        path = item["path"]
        if not path.startswith("data/benchmarks/"):
            continue
        actual = sha256_file(ROOT / path)
        if actual != item["sha256"]:
            raise RuntimeError(f"protected file hash mismatch: {path}")
        protected[path] = actual
    return protected


def require_files(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing historical replay inputs:\n" + "\n".join(missing))


def coarsen(array: np.ndarray) -> np.ndarray:
    return array.reshape(*array.shape[:-2], 180, 2, 360, 2).mean(
        axis=(-3, -1)
    ).astype(np.float32)


def uncoarsen(array: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(array, 2, axis=-2), 2, axis=-1).astype(np.float32)


def sig(array: np.ndarray, k: float, center: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-k * (array - center), -50, 50)))


def supp(array: np.ndarray, k: float, center: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(k * (array - center), -50, 50)))


def hump(array: np.ndarray, rise: float, decay: float) -> np.ndarray:
    rise = max(rise, 1e-9)
    decay = max(decay, 1e-9)
    return (1.0 - np.exp(-np.clip(array / rise, 0, 500))) * np.exp(
        -np.clip(array / decay, 0, 500)
    )


def load_params(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    params = payload.get("params", payload)
    return {key: float(value) for key, value in params.items()}


def load_reference(path: Path) -> np.ndarray:
    with xr.open_dataset(path) as dataset:
        values = dataset["burntArea"].isel(time=slice(0, N_MONTHS)).values.astype(
            np.float32
        )
        units = str(dataset["burntArea"].attrs.get("units", "")).strip()
    if units in {"%", "percent", "percentage"}:
        values = values / 100.0
    return coarsen(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))


def load_gfed4_source() -> np.ndarray:
    source = ROOT / "data" / "benchmarks" / "source" / "gfed4.1s"
    output = np.zeros((N_MONTHS, 180, 360), dtype=np.float32)
    index = 0
    for year in YEARS:
        with h5py.File(source / f"GFED4.1s_{year}.hdf5", "r") as handle:
            for month in range(1, 13):
                values = handle[f"burned_area/{month:02d}/burned_fraction"][:]
                values = values[::-1, :]
                output[index] = values.reshape(180, 4, 360, 4).mean(axis=(1, 3))
                index += 1
    return np.nan_to_num(output, nan=0.0)


def gpp_anomaly_terms(
    gpp: np.ndarray, anom_k: float, anom_c: float, fuel_anom_k: float
) -> tuple[np.ndarray, np.ndarray]:
    anomaly = gpp - gpp.mean(axis=0, keepdims=True)
    anomaly_suppression = supp(anomaly, anom_k, anom_c)
    negative = np.clip(-anomaly, 0, None)
    anomaly_boost = 1.0 - np.exp(-negative / (fuel_anom_k + 1e-9))
    return anomaly_suppression, anomaly_boost


def fire_legacy_c(drivers: dict[str, np.ndarray], p: dict[str, float]) -> np.ndarray:
    product = (
        sig(drivers["dbar"], p["k1"], p["D_low"])
        * supp(drivers["dbar"], p["k2"], p["D_high"])
        * drivers["p_ann"]
        / (drivers["p_ann"] + p["P_half"] + 1e-12)
        * (1.0 / (1.0 + drivers["p_month"] / (p["pre_dampen_half"] + 1e-12)))
        * hump(
            p["gpp_af"] * drivers["gpp_monthly"], p["gpp_b"], p["gpp_d"]
        )
        * sig(drivers["t_air"], p["ign_k"], p["ign_c"])
    )
    return np.power(np.clip(product, 0, None), p["fire_exp"]).astype(np.float32)


def fire_legacy_b(drivers: dict[str, np.ndarray], p: dict[str, float]) -> np.ndarray:
    anomaly_suppression, anomaly_boost = gpp_anomaly_terms(
        drivers["gpp_monthly"], p["anom_k"], p["anom_c"], p["fuel_anom_k"]
    )
    product = (
        sig(drivers["dbar"], p["k1"], p["D_low"])
        * supp(drivers["dbar"], p["k2"], p["D_high"])
        * drivers["p_ann"]
        / (drivers["p_ann"] + p["P_half"] + 1e-12)
        * (1.0 / (1.0 + drivers["p_month"] / (p["pre_dampen_half"] + 1e-12)))
        * hump(
            p["gpp_af"] * drivers["gpp_monthly"], p["gpp_b"], p["gpp_d"]
        )
        * anomaly_suppression
        * anomaly_boost
        * sig(drivers["t_surf"], p["ts_k"], p["ts_c"])
        * sig(drivers["t_air"], p["ign_k"], p["ign_c"])
    )
    return np.power(np.clip(product, 0, None), p["fire_exp"]).astype(np.float32)


def fire_legacy_a(drivers: dict[str, np.ndarray], p: dict[str, float]) -> np.ndarray:
    anomaly_suppression, anomaly_boost = gpp_anomaly_terms(
        drivers["gpp_monthly"], p["anom_k"], p["anom_c"], p["fuel_anom_k"]
    )
    deep_change = np.diff(
        drivers["t_deep"], axis=0, prepend=drivers["t_deep"][[0]]
    )
    fuel_argument = p["af"] * drivers["agb"] + p["af_lai"] * drivers["lai"]
    product = (
        sig(drivers["dbar"], p["k1"], p["D_low"])
        * supp(drivers["dbar"], p["k2"], p["D_high"])
        * hump(fuel_argument, p["fb"], p["fd"])
        * sig(drivers["t_deep"], p["ss2"], p["sc2"])
        * sig(deep_change, p["rate_k"], p["rate_c"])
        * drivers["p_ann"]
        / (drivers["p_ann"] + p["P_half"] + 1e-12)
        * (1.0 / (1.0 + drivers["p_month"] / (p["pre_dampen_half"] + 1e-12)))
        * supp(drivers["h_natr"], p["h_k"], p["h_crit"])
        * hump(
            p["gpp_af"] * drivers["gpp_monthly"], p["gpp_b"], p["gpp_d"]
        )
        * anomaly_suppression
        * anomaly_boost
        * sig(drivers["t_surf"], p["ts_k"], p["ts_c"])
        * sig(drivers["t_air"], p["ign_k"], p["ign_c"])
    )
    return np.power(np.clip(product, 0, None), p["fire_exp"]).astype(np.float32)


def load_legacy_drivers() -> dict[str, np.ndarray]:
    climate = ROOT / "data" / "inputs" / "climate" / "crujra-processed"
    drivers = {
        name: np.nan_to_num(np.load(climate / f"{name}_monthly.npy"), nan=0.0)
        for name in ("dbar", "t_deep", "p_ann", "t_air", "p_month", "t_surf")
    }
    ecosystem = ROOT / "data" / "inputs" / "ecosystem"
    trendy = ecosystem / "trendy-v14-ed"
    with xr.open_dataset(trendy / "EDv3_S3_gpp.nc", decode_times=False) as dataset:
        gpp = (
            dataset["gpp"]
            .isel(time=slice(3612, 3804))
            .values.astype(np.float32)
            * 86400
            * 365
        )
        latitude = (
            dataset.latitude.values
            if "latitude" in dataset.coords
            else dataset.lat.values
        )
    if latitude[0] > latitude[-1]:
        gpp = gpp[:, ::-1, :]
    drivers["gpp_monthly"] = coarsen(np.nan_to_num(gpp, nan=0.0))
    del gpp

    static = ecosystem / "ed-static"
    for name in ("h_natr", "h_scnd", "f_natr", "f_scnd"):
        drivers[name] = np.nan_to_num(
            np.load(static / f"{name}_monthly.npy"), nan=0.0
        )

    with xr.open_dataset(trendy / "EDv3_S3_cLeaf.nc", decode_times=False) as leaf_ds:
        leaf = leaf_ds["cLeaf"].isel(time=slice(301, 317)).values
    with xr.open_dataset(trendy / "EDv3_S3_cWood.nc", decode_times=False) as wood_ds:
        wood = wood_ds["cWood"].isel(time=slice(301, 317)).values
    if latitude[0] > latitude[-1]:
        leaf = leaf[:, ::-1, :]
        wood = wood[:, ::-1, :]
    annual_agb = coarsen(
        np.nan_to_num(leaf + wood, nan=0.0).astype(np.float32)
    )
    drivers["agb"] = np.repeat(annual_agb, 12, axis=0)
    del leaf, wood, annual_agb

    with xr.open_dataset(ecosystem / "ed-simulation.nc", decode_times=False) as sim:
        lai = sim["LAI"].isel(time=slice(240, 432)).mean("time").values
        sim_lat = sim.lat.values
    if sim_lat[0] > sim_lat[-1]:
        lai = lai[::-1, :]
    drivers["lai"] = coarsen(
        np.nan_to_num(lai, nan=0.0).astype(np.float32)
    )
    return drivers


def historical_rescale(
    prediction: np.ndarray, observation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mask = (observation > 0).any(axis=0)
    cosine = np.cos(np.deg2rad(np.arange(-89.5, 90.0, 1.0))).astype(np.float32)
    weights = np.broadcast_to(cosine[None, :, None], observation.shape)
    masked_weights = weights * mask[None, :, :]
    model_mean = float((prediction * masked_weights).sum() / masked_weights.sum())
    observed_mean = float((observation * masked_weights).sum() / masked_weights.sum())
    if model_mean <= 0:
        raise RuntimeError("historical model generated a non-positive mean")
    scaled = prediction * (observed_mean / model_mean)
    return scaled.astype(np.float32), mask


def fire_c(drivers: dict[str, np.ndarray], p: dict[str, float]) -> np.ndarray:
    onset = sig(drivers["dbar"], p["k1"], p["D_low"])
    suppression = supp(drivers["dbar"], p["k2"], p["D_high"])
    precipitation_floor = drivers["p_ann"] / (
        drivers["p_ann"] + p["P_half"] + 1e-12
    )
    precipitation_dampening = 1.0 / (
        1.0 + drivers["p_month"] / (p["pre_dampen_half"] + 1e-12)
    )
    gpp_modifier = hump(
        p["gpp_af"] * drivers["gpp_monthly"], p["gpp_b"], p["gpp_d"]
    )
    ignition = sig(drivers["t_air"], p["ign_k"], p["ign_c"])
    product = (
        onset
        * suppression
        * precipitation_floor
        * precipitation_dampening
        * gpp_modifier
        * ignition
    )
    if "agb" in drivers and "k_veg" in p and "agb_crit" in p:
        product *= supp(drivers["agb"], p["k_veg"], p["agb_crit"])
    if "agb" in drivers and "trop_k_veg" in p and "trop_agb_crit" in p:
        latitude = -89.5 + np.arange(180, dtype=np.float32)
        tropical = (np.abs(latitude) < p.get("trop_lat", 23.5)).astype(np.float32)[
            None, :, None
        ]
        ratio = np.clip(
            drivers["agb"] / (p["trop_agb_crit"] + 1e-12), 0, None
        )
        canopy = 1.0 / (1.0 + np.power(ratio, p["trop_k_veg"]))
        product *= tropical * canopy + (1.0 - tropical)
    rate = np.power(np.clip(product, 0, None), p["fire_exp"])
    if "fuel_k" in p:
        fuel_capacity = drivers["gpp_monthly"].mean(axis=0, keepdims=True)
        fuel = fuel_capacity / (fuel_capacity + p.get("fuel_half", 1.0) + 1e-9)
        rate *= 1.0 + p["fuel_k"] * fuel
    elif "fire_amp" in p:
        rate *= p["fire_amp"]
    return rate.astype(np.float32)


def transform_rate(rate: np.ndarray, seasonal: bool) -> np.ndarray:
    capped = np.minimum(rate, FIRE_MAX_RATE)
    if seasonal:
        return (1.0 - np.exp(-capped / 12.0)).astype(np.float32)
    return ((1.0 - np.exp(-capped)) / 12.0).astype(np.float32)


def load_coupled_drivers() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    state_path = ROOT / "data" / "inputs" / "coupled" / "state-1997-2016.nc"
    fuel_path = ROOT / "data" / "inputs" / "coupled" / "fuel-state-1997-2016.nc"
    climate_path = ROOT / "data" / "inputs" / "climate" / "crujra-processed"
    drivers = {
        name: np.load(climate_path / f"{name}_monthly.npy").astype(np.float32)
        for name in ("dbar", "p_ann", "p_month", "t_air")
    }
    dump_climate: dict[str, np.ndarray] = {}
    with xr.open_dataset(state_path) as state:
        time_slice = slice(48, 240)
        productivity = {
            tag: np.clip(
                np.nan_to_num(
                    state[f"GPP_month_{tag}"]
                    .isel(time=time_slice)
                    .values.astype(np.float32),
                    nan=0.0,
                ),
                0,
                None,
            )
            for tag in ("ntrl", "scnd", "past")
        }
        fractions = {
            tag: np.nan_to_num(
                state[f"area_frac_{tag}"]
                .isel(time=time_slice)
                .values.astype(np.float32),
                nan=0.0,
            )
            for tag in ("ntrl", "scnd", "past")
        }
        gpp_half_degree = (
            productivity["ntrl"] * fractions["ntrl"]
            + productivity["scnd"] * fractions["scnd"]
            + productivity["past"] * fractions["past"]
        ).astype(np.float32)
        gpp = coarsen(gpp_half_degree)
        del productivity, fractions, gpp_half_degree
        for output_name, source_name in (
            ("dbar", "D_bar"),
            ("p_ann", "P_ann"),
            ("p_month", "P_month"),
            ("t_air", "T_air"),
        ):
            dump_climate[output_name] = coarsen(
                np.nan_to_num(
                    state[source_name]
                    .isel(time=time_slice)
                    .values.astype(np.float32),
                    nan=0.0,
                )
            )
    drivers["gpp_monthly"] = gpp
    dump_climate["gpp_monthly"] = gpp
    with xr.open_dataset(fuel_path) as fuel:
        agb = coarsen(
            np.nan_to_num(
                fuel["AGB"].isel(time=slice(48, 240)).values.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
    drivers["agb"] = agb
    dump_climate["agb"] = agb
    return drivers, dump_climate


def assemble(model_id: str, drivers: dict[str, np.ndarray]) -> np.ndarray:
    specification = ASSEMBLIES[model_id]
    base = transform_rate(
        fire_c(drivers, load_params(PARAM_GF5 / specification["fallback"])),
        bool(specification["seasonal"]),
    )
    prediction = base.copy()
    latitude = -89.5 + np.arange(180)
    longitude = -179.5 + np.arange(360)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    assigned = np.zeros((180, 360), dtype=bool)
    for region, filename in specification["regions"].items():
        west, east, south, north = REGION_BOX[region]
        box = (
            (lon_grid >= west)
            & (lon_grid <= east)
            & (lat_grid >= south)
            & (lat_grid <= north)
            & ~assigned
        )
        region_prediction = transform_rate(
            fire_c(drivers, load_params(PARAM_GF5 / filename)),
            bool(specification["seasonal"]),
        )
        prediction[:, box] = region_prediction[:, box]
        assigned |= box
        del region_prediction
    return prediction


def load_gdp() -> np.ndarray:
    return np.load(
        ROOT
        / "data"
        / "inputs"
        / "human"
        / "historical"
        / "gdp-pcap-grid-1deg.npy"
    ).astype(np.float64)


def gdp_multiplier(gdp: np.ndarray, mask: np.ndarray, gamma: float) -> np.ndarray:
    wealth = np.log10(np.clip(gdp, 50.0, None))
    available = np.isfinite(gdp) & (gdp > 0)
    pivot = float(np.median(wealth[available & mask]))
    multiplier = np.power(10.0, gamma * (pivot - wealth))
    multiplier[~available] = 1.0
    return np.clip(multiplier, 0.15, 6.0)


def generate_f(drivers: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    base = load_params(PARAM_GF5 / "params.coupledE_gdp.json")
    base_without_gamma = {key: value for key, value in base.items() if key != "gdp_gamma"}
    rate = fire_c(drivers, base_without_gamma)
    saved = json.loads((PARAM_GF5 / "gdp_regional_gamma.json").read_text())
    latitude = -89.5 + np.arange(180)
    longitude = -179.5 + np.arange(360)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    region_of = np.full((180, 360), "fb", dtype=object)
    assigned = np.zeros((180, 360), dtype=bool)
    for region, bounds in REGION_BOX.items():
        west, east, south, north = bounds
        box = (
            (lon_grid >= west)
            & (lon_grid <= east)
            & (lat_grid >= south)
            & (lat_grid <= north)
            & ~assigned
        )
        region_of[box] = region
        assigned |= box
    gamma_field = np.zeros((180, 360), dtype=np.float64)
    for region, gamma in saved["per_region_gamma"].items():
        gamma_field[region_of == region] = float(gamma)
    gamma_field = gaussian_filter(
        gamma_field, float(saved["sigma"]), mode="nearest"
    )
    gdp = load_gdp()
    wealth = np.log10(np.clip(gdp, 50.0, None))
    available = np.isfinite(gdp) & (gdp > 0)
    pivot = float(np.median(wealth[available & mask]))
    multiplier = np.power(10.0, gamma_field * (pivot - wealth))
    multiplier[~available] = 1.0
    multiplier = np.clip(multiplier, 0.15, 6.0)
    return transform_rate(
        float(saved["s"]) * rate * multiplier[None, :, :], seasonal=True
    )


def generate_h(drivers: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    params = load_params(PARAM_GF5 / "params.H.json")
    gamma = params.pop("gdp_gamma")
    rate = fire_c(drivers, params)
    rate = rate * mask[None, :, :]
    rate = rate * gdp_multiplier(load_gdp(), mask, gamma)[None, :, :]
    return transform_rate(rate, seasonal=False)


def add_cf_bounds(dataset: xr.Dataset) -> xr.Dataset:
    times = dataset.time.values
    bounds = np.empty((len(times), 2), dtype=object)
    for index, time in enumerate(times):
        year, month = time.year, time.month
        bounds[index, 0] = cftime.DatetimeNoLeap(year, month, 1)
        bounds[index, 1] = cftime.DatetimeNoLeap(
            year + int(month == 12), month % 12 + 1, 1
        )
    dataset = dataset.assign(time_bounds=(("time", "nb"), bounds))
    dataset.time.attrs.update(
        {"bounds": "time_bounds", "standard_name": "time", "axis": "T"}
    )
    latitude = dataset.lat.values
    longitude = dataset.lon.values
    dataset = dataset.assign(
        lat_bounds=(
            ("lat", "nb"),
            np.stack([latitude - 0.25, latitude + 0.25], axis=1),
        ),
        lon_bounds=(
            ("lon", "nb"),
            np.stack([longitude - 0.25, longitude + 0.25], axis=1),
        ),
    )
    dataset.lat.attrs.update(
        {
            "bounds": "lat_bounds",
            "units": "degrees_north",
            "standard_name": "latitude",
            "axis": "Y",
        }
    )
    dataset.lon.attrs.update(
        {
            "bounds": "lon_bounds",
            "units": "degrees_east",
            "standard_name": "longitude",
            "axis": "X",
        }
    )
    return dataset


def write_model(
    prediction: np.ndarray,
    mask: np.ndarray,
    destination: Path,
    model_id: str,
    protocol: str,
) -> None:
    high_resolution = uncoarsen(
        np.where(mask[None, :, :], prediction, np.nan).astype(np.float32)
    )
    times = [
        cftime.DatetimeNoLeap(year, month, 15)
        for year in YEARS
        for month in range(1, 13)
    ]
    dataset = xr.Dataset(
        {
            "burntArea": (
                ("time", "lat", "lon"),
                high_resolution,
                {
                    "units": "1",
                    "standard_name": "burnt_area_fraction",
                    "long_name": "Burnt Area Fraction",
                },
            )
        },
        coords={
            "time": ("time", times),
            "lat": ("lat", np.arange(-89.75, 90.0, 0.5)),
            "lon": ("lon", np.arange(-179.75, 180.0, 0.5)),
        },
        attrs={
            "title": f"Historical ED-Fire replay: {model_id}",
            "Conventions": "CF-1.7",
            "historical_protocol": protocol,
            "admissible_for_current_research": "false",
            "warning": "Generation uses benchmark-derived information; provenance replay only.",
        },
    )
    dataset = add_cf_bounds(dataset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    time_units = "days since 2001-01-01 00:00:00"
    encoding = {
        "burntArea": {"zlib": True, "complevel": 4, "_FillValue": 1e20},
        "time": {"units": time_units, "calendar": "noleap", "dtype": "float64"},
        "time_bounds": {
            "units": time_units,
            "calendar": "noleap",
            "dtype": "float64",
        },
    }
    temporary = destination.with_suffix(".nc.tmp")
    dataset.to_netcdf(temporary, encoding=encoding, format="NETCDF4_CLASSIC")
    os.replace(temporary, destination)
    dataset.close()


def model_output(output: Path, model_id: str) -> Path:
    group = "abc-gfed4.1s" if model_id in ABC_MODELS else "coupled-gfed5"
    return output / group / "models" / model_id / "burntArea.nc"


def generate_abc(selected: set[str], output: Path) -> None:
    wanted = selected.intersection(ABC_MODELS)
    if not wanted:
        return
    print("Loading GFED4.1s-era drivers...")
    drivers = load_legacy_drivers()
    observation = load_gfed4_source()
    formula = {
        "A-legacy": (fire_legacy_a, PARAM_ABC / "A.json"),
        "B-legacy": (fire_legacy_b, PARAM_ABC / "B.json"),
        "C-legacy": (fire_legacy_c, PARAM_ABC / "C.json"),
    }
    for model_id in ABC_MODELS:
        if model_id not in wanted:
            continue
        print(f"Generating {model_id}...")
        function, parameter_path = formula[model_id]
        raw = function(drivers, load_params(parameter_path))
        scaled, mask = historical_rescale(raw, observation)
        write_model(
            scaled,
            mask,
            model_output(output, model_id),
            model_id,
            "abc_gfed4_1s",
        )
        del raw, scaled
        gc.collect()
    del drivers, observation
    gc.collect()


def generate_gfed5(selected: set[str], output: Path) -> None:
    wanted = selected.intersection(GFED5_MODELS)
    if not wanted:
        return
    print("Loading coupled-refit drivers...")
    drivers, dump_drivers = load_coupled_drivers()
    gfed5 = load_reference(
        ROOT / "data" / "benchmarks" / "observations" / "gfed5-burned-area.nc"
    )
    gfed4 = load_gfed4_source()
    mask_gfed5 = (gfed5 > 0).any(axis=0)
    mask_gfed4 = (gfed4 > 0).any(axis=0)
    for model_id in GFED5_MODELS:
        if model_id not in wanted:
            continue
        print(f"Generating {model_id}...")
        if model_id == "C":
            prediction = transform_rate(
                fire_c(drivers, load_params(PARAM_GF5 / "params.nsga2.json")),
                seasonal=False,
            )
            mask = mask_gfed5
        elif model_id == "D":
            prediction = transform_rate(
                fire_c(drivers, load_params(PARAM_GF5 / "params.paperD.k1.json")),
                seasonal=False,
            )
            mask = mask_gfed5
        elif model_id == "F":
            prediction = generate_f(dump_drivers, mask_gfed5)
            mask = mask_gfed5
        elif model_id == "H":
            prediction = generate_h(drivers, mask_gfed5)
            mask = mask_gfed5
        else:
            prediction = assemble(model_id, drivers)
            mask = mask_gfed4
        write_model(
            prediction,
            mask,
            model_output(output, model_id),
            model_id,
            "coupled_gfed5",
        )
        del prediction
        gc.collect()
    del drivers, dump_drivers, gfed5, gfed4
    gc.collect()


def ensure_reference_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"reference link points to the wrong file: {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing reference path: {link}")
    link.symlink_to(target)


def run_ilamb(group: str, output: Path, selected: set[str]) -> Path | None:
    models = [
        model_id
        for model_id in (ABC_MODELS if group == "abc-gfed4.1s" else GFED5_MODELS)
        if model_id in selected
    ]
    if not models:
        return None
    model_root = output / group / "models"
    missing = [model for model in models if not model_output(output, model).exists()]
    if missing:
        raise FileNotFoundError(f"generate models before evaluation: {', '.join(missing)}")

    contract = json.loads(CONTRACT.read_text())
    executable = Path(contract["evaluation"]["ilamb"]["executable"])
    if not executable.exists():
        raise FileNotFoundError(f"pinned ILAMB executable not found: {executable}")
    references = output / ".references"
    if group == "abc-gfed4.1s":
        config = ROOT / "data" / "benchmarks" / "configs" / "gfed4.1s-burned-area.cfg"
        ensure_reference_link(
            references / "DATA" / "burntArea" / "GFED4.1S" / "burntArea.nc",
            ROOT
            / "data"
            / "benchmarks"
            / "observations"
            / "gfed4.1s-burned-area.nc",
        )
    else:
        config = ROOT / "data" / "benchmarks" / "configs" / "gfed5-burned-area.cfg"
        ensure_reference_link(
            references / "DATA" / "burntArea" / "GFED5" / "burntArea.nc",
            ROOT / "data" / "benchmarks" / "observations" / "gfed5-burned-area.nc",
        )
    build = output / group / "ilamb"
    build.mkdir(parents=True, exist_ok=True)
    mpl = output / ".mplconfig"
    mpl.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "ILAMB_ROOT": str(references),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(mpl),
            "PYTHONNOUSERSITE": "1",
        }
    )
    command = [
        str(executable),
        "--config",
        str(config),
        "--model_root",
        str(model_root),
        "--models",
        *models,
        "--build_dir",
        str(build),
        "--clean",
    ]
    print(f"Running ILAMB for {group} ({len(models)} models)...")
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ILAMB failed for {group} with code {completed.returncode}")
    scalar_database = build / "scalar_database.csv"
    if not scalar_database.exists():
        raise FileNotFoundError(f"ILAMB did not write {scalar_database}")
    return scalar_database


def burned_area_mha(path: Path) -> float:
    with xr.open_dataset(path) as dataset:
        values = dataset["burntArea"].values.astype(np.float64)
        latitude = dataset.lat.values.astype(np.float64)
        longitude = dataset.lon.values.astype(np.float64)
    dlat = np.deg2rad(abs(float(latitude[1] - latitude[0])))
    dlon = np.deg2rad(abs(float(longitude[1] - longitude[0])))
    radians = np.deg2rad(latitude)
    cell_area = (
        6371000.0**2
        * dlon
        * (np.sin(radians + dlat / 2) - np.sin(radians - dlat / 2))
    )[:, None] * np.ones((1, len(longitude)))
    annual = np.nansum(
        values.reshape(len(YEARS), 12, len(latitude), len(longitude)), axis=1
    ).mean(axis=0)
    return float(np.nansum(annual * cell_area) / 1e10)


def parse_scores(path: Path) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Region") != "global" or row.get("ScalarName") not in SCORE_NAMES:
                continue
            model_id = row["Model"]
            output.setdefault(model_id, {})[SCORE_NAMES[row["ScalarName"]]] = float(
                row["Data"]
            )
    return output


def write_results(
    output: Path,
    selected: set[str],
    scalar_databases: list[Path],
    protected_before: dict[str, str],
) -> dict[str, Any]:
    scores: dict[str, dict[str, float]] = {}
    for database in scalar_databases:
        scores.update(parse_scores(database))
    rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    for model_id in ALL_MODELS:
        if model_id not in selected:
            continue
        metrics = scores.get(model_id, {})
        if not metrics:
            continue
        mha = burned_area_mha(model_output(output, model_id))
        metrics["mha_per_year"] = mha
        expected = REPLAY_EXPECTED[model_id]
        differences = {
            key: metrics[key] - value for key, value in expected.items() if key in metrics
        }
        tolerances = {
            key: (
                0.5
                if key == "mha_per_year"
                else 1.2e-3
                if model_id == "H" and key == "seasonal_cycle_score"
                else 5e-4
            )
            for key in differences
        }
        passed = all(
            abs(value) <= tolerances[key]
            for key, value in differences.items()
        )
        phase_tie_drift = (
            model_id == "H"
            and abs(differences.get("seasonal_cycle_score", 0.0)) > 5e-4
        )
        checks[model_id] = {
            "status": (
                "pass-with-phase-tie-drift"
                if passed and phase_tie_drift
                else "pass"
                if passed
                else "fail"
            ),
            "tolerances": tolerances,
            "replay_differences": differences,
            "reported_differences": {
                key: metrics[key] - value
                for key, value in ARCHIVED_REPORTED[model_id].items()
                if key in metrics
            },
        }
        row = {
            "protocol": "abc_gfed4_1s" if model_id in ABC_MODELS else "coupled_gfed5",
            "model_id": model_id,
            **metrics,
            "reported_overall_score": ARCHIVED_REPORTED[model_id]["overall_score"],
            "delta_from_reported_overall": metrics["overall_score"]
            - ARCHIVED_REPORTED[model_id]["overall_score"],
            "verification": checks[model_id]["status"],
        }
        rows.append(row)

    metrics_path = output / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protocol",
        "model_id",
        "bias_score",
        "rmse_score",
        "seasonal_cycle_score",
        "spatial_distribution_score",
        "overall_score",
        "mha_per_year",
        "reported_overall_score",
        "delta_from_reported_overall",
        "verification",
    ]
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    protected_after = protected_benchmark_hashes()
    if protected_after != protected_before:
        raise RuntimeError("protected benchmark identities changed during replay")
    verification = {
        "schema": "ed-fire-historical-replay/v1",
        "admissible_for_current_research": False,
        "runtime": runtime_identity(),
        "protected_files_unchanged": True,
        "protected_files": protected_after,
        "checks": checks,
        "outputs": {
            model_id: {
                "path": str(model_output(output, model_id).relative_to(ROOT)),
                "sha256": sha256_file(model_output(output, model_id)),
            }
            for model_id in selected
            if model_output(output, model_id).exists()
        },
    }
    (output / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    return verification


def write_parameter_inventory() -> Path:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "model" / "other-models" / "parameters").rglob("*")):
        if not path.is_file():
            continue
        row: dict[str, Any] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text())
                params = payload.get("params") if isinstance(payload, dict) else None
                row["parameter_count"] = len(params) if isinstance(params, dict) else ""
                row["declared_model"] = (
                    payload.get("model", "") if isinstance(payload, dict) else ""
                )
            except json.JSONDecodeError:
                row["parameter_count"] = ""
                row["declared_model"] = ""
        rows.append(row)
    destination = ROOT / "model" / "other-models" / "parameter-inventory.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "sha256",
                "bytes",
                "parameter_count",
                "declared_model",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model IDs to replay, or all. IDs: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Replay output root inside this repository.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the pinned ILAMB executable and verify the historical values.",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Regenerate the parameter inventory without generating model output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = write_parameter_inventory()
    print(f"Parameter inventory: {inventory.relative_to(ROOT)}")
    if args.inventory_only:
        return 0
    selected = set(ALL_MODELS if "all" in args.models else args.models)
    unknown = selected.difference(ALL_MODELS)
    if unknown:
        raise ValueError("unknown model IDs: " + ", ".join(sorted(unknown)))
    output = args.output.resolve()
    try:
        output.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("--output must remain inside the ED-Fire repository") from error

    required = [
        ROOT / "data" / "inputs" / "ecosystem" / "ed-simulation.nc",
        ROOT / "data" / "inputs" / "coupled" / "state-1997-2016.nc",
        ROOT / "data" / "inputs" / "coupled" / "fuel-state-1997-2016.nc",
        ROOT
        / "data"
        / "inputs"
        / "human"
        / "historical"
        / "gdp-pcap-grid-1deg.npy",
        ROOT / "data" / "benchmarks" / "observations" / "gfed5-burned-area.nc",
        ROOT
        / "data"
        / "benchmarks"
        / "observations"
        / "gfed4.1s-burned-area.nc",
    ]
    require_files(required)
    protected_before = protected_benchmark_hashes()
    generate_abc(selected, output)
    generate_gfed5(selected, output)
    if not args.evaluate:
        protected_after = protected_benchmark_hashes()
        if protected_after != protected_before:
            raise RuntimeError("protected benchmark identities changed during generation")
        print("Generation complete. Historical outputs are not admissible candidates.")
        return 0

    scalar_databases = [
        path
        for path in (
            run_ilamb("abc-gfed4.1s", output, selected),
            run_ilamb("coupled-gfed5", output, selected),
        )
        if path is not None
    ]
    verification = write_results(
        output, selected, scalar_databases, protected_before
    )
    failures = [
        model_id
        for model_id, result in verification["checks"].items()
        if not result["status"].startswith("pass")
    ]
    if failures:
        print("Replay verification failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    print(f"Replay verified: {output.relative_to(ROOT) / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
