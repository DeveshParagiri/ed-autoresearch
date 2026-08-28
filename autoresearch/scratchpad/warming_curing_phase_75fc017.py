"""Translate the held-ML warming-by-curing interaction mechanistically.

The learned surfaces never enter this equation. A fixed causal redistribution
moves existing surface hazard toward rapid warming before peak GPP senescence,
relative to its own trailing local mean. It is globally shared and annual-source
free. Exact replay occurs only for a formulation improving normalized-cycle
loss in every held whole-cell block.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRATCH))

from scripts.fast_ilamb import GFED5Evaluator  # noqa: E402
from scripts.runtime import GFED5_PATH, load_inputs, validate_prediction  # noqa: E402
from secondary_regrowth_footprint_33ac854 import MONTH_DAYS, losses  # noqa: E402


PINNED = "75fc017"


def load_pinned():
    source = subprocess.run(
        ["git", "show", f"{PINNED}:autoresearch/model.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType(f"model_{PINNED}")
    exec(compile(source, f"{PINNED}:autoresearch/model.py", "exec"), module.__dict__)
    return module


def antecedent(values: np.ndarray, months: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / months)
    state = np.asarray(values[0], dtype=np.float64).copy()
    output = np.empty_like(values, dtype=np.float64)
    for time in range(values.shape[0]):
        state += alpha * (values[time] - state)
        output[time] = state
    return output


def rising(values: np.ndarray, scale: float, center: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-(values-center)/scale,-50.0,50.0)))


def phase_signal(data) -> np.ndarray:
    gpp = np.clip(np.asarray(data["gpp"], dtype=np.float64), 0.0, None)
    gpp3 = antecedent(gpp, 3.0)
    curing = np.maximum((gpp3-gpp)/(gpp3+gpp+0.2),0.0)
    curing_fraction = curing/(curing+0.05)
    gpp12 = antecedent(gpp, 12.0)
    fine_fuel = gpp12/(gpp12+0.35)

    temperature = np.asarray(data["air_temperature"], dtype=np.float64)
    temperature3 = antecedent(temperature, 3.0)
    warming = rising(temperature-temperature3,1.5,0.5)
    thermal = rising(temperature,3.0,5.0)

    crop = np.clip(data["luh2_cropland_fraction"],0.0,1.0)
    range_ = np.clip(data["luh2_rangeland_fraction"],0.0,1.0)
    pasture = np.clip(data["luh2_pasture_fraction"],0.0,1.0)
    urban = np.clip(data["luh2_urban_fraction"],0.0,1.0)
    natural = np.clip(data["natural_vegetation_fraction"],0.0,1.0)
    secondary = np.clip(data["secondary_vegetation_fraction"],0.0,1.0)
    canopy = np.clip(data["natural_canopy_height"],0.0,None)
    secondary_canopy = np.clip(data["secondary_canopy_height"],0.0,None)
    biomass = np.clip(data["aboveground_biomass"],0.0,None)
    continuity = 1.0/(1.0+2.0*crop**1.5+5.0*urban)
    open_cover = np.clip(
        range_+pasture+natural*8/(canopy+8)+secondary*8/(secondary_canopy+8),
        0.0,2.0,
    )
    surface = (1.0-crop)*fine_fuel*open_cover*continuity
    woody = (
        natural*canopy/(canopy+8)
        + secondary*secondary_canopy/(secondary_canopy+8)
    )*biomass/(biomass+1)
    crop_capacity = crop*fine_fuel
    surface_share = surface/(0.05+surface+woody+crop_capacity)
    return np.clip(
        surface_share*thermal*warming*(1.0-curing_fraction),0.0,1.0
    )


def candidate(incumbent, signal, strength):
    hazard = -np.log1p(-np.clip(incumbent,0.0,1.0-1e-7))
    raw = np.exp(np.clip(strength*signal,-0.5,0.5))
    reference = antecedent(raw,12.0)
    factor = np.clip(raw/np.maximum(reference,1e-6),0.5,2.0)
    return np.asarray(1.0-np.exp(-np.clip(hazard*factor,0.0,50.0)),dtype=np.float32)


def main() -> None:
    model = load_pinned()
    data = load_inputs(model.INPUTS)
    incumbent = validate_prediction(model.predict(data,dict(model.PARAMS),None))
    signal = phase_signal(data)
    evaluator = GFED5Evaluator(GFED5_PATH)
    with Dataset(GFED5_PATH) as dataset:
        fine = np.asarray(dataset.variables["burntArea"][:192])
    observed = fine.reshape(192,180,2,360,2).mean(axis=(2,4))/100.0
    area = evaluator.area.reshape(180,2,360,2).sum(axis=(1,3))
    obs_ann = np.average(observed,axis=0,weights=MONTH_DAYS)
    pred_ann = np.average(incumbent,axis=0,weights=MONTH_DAYS)
    obs_weight = area*obs_ann
    excess_weight = area*np.maximum(pred_ann-obs_ann,0.0)
    def top(weight):
        order=np.argsort(weight.ravel())[::-1]
        cumulative=np.cumsum(weight.ravel()[order])/weight.sum()
        return order[:int(np.searchsorted(cumulative,0.90)+1)]
    cells=np.union1d(top(obs_weight),top(excess_weight))
    rows,cols=cells//360,cells%360
    folds=((rows//15)+3*(cols//15))%4
    base_annual,base_cycle=losses(incumbent,observed,area,cells,folds)
    base=evaluator.score(incumbent)["global"]
    print(f"BASE overall={base['overall_score']:.9f}")
    survivor=None
    for strength in (0.1,0.25,0.5,1.0):
        trial=candidate(incumbent,signal,strength)
        annual,cycle=losses(trial,observed,area,cells,folds)
        annual_gain=base_annual-annual
        cycle_gain=base_cycle-cycle
        held=bool(np.all(cycle_gain>0.0) and annual_gain.sum()>=-0.05*cycle_gain.sum())
        print(
            f"strength={strength:g} held={held} annual_gain="
            + ",".join(f"{value:+.6f}" for value in annual_gain)
            + " cycle_gain="
            + ",".join(f"{value:+.6f}" for value in cycle_gain)
        )
        if held and survivor is None:
            survivor=(strength,trial)
    if survivor is None:
        print("EXACT skipped: no stable held survivor")
        return
    strength,trial=survivor
    score=evaluator.score(validate_prediction(trial))["global"]
    print(
        f"EXACT strength={strength:g} overall={score['overall_score']:.9f} "
        f"bias={score['bias_score']:.9f} rmse={score['rmse_score']:.9f} "
        f"seasonal={score['seasonal_cycle_score']:.9f} "
        f"spatial={score['spatial_distribution_score']:.9f}"
    )


if __name__ == "__main__":
    main()
