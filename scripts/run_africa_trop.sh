#!/usr/bin/env bash
# Africa only. Model G's exact recipe with the tropical suppression term ENABLED.
#
# G was run with TROP_MASK=0, which strips trop_agb_crit and trop_k_veg so that G differed
# from Model C in nothing but the spatial resolution. That is what left the Congo rainforest
# defenceless: the Africa region spans savanna and rainforest, savanna dominates the area, and
# a single parameter set with no vegetation gate ignites the forest along with the grass.
# G burns 35.3 %/yr of Congo closed canopy (AGB > 10) against an observed 3.5.
#
# This run changes ONE thing from G_Africa: TROP_MASK=1, giving 14 parameters instead of 12.
# Not E's fuel-headroom terms, not the seasonal transform. Just the vegetation gate.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-C:/Users/owusu/miniforge3/envs/edfire/python.exe}"
echo "=== Africa, G recipe + tropical suppression (TROP_MASK=1)  $(date '+%H:%M:%S') ==="
PHYSICAL=1 MAG_BAND=1.3 FP_MIN=0.80 SAMPLER=nsga2 \
WARM=params.nsga2.json TROP_MASK=1 \
SEASONAL_TRANSFORM=0 FUEL_AMP=0 RATE_AMP=0 \
REGION=Africa N_TRIALS="${N_TRIALS:-1500}" TAG=Gtrop_Africa "$PY" scripts/optimize_modelC_coupled.py
echo "=== done  $(date '+%H:%M:%S') ==="
