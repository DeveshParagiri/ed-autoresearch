#!/usr/bin/env bash
# Model H = Model C + the GDP-per-capita human term, on THE SAME INPUTS as C, D, E and G
# (CRUJRA climate, not the ED dump). Everything else is held at C's recipe exactly, so H
# differs from C in exactly one attribute, the human factor.
#
# This is the comparable replacement for Model F. F was fitted on ED-dump climate with its
# global total pinned to GFED5 by construction and its regional GDP coefficients fitted
# afterwards by coordinate descent, so F cannot be read as one more rung on the same ladder.
# H fits gdp_gamma JOINTLY with the other 12 parameters and pins nothing.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-C:/Users/owusu/miniforge3/envs/edfire/python.exe}"
echo "=== Model H: C + GDP term, CRUJRA climate, $(date '+%H:%M:%S') ==="
PHYSICAL=1 MAG_BAND=1.3 FP_MIN=0.80 SAMPLER=nsga2 \
WARM=params.nsga2.json TROP_MASK=0 \
SEASONAL_TRANSFORM=0 FUEL_AMP=0 RATE_AMP=0 \
GDP_TERM=1 N_TRIALS="${N_TRIALS:-1500}" TAG=H "$PY" scripts/optimize_modelC_coupled.py
echo "=== done Model H $(date '+%H:%M:%S') ==="
