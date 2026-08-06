#!/usr/bin/env bash
# The remaining six regions for the vegetation-gated version. Model G's exact recipe
# with TROP_MASK=1 (14 params, adding only trop_agb_crit and trop_k_veg).
#
# Africa was run first as the test, 2026-08-06 09:16, and it worked: Congo closed canopy
# fell from 10.7x observed to 0.37x, the savanna IMPROVED from 1.26x to 1.09x, Africa's
# total went 617 -> 473 Mha against an observed 496, and the regional score rose slightly
# from 0.6343 to 0.6369. The gate is free.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-C:/Users/owusu/miniforge3/envs/edfire/python.exe}"
N="${N_TRIALS:-1500}"
for R in Boreal S.America SEAsia Europe N.America Australia; do
  TAG="Gtrop_$(echo "$R" | tr -d '.')"
  echo "=== $R  (tag=$TAG, N_TRIALS=$N)  $(date '+%H:%M:%S') ==="
  PHYSICAL=1 MAG_BAND=1.3 FP_MIN=0.80 SAMPLER=nsga2 \
  WARM=params.nsga2.json TROP_MASK=1 \
  SEASONAL_TRANSFORM=0 FUEL_AMP=0 RATE_AMP=0 \
  REGION="$R" N_TRIALS="$N" TAG="$TAG" "$PY" scripts/optimize_modelC_coupled.py
  echo "=== done $R  $(date '+%H:%M:%S') ==="
done
echo "=== ALL SIX COMPLETE $(date '+%H:%M:%S') ==="
