#!/usr/bin/env bash
# Model G = Model C's form + C's objective (S_overall), fitted PER CONTINENT.
# Isolates the spatial-parameterization column of George's table. Everything else is
# held at C's recipe exactly: 12-param form (TROP_MASK=0 drops trop_agb_crit/trop_k_veg),
# NSGA-II with the physical objective, MAG_BAND 1.3, FP_MIN 0.80, warm-started from C.
# The ONLY change from C is REGION. Five regions to match Model E's clean assembly
# (N.America and Australia stay on the global fallback there, so they do here too).
set -u
cd "$(dirname "$0")/.."
PY="${PY:-C:/Users/owusu/miniforge3/envs/edfire/python.exe}"
N="${N_TRIALS:-1500}"
REGIONS="${REGIONS:-Africa Boreal S.America SEAsia Europe}"
for R in $REGIONS; do
  TAG="G_$(echo "$R" | tr -d '.')"
  echo "=== Model G: fitting $R  (tag=$TAG, N_TRIALS=$N)  $(date '+%H:%M:%S') ==="
  PHYSICAL=1 MAG_BAND=1.3 FP_MIN=0.80 SAMPLER=nsga2 \
  WARM=params.nsga2.json TROP_MASK=0 \
  SEASONAL_TRANSFORM=0 FUEL_AMP=0 RATE_AMP=0 \
  REGION="$R" N_TRIALS="$N" TAG="$TAG" "$PY" scripts/optimize_modelC_coupled.py
  echo "=== done $R  $(date '+%H:%M:%S') ==="
done
echo "=== ALL REGIONS COMPLETE $(date '+%H:%M:%S') ==="
