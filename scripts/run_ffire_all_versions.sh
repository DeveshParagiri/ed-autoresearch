#!/usr/bin/env bash
# Compute fFire for every paper version from its burned area, using ED's own carbon pools
# and the GFED5-retuned combustion betas. Output goes to a model root ILAMB can score directly.
#
#   bash scripts/run_ffire_all_versions.sh
#
# Fills the fire-carbon-emissions half of Table 2, which George's Results skeleton asks for
# under both the global and the regional bullet.
set -u
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8   # the script prints an approx sign; a cp1252 pipe kills it

BETAS="models/combustion-params/betas.gfed5.json"
ROOT="paper_gmd/models_ffire_paper"
mkdir -p "$ROOT"

for V in ED-stock C D E-clean F G7 H I; do
  BA="paper_gmd/models/$V/burntArea.nc"
  if [ ! -f "$BA" ]; then
    echo "SKIP $V, no burntArea.nc"
    continue
  fi
  echo "=== $V ==="
  python scripts/compute_emissions.py \
      --model "$V" --ba-path "$BA" \
      --betas-json "$BETAS" \
      --out-dir "$ROOT/$V" 2>&1 | grep -Ev "pyproj|_set_context" | sed 's/^/  /'
done

echo
echo "done. fFire written under $ROOT/<version>/fFire.nc"
