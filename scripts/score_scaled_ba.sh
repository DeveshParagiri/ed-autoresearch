#!/usr/bin/env bash
# Score the 0.8x-scaled burned area against GFED5 with official ILAMB.
# Tests the "just multiply our model by 0.8" idea. Run inside the ed-fire env.
#   source $(conda info --base)/etc/profile.d/conda.sh && conda activate ed-fire
#   bash scripts/score_scaled_ba.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export ILAMB_ROOT=ilamb_ref_official
OUT="$PWD/ilamb_out_scaled_ba"; rm -rf "$OUT"; mkdir -p "$OUT"
cp ilamb/burntArea_gfed5.cfg "$OUT/ilamb.cfg"

# Score canonical + both scaled variants side by side so the delta is direct.
# MODELS_SCALED holds ED-ModelC-scaled08 (0.8x) and ED-ModelC-scaled0792 (mean-match).
mkdir -p ilamb/MODELS_SCALED/ED-ModelC-canonical
cp ilamb/MODELS/ED-ModelC-final/burntArea.nc ilamb/MODELS_SCALED/ED-ModelC-canonical/burntArea.nc

ilamb-run --config "$OUT/ilamb.cfg" --model_root "$PWD/ilamb/MODELS_SCALED" \
  --regions global --build_dir "$OUT"

echo "=== Overall Scores (global) ==="
python3 - <<'PY'
import csv
rows=[r for r in csv.DictReader(open("ilamb_out_scaled_ba/scalar_database.csv"))]
for r in rows:
    if r.get("Region")=="global" and r.get("ScalarName")=="Overall Score":
        print(f"{r['Model']:>28}  {float(r['Data']):.4f}")
PY
