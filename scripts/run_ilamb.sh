#!/usr/bin/env bash
# Score paper models with official ilamb-run against GFED5 burned area.
#
# Prerequisite:
#   python scripts/reproduce_paper.py   # produces ilamb/MODELS/paper/Model-*/
#   ILAMB_ROOT pointing at GFED5 reference (default: ilamb_ref_official)
#
# Usage:
#   bash scripts/run_ilamb.sh
#   MODEL_ROOT=ilamb/MODELS/paper OUT=/tmp/ilamb_out bash scripts/run_ilamb.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export ILAMB_ROOT="${ILAMB_ROOT:-$REPO/ilamb_ref_official}"
CFG="${ILAMB_CFG:-$REPO/ilamb/burntArea_gfed5.cfg}"
MODEL_ROOT="${MODEL_ROOT:-$REPO/ilamb/MODELS/paper}"
OUT="${OUT:-$REPO/ilamb_out_paper}"

if [ ! -f "$CFG" ]; then
  echo "FATAL: config $CFG not found"
  exit 2
fi
if [ ! -d "$MODEL_ROOT" ]; then
  echo "FATAL: model root $MODEL_ROOT not found"
  echo "Run: python scripts/reproduce_paper.py"
  exit 2
fi

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$CFG" "$OUT/ilamb.cfg"

echo "Running ilamb-run:"
echo "  config:     $OUT/ilamb.cfg"
echo "  model_root: $MODEL_ROOT"
echo "  build_dir:  $OUT"
echo "  ILAMB_ROOT: $ILAMB_ROOT"
echo

ilamb-run --config "$OUT/ilamb.cfg" --model_root "$MODEL_ROOT" \
  --regions global --build_dir "$OUT"

echo
echo "Scores: $OUT/scalar_database.csv  (Region=global, ScalarName='Overall Score')"
