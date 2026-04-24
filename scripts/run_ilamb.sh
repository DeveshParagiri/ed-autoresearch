#!/usr/bin/env bash
# Run official ilamb-run against produced Model C burntArea.nc.
#
# Prerequisite:
#   - models/C/params.json is the retuned 12-param set
#   - ilamb/MODELS/ED-ModelC-final/burntArea.nc produced by reproduce_modelC.py
#   - ILAMB GFED4.1s reference downloaded via `ilamb-fetch`
#   - ilamb/burntArea_official.cfg in place (stock ConfBurntArea config)
#
# The config path is fetched from $ILAMB_CFG or defaults to ilamb/burntArea_official.cfg.
# Output goes to ilamb/output_modelC/.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${ILAMB_CFG:-$REPO/ilamb/burntArea_official.cfg}"
MODEL_ROOT="${MODEL_ROOT:-$REPO/ilamb/MODELS}"
OUT="${OUT:-$REPO/ilamb/output_modelC}"

if [ ! -f "$CFG" ]; then
  echo "FATAL: config $CFG not found"
  echo "Expected a stock ConfBurntArea config pointing at DATA/burntArea/GFED4.1S/burntArea.nc"
  exit 2
fi

if [ ! -d "$MODEL_ROOT/ED-ModelC-final" ]; then
  echo "FATAL: $MODEL_ROOT/ED-ModelC-final not found"
  echo "Run scripts/reproduce_modelC.py first."
  exit 2
fi

mkdir -p "$OUT"

echo "Running ilamb-run:"
echo "  config:     $CFG"
echo "  model_root: $MODEL_ROOT"
echo "  build_dir:  $OUT"
echo

ilamb-run --config "$CFG" --model_root "$MODEL_ROOT" --regions global --build_dir "$OUT"

echo
echo "Overall Score for ED-ModelC-final:"
python3 - <<'EOF'
import pandas as pd, sys, os
csv = os.path.join(os.environ.get("OUT", ""), "scalar_database.csv")
if not os.path.exists(csv):
    # fall back to default path
    import glob
    cs = glob.glob("ilamb/output_modelC/scalar_database.csv")
    if cs: csv = cs[0]
if not os.path.exists(csv):
    print("scalar_database.csv not found"); sys.exit(1)
df = pd.read_csv(csv)
df = df[(df["Region"]=="global") & (df["Model"]=="ED-ModelC-final")]
for name in ["Bias Score","RMSE Score","Seasonal Cycle Score","Spatial Distribution Score","Overall Score"]:
    r = df[df["ScalarName"]==name]["Data"]
    if len(r): print(f"  {name:<30} {float(r.iloc[0]):.4f}")
EOF
