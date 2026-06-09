#!/usr/bin/env bash
# Official ILAMB scoring of the seasonal-transform PROTOTYPE vs GFED5.
# Run on a machine with the ed-fire env (ilamb-run on PATH). Bash, not PowerShell.
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ed-fire

export ILAMB_ROOT=ilamb_ref_official
# AppleDouble guard (exFAT drive on macOS): ._*.nc would crash the model-dir merge
find ilamb/MODELS_SEASONAL_PROTO -name '._*' -delete || true

OUT="$PWD/ilamb_out_proto_seasonal"; rm -rf "$OUT"; mkdir -p "$OUT"
cp ilamb/burntArea_gfed5.cfg "$OUT/ilamb.cfg"
ilamb-run --config "$OUT/ilamb.cfg" \
  --model_root "$PWD/ilamb/MODELS_SEASONAL_PROTO" \
  --regions global --build_dir "$OUT"

echo "=== Overall scores (global) ==="
grep -i "Overall Score" "$OUT/scalar_database.csv" | grep -i global || \
  echo "read $OUT/scalar_database.csv (filter Region==global, ScalarName=='Overall Score')"
