#!/usr/bin/env bash
# Official ILAMB verification of paper model ladder (GFED5 burned area).
#
# Usage:
#   conda activate edfire
#   bash scripts/verify_paper_ilamb.sh
#
# Stages BA under ilamb/MODELS/paper/, runs ilamb-run, writes:
#   ilamb_out_paper_verify/scalar_database.csv
#   paper/ILAMB_VERIFY.md
#   paper/official_ilamb_scores.csv
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if ! command -v ilamb-run >/dev/null 2>&1; then
  echo "FATAL: ilamb-run not on PATH. Activate the edfire env:"
  echo "  conda activate edfire"
  exit 2
fi

export ILAMB_ROOT="${ILAMB_ROOT:-$REPO/ilamb_ref_official}"
export MPLBACKEND=Agg
OUT="$REPO/ilamb_out_paper_verify"
MODEL_ROOT="$REPO/ilamb/MODELS/paper"

mkdir -p "$MODEL_ROOT/Model-C" "$MODEL_ROOT/Model-D" "$MODEL_ROOT/Model-E" "$MODEL_ROOT/ED-stock"

# Prefer already-reproduced paper paths; else fall back to known lab outputs
copy_if() {
  local src="$1" dst="$2"
  if [ -f "$src" ]; then
    cp -f "$src" "$dst"
    echo "staged $dst <- $src"
  fi
}
copy_if "$MODEL_ROOT/Model-C/burntArea.nc" "$MODEL_ROOT/Model-C/burntArea.nc" || true
[ -f "$MODEL_ROOT/Model-C/burntArea.nc" ] || copy_if "$REPO/ilamb/MODELS/ED-ModelC-final/burntArea.nc" "$MODEL_ROOT/Model-C/burntArea.nc"
[ -f "$MODEL_ROOT/Model-D/burntArea.nc" ] || copy_if "$REPO/ilamb/MODELS_TOPK_spatial/ED-ModelC-spatial-k1/burntArea.nc" "$MODEL_ROOT/Model-D/burntArea.nc"
[ -f "$MODEL_ROOT/Model-E/burntArea.nc" ] || copy_if "$REPO/ilamb/MODELS_CONTINENTAL/ED-ModelC-continental/burntArea.nc" "$MODEL_ROOT/Model-E/burntArea.nc"
[ -f "$MODEL_ROOT/ED-stock/burntArea.nc" ] || copy_if "$REPO/ilamb/MODELS/EDv3/burntArea.nc" "$MODEL_ROOT/ED-stock/burntArea.nc"

for m in Model-C Model-D Model-E; do
  [ -f "$MODEL_ROOT/$m/burntArea.nc" ] || { echo "FATAL: missing $MODEL_ROOT/$m/burntArea.nc"; exit 2; }
done

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$REPO/ilamb/burntArea_gfed5.cfg" "$OUT/ilamb.cfg"

echo "Running official ilamb-run ..."
echo "  ILAMB_ROOT=$ILAMB_ROOT"
echo "  model_root=$MODEL_ROOT"
echo "  build_dir=$OUT"

ilamb-run --config "$OUT/ilamb.cfg" \
  --model_root "$MODEL_ROOT" \
  --regions global \
  --build_dir "$OUT"

python - <<'PY'
import pandas as pd
from pathlib import Path
REPO = Path(".").resolve()
df = pd.read_csv(REPO / "ilamb_out_paper_verify/scalar_database.csv")
df = df[(df.Region == "global") & (df.Source == "GFED5")]
want = {
    "Bias Score": "Bias",
    "RMSE Score": "RMSE",
    "Seasonal Cycle Score": "Seasonal",
    "Spatial Distribution Score": "Spatial",
    "Overall Score": "Overall",
}
rows = []
for model in ["ED-stock", "Model-C", "Model-D", "Model-E"]:
    sub = df[df.Model == model]
    if sub.empty:
        continue
    r = {"Model": model}
    for sn, sh in want.items():
        m = sub[sub.ScalarName == sn]
        r[sh] = float(m.Data.iloc[0]) if len(m) else float("nan")
    rows.append(r)
res = pd.DataFrame(rows)
print("\n=== Official ILAMB Overall (global, GFED5) ===")
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
out = REPO / "paper" / "official_ilamb_scores.csv"
out.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(out, index=False)
print(f"\nwrote {out}")
print("full database: ilamb_out_paper_verify/scalar_database.csv")
print("narrative:     paper/ILAMB_VERIFY.md (regenerate by re-running this session analysis if needed)")
PY

echo
echo "Done. Official scores above. Compare to paper Table 1 in paper/ILAMB_VERIFY.md"
