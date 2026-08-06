#!/usr/bin/env bash
# Waits for the running optimizer fits to finish, then assembles the seven-region Model G,
# stages G7 and H, and scores everything against GFED5 in ONE official ILAMB run.
set -u
cd "$(dirname "$0")/.."
EDPY="C:/Users/owusu/miniforge3/envs/edfire/python.exe"
BASEPY="C:/Users/owusu/miniforge3/python.exe"

echo "[wait] for params.G_Australia.json and params.H.json ..."
until [ -f models/C/params.G_Australia.json ] && [ -f models/C/params.H.json ]; do sleep 20; done
echo "[wait] both present; letting the optimizer finish writing"
until ! tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q "optimize"; do sleep 10; break; done
sleep 30

echo "[assemble] Model G with all seven regions, and with six (Australia dropped)"
SEASONAL_TRANSFORM=0 ASSEMBLY=G7 ASSEMBLE_FALLBACK=params.nsga2.json "$EDPY" scripts/assemble_continental.py
SEASONAL_TRANSFORM=0 ASSEMBLY=G6 ASSEMBLE_FALLBACK=params.nsga2.json "$EDPY" scripts/assemble_continental.py

mkdir -p paper_gmd/models/G7 paper_gmd/models/G6 paper_gmd/models/H
cp ilamb/MODELS_CONTINENTAL_G7/ED-ModelG7-continental/burntArea.nc paper_gmd/models/G7/burntArea.nc
cp ilamb/MODELS_CONTINENTAL_G6/ED-ModelG6-continental/burntArea.nc paper_gmd/models/G6/burntArea.nc
cp ilamb/MODELS/ED-ModelC-final/burntArea.H.nc paper_gmd/models/H/burntArea.nc
find paper_gmd/models -name "._*" -delete 2>/dev/null

echo "[ilamb] scoring every version together"
export ILAMB_ROOT="$PWD/ilamb_ref_official"
OUT="$PWD/paper_gmd/scoring/ba_all"; rm -rf "$OUT"; mkdir -p "$OUT"
cp ilamb/burntArea_gfed5.cfg "$OUT/ilamb.cfg"
ilamb-run --config "$OUT/ilamb.cfg" --model_root "$PWD/paper_gmd/models" \
          --regions global --build_dir "$OUT"

echo "[figures] burned-area maps for every version"
"$BASEPY" paper_gmd/figures/make_fig_maps_all.py
echo "=== FINISH_G7_H COMPLETE ==="
