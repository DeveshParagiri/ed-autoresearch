#!/usr/bin/env bash
# Download the pinned Model C input bundle from Google Drive.
#
# Bundle contents (~5 GB):
#   crujra/       dbar, p_ann, p_month, t_air  (.npy, 1deg, 192 months)
#   trendy_v14/   EDv3_S3_gpp.nc (raw, full time range)
#   gfed/         GFED4.1s_{2001..2016}.hdf5  (for rescale)
#   outputs/      reference burntArea.nc + params.json
#   CHECKSUMS.txt + README.txt
#
# SHA256 of modelC-inputs.zip: f124b21e778c3a28532acd3fdaea70a701a6d8cb714fafa423a8d748b4a7b4d3
#
# After download, contents merge into ed-autoresearch/data/ and
# ed-autoresearch/ilamb/MODELS/ED-ModelC-final/ automatically.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ZIP_URL="https://drive.google.com/uc?id=1ID5pswHyaaF9Ej1CDgZ7j7pGjpQkKEhQ"
ZIP_SHA256="f124b21e778c3a28532acd3fdaea70a701a6d8cb714fafa423a8d748b4a7b4d3"
ZIP_PATH="$REPO/modelC-inputs.zip"

if ! command -v gdown &>/dev/null; then
  echo "Installing gdown (for Google Drive downloads)..."
  pip install --quiet gdown
fi

if [ -f "$ZIP_PATH" ]; then
  echo "Zip already present at $ZIP_PATH — verifying SHA256..."
  actual=$(shasum -a 256 "$ZIP_PATH" | awk '{print $1}')
  if [ "$actual" = "$ZIP_SHA256" ]; then
    echo "  SHA256 match — skipping download."
  else
    echo "  SHA256 mismatch (got $actual); re-downloading."
    rm -f "$ZIP_PATH"
  fi
fi

if [ ! -f "$ZIP_PATH" ]; then
  echo "Downloading modelC-inputs.zip (~5 GB) from Drive..."
  gdown "$ZIP_URL" -O "$ZIP_PATH"
  actual=$(shasum -a 256 "$ZIP_PATH" | awk '{print $1}')
  if [ "$actual" != "$ZIP_SHA256" ]; then
    echo "FATAL: downloaded zip SHA256 mismatch"
    echo "  expected $ZIP_SHA256"
    echo "  got      $actual"
    exit 2
  fi
  echo "  SHA256 verified."
fi

echo
echo "Extracting into repo layout..."
TMP="$REPO/.modelC-extract-tmp"
rm -rf "$TMP"
mkdir -p "$TMP"
unzip -q "$ZIP_PATH" -d "$TMP"

# Zip's top-level dir is "modelC-inputs/"
SRC="$TMP/modelC-inputs"
if [ ! -d "$SRC" ]; then
  echo "FATAL: unexpected zip layout; no modelC-inputs/ at top level"
  exit 2
fi

mkdir -p "$REPO/data/crujra" "$REPO/data/trendy_v14" "$REPO/data/gfed"
mkdir -p "$REPO/ilamb/MODELS/ED-ModelC-final"

cp -v "$SRC"/crujra/*.npy          "$REPO/data/crujra/"
cp -v "$SRC"/trendy_v14/*.nc       "$REPO/data/trendy_v14/"
cp -v "$SRC"/gfed/*.hdf5           "$REPO/data/gfed/"
cp -v "$SRC"/outputs/burntArea.nc  "$REPO/ilamb/MODELS/ED-ModelC-final/"
# Don't overwrite the repo's params.json with the bundle's — they should be the same.
# But warn if they differ:
if ! diff -q "$SRC/outputs/params.json" "$REPO/models/C/params.json" >/dev/null 2>&1; then
  echo "WARNING: bundle params.json differs from repo's models/C/params.json"
  echo "         Keeping the repo's version. Bundle's copy is at $SRC/outputs/params.json."
fi

rm -rf "$TMP"
echo
echo "Extraction complete."
echo "Next: python scripts/verify.py"
