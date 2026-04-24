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
# Optional: pass --with-terms to also fetch the 107 MB per-term debug NetCDF
# (out_terms/modelC_terms.nc — 5 input drivers + 6 intermediate mechanism
# terms + final product; useful for coupled-ED cross-comparison).
#
# After download, contents merge into ed-autoresearch/data/,
# ed-autoresearch/ilamb/MODELS/ED-ModelC-final/, and ed-autoresearch/out_terms/.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ZIP_URL="https://drive.google.com/uc?id=1ID5pswHyaaF9Ej1CDgZ7j7pGjpQkKEhQ"
ZIP_SHA256="f124b21e778c3a28532acd3fdaea70a701a6d8cb714fafa423a8d748b4a7b4d3"
ZIP_PATH="$REPO/modelC-inputs.zip"

TERMS_URL="https://drive.google.com/uc?id=1ges8y2qw1KF8eNt8ruIgO3Akj8HGaTYo"
TERMS_SHA256="c25f60810a1b49e228c4dd6905da99acd2831727ad0cb7668a23567e71ebb802"
TERMS_PATH="$REPO/out_terms/modelC_terms.nc"

WITH_TERMS=0
for arg in "$@"; do
  case "$arg" in
    --with-terms) WITH_TERMS=1 ;;
    -h|--help)
      echo "Usage: bash scripts/download_inputs.sh [--with-terms]"
      echo ""
      echo "  --with-terms    Also download the 107 MB per-term debug NetCDF"
      echo "                  (modelC_terms.nc). Useful for coupled-ED"
      echo "                  cross-comparison. Not needed for reproducing"
      echo "                  the ILAMB score."
      exit 0 ;;
    *) echo "Unknown arg: $arg (use --help)"; exit 2 ;;
  esac
done

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

# Optional: fetch per-term debug NetCDF
if [ "$WITH_TERMS" -eq 1 ]; then
  echo
  echo "Fetching per-term debug NetCDF (modelC_terms.nc, ~107 MB)..."
  mkdir -p "$(dirname "$TERMS_PATH")"
  if [ -f "$TERMS_PATH" ]; then
    actual=$(shasum -a 256 "$TERMS_PATH" | awk '{print $1}')
    if [ "$actual" = "$TERMS_SHA256" ]; then
      echo "  Already present and SHA256 match — skipping download."
    else
      rm -f "$TERMS_PATH"
    fi
  fi
  if [ ! -f "$TERMS_PATH" ]; then
    gdown "$TERMS_URL" -O "$TERMS_PATH"
    actual=$(shasum -a 256 "$TERMS_PATH" | awk '{print $1}')
    if [ "$actual" != "$TERMS_SHA256" ]; then
      echo "FATAL: terms SHA256 mismatch (got $actual, expected $TERMS_SHA256)"
      exit 2
    fi
    echo "  SHA256 verified."
  fi
fi

echo
echo "Next: python scripts/verify.py"
