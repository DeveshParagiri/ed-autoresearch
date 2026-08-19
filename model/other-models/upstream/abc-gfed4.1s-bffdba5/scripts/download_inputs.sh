#!/bin/bash
# Download TRENDY v14 ED inputs from Wasabi + GFED4.1s reference.
# Total: ~10 GB for full Model A inputs, ~5 GB for B, ~4 GB for C.

set -u
DATA=$(dirname "$0")/../data
mkdir -p "$DATA/trendy_v14" "$DATA/gfed"

BASE=https://s3.eu-west-1.wasabisys.com/gcb-2025-upload/land/output/ED/S3

# Required by all 3 models
for v in cLeaf cWood; do
    target="$DATA/trendy_v14/EDv3_S3_${v}.nc"
    if [[ -f "$target" ]]; then
        echo "SKIP $v (already have)"
        continue
    fi
    echo "downloading $v..."
    curl -L -# -C - -o "$target" "$BASE/EDv3_S3_${v}.nc"
done

# Required by Model A only
echo "downloading gpp (4 GB, ~20 min)..."
curl -L -# -C - -o "$DATA/trendy_v14/EDv3_S3_gpp.nc" "$BASE/EDv3_S3_gpp.nc"

echo "downloading cSoil..."
curl -L -# -C - -o "$DATA/trendy_v14/EDv3_S3_cSoil.nc" "$BASE/EDv3_S3_cSoil.nc"

# GFED4.1s reference (for scoring)
for yr in 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016; do
    target="$DATA/gfed/GFED4.1s_${yr}.hdf5"
    if [[ -f "$target" ]]; then continue; fi
    echo "downloading GFED $yr..."
    curl -L -# -C - -o "$target" "https://www.globalfiredata.org/data_new/GFED4.1s_${yr}.hdf5" 2>/dev/null || echo "GFED download needs manual setup; see globalfiredata.org"
done

echo ""
echo "DONE. Inputs in $DATA/"
echo ""
echo "Still need (not on public bucket):"
echo "  - CRUJRA v3.5 monthly climate 2001-2016 (any source)"
echo "  - ED frozen-sim mean_height_natr/scnd + frac_natr/scnd"
echo ""
echo "See README.md for input specifications."
