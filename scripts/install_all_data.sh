#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${PYTHON:-}" ]; then
  python_bin="$PYTHON"
elif [ -x "$repo_root/.venv/bin/python" ]; then
  python_bin="$repo_root/.venv/bin/python"
else
  python_bin="python3"
fi
fetch_public=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --fetch-public)
      fetch_public=1
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/install_all_data.sh [--fetch-public]"
      echo ""
      echo "Without --fetch-public, the script validates data already stored in this project."
      echo "With --fetch-public, it also retrieves every anonymous official dataset."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ "$fetch_public" -eq 1 ]; then
  "$python_bin" "$repo_root/scripts/download_public_data.py" --project-root "$repo_root" --only all
  "$python_bin" "$repo_root/scripts/install_stock_ed.py" --project-root "$repo_root" --only all

  luh_root="$repo_root/data/inputs/candidate-drivers/luh2"
  if [ ! -f "$luh_root/states.nc" ] || [ ! -f "$luh_root/transitions.nc" ] || [ ! -f "$luh_root/management.nc" ]; then
    "$python_bin" "$repo_root/scripts/download_luh2_gcb2026.py" --output-dir "$luh_root"
  else
    echo "LUH2-GCB2026 files already exist at $luh_root; catalog validation will check their exact sizes."
  fi

  gfed5_burned_area="$repo_root/data/benchmarks/observations/gfed5-burned-area.nc"
  if [ ! -f "$gfed5_burned_area" ]; then
    "$python_bin" "$repo_root/scripts/build_gfed5_burned_area.py" \
      --input-dir "$repo_root/data/benchmarks/source/gfed5-monthly" \
      --output "$gfed5_burned_area"
  fi

  gfed5_emissions="$repo_root/data/benchmarks/observations/gfed5-fire-emissions.nc"
  if [ ! -f "$gfed5_emissions" ]; then
    "$python_bin" "$repo_root/scripts/build_gfed5_fire_emissions.py" \
      --source "$repo_root/data/benchmarks/source/gfed5-annual" \
      --burned-area-mask "$gfed5_burned_area" \
      --output "$gfed5_emissions"
  fi
fi

PYTHONPATH="$repo_root/src" "$python_bin" "$repo_root/scripts/check_workspace.py"
