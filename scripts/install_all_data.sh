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
default_source_root=""
if [ -d "$repo_root/../ED" ]; then
  default_source_root="$(cd "$repo_root/../ED" && pwd)"
fi
source_root="${ED_FIRE_SOURCE_ROOT:-$default_source_root}"
fetch_public=0
command="install"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-root)
      source_root="$2"
      shift 2
      ;;
    --fetch-public)
      fetch_public=1
      shift
      ;;
    --plan)
      command="plan"
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/install_all_data.sh [--source-root PATH] [--fetch-public] [--plan]"
      echo ""
      echo "Without --fetch-public, the script links a prepared canonical data root."
      echo "With --fetch-public, it retrieves the anonymous official datasets before linking."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$source_root" ]; then
  echo "No canonical data root was found. Pass --source-root PATH or set ED_FIRE_SOURCE_ROOT." >&2
  exit 2
fi

if [ "$fetch_public" -eq 1 ]; then
  mkdir -p "$source_root"
  "$python_bin" "$repo_root/scripts/download_public_data.py" --source-root "$source_root" --only all

  luh_root="$source_root/data/luh2/GCB2026"
  if [ ! -f "$luh_root/states.nc" ] || [ ! -f "$luh_root/transitions.nc" ] || [ ! -f "$luh_root/management.nc" ]; then
    "$python_bin" "$repo_root/scripts/download_luh2_gcb2026.py" --output-dir "$luh_root"
  else
    echo "LUH2-GCB2026 files already exist at $luh_root; catalog validation will check their exact sizes."
  fi

  gfed5_burned_area="$source_root/ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc"
  if [ ! -f "$gfed5_burned_area" ]; then
    "$python_bin" "$repo_root/scripts/build_gfed5_burned_area.py" \
      --input-dir "$source_root/data/gfed5" \
      --output "$gfed5_burned_area"
  fi

  gfed5_emissions="$source_root/ilamb_ref_official/DATA/fFire/GFED5/fFire.nc"
  if [ ! -f "$gfed5_emissions" ]; then
    "$python_bin" "$repo_root/scripts/build_gfed5_fire_emissions.py" \
      --source "$source_root/data/gfed/5" \
      --burned-area-mask "$gfed5_burned_area" \
      --output "$gfed5_emissions"
  fi
fi

"$python_bin" "$repo_root/scripts/install_data.py" "$command" --source-root "$source_root"

if [ "$command" = "install" ]; then
  PYTHONPATH="$repo_root/src" "$python_bin" "$repo_root/scripts/check_workspace.py"
fi
