# Environment and data

## Conda env

```bash
conda create -n edfire -c conda-forge python=3.10 ilamb xarray netcdf4 \
  numpy matplotlib h5py pandas cftime cartopy optuna
conda activate edfire
```

Native stack (ILAMB, Cartopy, NetCDF/HDF5) is why this project uses conda rather than uv alone. Pure-Python extras can still be installed into the same env with `uv pip` if you prefer.

## Required inputs (not in git)

| Path | Role |
|---|---|
| `data/crujra/*_monthly.npy` | Climate (dryness, T, precip) |
| `global_baseline_modelC_inputs_1997-2016.nc` | Lei coupled-ED GPP dump |
| `global_baseline_modelCfuel_inputs_1997-2016.nc` | AGB for fuel / fFire |
| `data/gfed/4.1/GFED4.1s_YYYY.hdf5` | Land / fire mask for offline run |
| `ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc` | ILAMB burned-area reference |
| `ilamb_ref_official/DATA/fFire/GFED5/fFire.nc` | ILAMB emissions reference (optional) |
| `data/trendy_v14/EDv3_S3_cSoil.nc` | Soil C for combustion step |

Optional: `bash scripts/download_inputs.sh` for an older pinned climate/GFED4.1 bundle.
