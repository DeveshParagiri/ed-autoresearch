"""Score (a), (b), (c) Model C variants on standard ILAMB weighting."""
import json, numpy as np, xarray as xr, sys
sys.path.insert(0, "scripts")
from refit_modelC_magaware import predict_monthly, load_inputs
from refit_modelA_multiobj import four_scores

ds = xr.open_dataset("global_baseline_modelC_inputs_1997-2016.nc")
yr = np.array([d.year for d in ds["time"].values])
m = (yr >= 2001) & (yr <= 2010)
d_tr = load_inputs(ds.isel(time=m))
gfed = xr.open_dataset("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc")
yg = np.array([t.year for t in gfed.time.values])
gm = (yg >= 2001) & (yg <= 2010)
gfed_tr = (gfed["burntArea"].values[gm] / 100.0).astype(np.float32)
lat = ds["lat"].values
cos_lat = np.cos(np.deg2rad(lat)).astype(np.float32)
land_fire = (gfed_tr > 0).any(axis=0)
w2 = (cos_lat[:, None] * land_fire).astype(np.float32)
for tag, path in [("(a) fire eq", "models/C-gfed5/params.gfed5.json"),
                  ("(b) continent eq", "models/C-gfed5-continent/params.gfed5-continent.json"),
                  ("(c) firetype eq", "models/C-gfed5-firetype/params.gfed5-firetype.json")]:
    p = json.load(open(path))["params"]
    pred = predict_monthly(d_tr, p)
    b, r, s, sp = four_scores(pred, gfed_tr, w2)
    o = (2*b + 2*r + s + sp) / 6
    print(f"{tag:20s} std overall={o:.4f}  bias={b:.3f} rmse={r:.3f} seas={s:.3f} spat={sp:.3f}")
