"""Show Option 3 removes the seams: parameter field (hard tiles vs smooth gradient)
and BA (hard vs smooth). Output PNG path from argv[1]."""
import json, sys
from pathlib import Path
import numpy as np, xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature

OUT = sys.argv[1]
REPO = Path("."); SIGMA = 4.0
REGION_BOX = {"Africa": (-20,52,-36,18), "Boreal": (40,180,48,78),
              "S.America": (-82,-34,-56,14), "SEAsia": (60,150,-11,30), "Europe": (-12,40,36,72)}
RP = {r: json.load(open(REPO/"models/C"/f))["params"] for r,f in
      {"Africa":"params.africafuel.json","Boreal":"params.boreal.json","S.America":"params.samerica.json",
       "SEAsia":"params.seasia.json","Europe":"params.europe.json"}.items()}
FB = json.load(open(REPO/"models/C/params.spatial.k1.json"))["params"]
lat1=-89.5+np.arange(180); lon1=-179.5+np.arange(360); LON,LAT=np.meshgrid(lon1,lat1)
assigned=np.zeros((180,360),bool); reg=np.full((180,360),"fb",dtype=object)
for r,b in REGION_BOX.items():
    box=(LON>=b[0])&(LON<=b[1])&(LAT>=b[2])&(LAT<=b[3])&~assigned; reg[box]=r; assigned|=box
KEY="fire_exp"
hard=np.full((180,360),FB[KEY],float)
for r in REGION_BOX: hard[reg==r]=RP[r][KEY]
smooth=np.exp(gaussian_filter(np.log(hard),SIGMA,mode="nearest"))

def ba(nc):
    ds=xr.open_dataset(nc); a=ds["burntArea"]; m=np.nan_to_num(a.isel(time=slice(0,192)).values).mean(0)
    lat=a.lat.values; lon=a.lon.values; ds.close(); return lat,lon,m
la_h,lo_h,ba_h=ba("ilamb/MODELS_SMOOTH/ED-ModelC-hard/burntArea.nc")
la_s,lo_s,ba_s=ba("ilamb/MODELS_SMOOTH/ED-ModelC-smooth/burntArea.nc")
vmax=float(np.percentile(ba_h[ba_h>0],99)); bcmap=plt.get_cmap("YlOrRd").copy(); bcmap.set_under("white")

fig,axes=plt.subplots(2,2,figsize=(15,8),subplot_kw={"projection":ccrs.PlateCarree()})
def draw(ax,lon,lat,f,title,cmap,vmin,vmax,under=None):
    if under: cmap=cmap;
    pcm=ax.pcolormesh(lon,lat,f,transform=ccrs.PlateCarree(),cmap=cmap,vmin=vmin,vmax=vmax,shading="auto")
    ax.add_feature(cfeature.COASTLINE,lw=0.4,edgecolor="#333"); ax.set_global()
    ax.set_title(title,fontsize=11,weight="bold"); return pcm
# top: parameter field (the seams)
p1=draw(axes[0,0],lon1,lat1,hard,"fire_exp parameter  -  HARD borders (blocky tiles)","viridis",1,3)
p2=draw(axes[0,1],lon1,lat1,smooth,"fire_exp parameter  -  SMOOTHED (no seams)","viridis",1,3)
fig.colorbar(p2,ax=axes[0,:],orientation="vertical",fraction=0.02,pad=0.01,aspect=30).set_label("fire_exp")
# bottom: BA
mh=np.ma.masked_less_equal(ba_h,1e-6); ms=np.ma.masked_less_equal(ba_s,1e-6)
b1=draw(axes[1,0],lo_h,la_h,mh,"Burned area  -  HARD (0.6649)",bcmap,1e-6,vmax)
b2=draw(axes[1,1],lo_s,la_s,ms,"Burned area  -  SMOOTH (0.6641)",bcmap,1e-6,vmax)
fig.colorbar(b2,ax=axes[1,:],orientation="vertical",fraction=0.02,pad=0.01,aspect=30).set_label("mean monthly burned fraction")
fig.suptitle("Option 3 prototype: smooth regional parameters remove the seams at ~zero score cost",
             fontsize=13,weight="bold",y=0.98)
fig.savefig(OUT,dpi=170,bbox_inches="tight",facecolor="white")
print(f"wrote {OUT}")
