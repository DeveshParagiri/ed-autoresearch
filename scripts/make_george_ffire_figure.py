"""fFire side-by-side for George: GFED5 | A leaderboard | B coupling(+curing).
Emissions maps (g C/m2/yr) + regional emissions bars (TgC/yr). Each model on its own betas.
"""
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import cartopy.crs as ccrs, cartopy.feature as cfeat
SEC_YR=86400*365

def load(p):
    d=xr.open_dataset(p); v=d["fFire"]
    lat=d["lat"].values if "lat" in d else d["latitude"].values
    lon=d["lon"].values if "lon" in d else d["longitude"].values
    a=np.nan_to_num(v.values.astype(float))
    n=a.shape[0]//12; mean=a[:n*12].reshape(n,12,*a.shape[1:]).mean(0).mean(0)   # kg/m2/s time-mean
    if lat[0]>lat[-1]: mean=mean[::-1]; lat=lat[::-1]
    d.close(); return mean, lat, lon
gf,latg,long=load("ilamb_ref_official/DATA/fFire/GFED5/fFire.nc")
A,latA,lonA=load("ilamb/MODELS_GEORGE_FFIRE/ED-ModelC-A-leaderboard/fFire.nc")
B,latB,lonB=load("ilamb/MODELS_GEORGE_FFIRE/ED-ModelC-B-coupling/fFire.nc")
gC=lambda m:m*SEC_YR*1000.0                      # kg/m2/s -> g C/m2/yr
def emis(m,lat,lon,box=None):                    # TgC/yr over box
    R=6371e3; area=(R**2*np.deg2rad(abs(lat[1]-lat[0]))**2*np.cos(np.deg2rad(lat)))[:,None]*np.ones((1,len(lon)))
    LON,LAT=np.meshgrid(lon,lat); msk=np.ones_like(area,bool) if box is None else ((LON>=box[0])&(LON<=box[1])&(LAT>=box[2])&(LAT<=box[3]))
    return float((np.nan_to_num(m)*area*msk).sum()*SEC_YR)/1e9
RB={"Africa":(-20,52,-36,18),"S.Amer":(-82,-34,-56,14),"Boreal":(40,180,48,78),"India+SEA":(60,150,-11,35),"N.Amer":(-168,-52,14,74),"Australia":(112,154,-44,-10),"Steppe":(48,90,45,56)}

fig=plt.figure(figsize=(15,9)); gs=gridspec.GridSpec(2,3,height_ratios=[1.25,1.0],hspace=0.18,wspace=0.08)
def mapax(i,m,lat,lon,title):
    ax=fig.add_subplot(gs[0,i],projection=ccrs.Robinson()); ax.set_global()
    ax.coastlines(linewidth=0.35,color="0.3"); ax.add_feature(cfeat.BORDERS,linewidth=0.15,edgecolor="0.6")
    d=gC(m); im=ax.pcolormesh(lon,lat,np.where(d>0.1,d,np.nan),transform=ccrs.PlateCarree(),cmap="OrRd",vmin=0,vmax=200,shading="auto")
    ax.set_title(title,fontsize=11); return im
mapax(0,gf,latg,long,"GFED5 emissions")
mapax(1,A,latA,lonA,"F  Regional-GDP\nfFire ILAMB 0.667  (3.4 PgC/yr)")
im=mapax(2,B,latB,lonB,"G  = F + curing\nfFire ILAMB 0.656  (3.4 PgC/yr)")
cax=fig.add_axes([0.35,0.53,0.32,0.017]); cb=fig.colorbar(im,cax=cax,orientation="horizontal"); cb.set_label("fire emissions (g C / m2 / yr)",fontsize=9); cb.ax.tick_params(labelsize=8)

axb=fig.add_subplot(gs[1,:]); regs=list(RB); x=np.arange(len(regs)); wb=0.26
gv=[emis(gf,latg,long,RB[r]) for r in regs]; av=[emis(A,latA,lonA,RB[r]) for r in regs]; bv=[emis(B,latB,lonB,RB[r]) for r in regs]
axb.bar(x-wb,gv,wb,label="GFED5 (truth)",color="0.35"); axb.bar(x,av,wb,label="F  regional-GDP",color="#0072B2"); axb.bar(x+wb,bv,wb,label="G  = F + curing",color="#D55E00")
for xi,(g0,a0,b0) in enumerate(zip(gv,av,bv)):
    for dx,val,c in [(-wb,g0,"0.35"),(0,a0,"#0072B2"),(wb,b0,"#D55E00")]:
        axb.text(xi+dx,val+8,f"{val:.0f}",ha="center",va="bottom",fontsize=7,color=c)
axb.set_xticks(x); axb.set_xticklabels(regs,fontsize=10); axb.set_ylabel("fire emissions (Tg C / yr)")
axb.set_title("Regional fire emissions: curing adds burned AREA in grassland but little CARBON (low biomass)",fontsize=11)
axb.legend(fontsize=9,frameon=False,ncol=3,loc="upper right")
for sp in ("top","right"): axb.spines[sp].set_visible(False)
fig.suptitle("Fire emissions (fFire): leaderboard vs coupling+curing",fontsize=14,y=0.98)
fig.savefig("george_two_models_ffire.png",dpi=150,bbox_inches="tight"); print("[fig] george_two_models_ffire.png")
print("regional TgC/yr (GFED,A,B):",{r:(round(emis(gf,latg,long,RB[r])),round(emis(A,latA,lonA,RB[r])),round(emis(B,latB,lonB,RB[r]))) for r in regs})
