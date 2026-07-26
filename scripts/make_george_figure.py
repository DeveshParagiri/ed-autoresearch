"""Side-by-side for George: the two candidate models from the same base.
  A = leaderboard (no curing, ILAMB 0.679)   B = coupling (+curing, ILAMB 0.654)
Top: GFED5 | A | B burned-area maps (house style). Bottom: regional burned area bars
(GFED vs A vs B) showing A misses the steppe/Australia and B fixes them.
"""
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs, cartopy.feature as cfeat

def ann_pct(p, pct=False):
    d=xr.open_dataset(p); a=np.nan_to_num(d["burntArea"].values.astype(float)); d.close()
    if pct: a/=100.0
    return a[:192].reshape(16,12,360,720).sum(1).mean(0)*100.0
gf=ann_pct("ilamb_ref_official/DATA/burntArea/GFED5/burntArea.nc",pct=True)
A =ann_pct("ilamb/MODELS_GEORGE/ED-ModelC-A-leaderboard/burntArea.nc")
B =ann_pct("ilamb/MODELS_GEORGE/ED-ModelC-B-coupling/burntArea.nc")
d=xr.open_dataset("ilamb/MODELS_GEORGE/ED-ModelC-A-leaderboard/burntArea.nc"); lat=d.lat.values; lon=d.lon.values; d.close()
land=A>0; gf=np.where(gf>0,gf,np.nan)
LON,LAT=np.meshgrid(lon,lat); R=6371e3; area=(R**2*np.deg2rad(0.5)**2*np.cos(np.deg2rad(lat)))[:,None]*np.ones((1,720))
def mha(a,b): m=(LON>=b[0])&(LON<=b[1])&(LAT>=b[2])&(LAT<=b[3]); return float((np.nan_to_num(a)/100.0*area*m).sum())/1e10
RB={"Africa":(-20,52,-36,18),"S.Amer":(-82,-34,-56,14),"Boreal":(40,180,48,78),"India+SEA":(60,150,-11,35),"N.Amer":(-168,-52,14,74),"Australia":(112,154,-44,-10),"Steppe":(48,90,45,56)}

fig=plt.figure(figsize=(15,9))
gs=gridspec.GridSpec(2,3,height_ratios=[1.25,1.0],hspace=0.18,wspace=0.08)
def mapax(i,data,title):
    ax=fig.add_subplot(gs[0,i],projection=ccrs.Robinson()); ax.set_global()
    ax.coastlines(linewidth=0.35,color="0.3"); ax.add_feature(cfeat.BORDERS,linewidth=0.15,edgecolor="0.6")
    im=ax.pcolormesh(lon,lat,np.where(np.nan_to_num(data)>0,data,np.nan),transform=ccrs.PlateCarree(),cmap="YlOrRd",vmin=0,vmax=15,shading="auto")
    ax.set_title(title,fontsize=11); return im
mapax(0,gf,"GFED5 observed")
mapax(1,A,"F  Regional-GDP model\nILAMB 0.679  (best score)")
im=mapax(2,B,"G  = F + curing\nILAMB 0.654  (steppe fixed)")
cax=fig.add_axes([0.35,0.53,0.32,0.017]); cb=fig.colorbar(im,cax=cax,orientation="horizontal")
cb.set_label("burned fraction (% / yr)",fontsize=9); cb.ax.tick_params(labelsize=8)

# regional bars
axb=fig.add_subplot(gs[1,:])
regs=list(RB); x=np.arange(len(regs)); wbar=0.26
gv=[mha(gf,RB[r]) for r in regs]; av=[mha(A,RB[r]) for r in regs]; bv=[mha(B,RB[r]) for r in regs]
axb.bar(x-wbar,gv,wbar,label="GFED5 (truth)",color="0.35")
axb.bar(x,av,wbar,label="F  regional-GDP (no curing)",color="#0072B2")
axb.bar(x+wbar,bv,wbar,label="G  = F + curing",color="#D55E00")
for xi,(g0,a0,b0) in enumerate(zip(gv,av,bv)):
    for dx,val,c in [(-wbar,g0,"0.35"),(0,a0,"#0072B2"),(wbar,b0,"#D55E00")]:
        axb.text(xi+dx,val+4,f"{val:.0f}",ha="center",va="bottom",fontsize=7,color=c)
axb.set_xticks(x); axb.set_xticklabels(regs,fontsize=10)
axb.set_ylabel("burned area (Mha / yr)"); axb.set_ylim(0,620)
axb.set_title("Regional burned area: F misses the steppe & Australia; G fixes them (small aggregate-score cost)",fontsize=11)
axb.legend(fontsize=9,frameon=False,ncol=3,loc="upper right")
for sp in ("top","right"): axb.spines[sp].set_visible(False)
_st=(mha(A,RB["Steppe"]),mha(B,RB["Steppe"])); _au=(mha(A,RB["Australia"]),mha(B,RB["Australia"]))
axb.annotate(f"steppe {_st[0]:.0f} -> {_st[1]:.0f}\nAustralia {_au[0]:.0f} -> {_au[1]:.0f}",xy=(6,75),fontsize=8.5,color="#D55E00",ha="center")

fig.suptitle("Two models, two purposes: aggregate ILAMB vs regional carbon fidelity",fontsize=14,y=0.98)
fig.savefig("modelF_vs_G_burned_area.png",dpi=150,bbox_inches="tight"); print("[fig] modelF_vs_G_burned_area.png")
print("regions:",{r:(round(mha(gf,RB[r])),round(mha(A,RB[r])),round(mha(B,RB[r]))) for r in regs})
