# -*- coding: utf-8 -*-
"""Apophis CAD (radar mesh) renderer: lit shape model + coverage-colored variant."""
import os, numpy as np

def _data_dir():
    """Locate nominal_v2/data (radar_mesh.npz etc.) portably on any machine."""
    d=os.environ.get("ARISE_NOMINAL_DATA","")
    if d and os.path.isfile(os.path.join(d,"radar_mesh.npz")): return d
    try:
        import config_redock as C
        c=os.path.join(C.ANIM_DIR,"Designs","D_strobe_fullbody","nominal_v2","data")
        if os.path.isfile(os.path.join(c,"radar_mesh.npz")): return c
    except Exception: pass
    d=os.path.dirname(os.path.abspath(__file__))
    for _ in range(9):
        for sub in ("Designs/D_strobe_fullbody/nominal_v2/data","nominal_v2/data","data"):
            c=os.path.join(d,sub)
            if os.path.isfile(os.path.join(c,"radar_mesh.npz")): return c
        d=os.path.dirname(d)
    raise FileNotFoundError("radar_mesh.npz not found — set env ARISE_NOMINAL_DATA to nominal_v2/data")
def load_apophis():
    NM=_data_dir()
    m=np.load(NM+"/radar_mesh.npz",allow_pickle=True)
    V=m["vertices"].astype(float); F=m["faces"].astype(int); N=m["normals"].astype(float)
    V=V-V.mean(0)                       # center
    try:
        at=np.load(NM+"/mc/draw_0000/apophis/attitude_truth.npz",allow_pickle=True)
        sun=at["sun_hat"].astype(float); Rser=at["R_series"]
    except Exception:
        sun=np.array([1,0.3,0.2]); Rser=None
    try:
        cov=np.load(NM+"/mc/draw_0000/coverage/facet_stereo.npz",allow_pickle=True)["category"].astype(int)
    except Exception:
        cov=np.zeros(len(F),int)
    return dict(V=V,F=F,N=N,sun=sun/np.linalg.norm(sun),Rser=Rser,cov=cov,scale=np.linalg.norm(V,axis=1).max())

def apophis_polys(ap, R, mode="lit", zorder=2, stride=1):
    """Return (Poly3DCollection-ready verts list, facecolors) for the rotated mesh."""
    V=ap["V"]@R.T; N=(ap["N"]@R.T)[::stride]
    tris=V[ap["F"][::stride]]            # (nf,3,3)
    shade=np.clip(N@ap["sun"],0,1)      # Lambertian
    amb=0.18
    if mode=="coverage":
        # emerald=covered@0.15m(0), cyan(1), amber(2), dark(3/none)
        base=np.array([[0.15,0.85,0.45],[0.20,0.75,0.95],[0.98,0.78,0.25],[0.28,0.30,0.36]])
        idx=np.clip(ap["cov"][::stride],0,3); col=base[idx]
        fc=col*(amb+ (1-amb)*shade[:,None])
    else:
        rock=np.array([0.62,0.57,0.50])   # warm regolith grey
        fc=rock*(amb+(1-amb)*shade[:,None])
    fc=np.clip(fc,0,1)
    return tris, fc

def draw_apophis(ax, ap, R, center=(0,0,0), s=1.0, mode="lit", alpha=1.0, stride=1):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tris,fc=apophis_polys(ap,R,mode,stride=stride)
    tris=tris*s + np.asarray(center)
    pc=Poly3DCollection(tris, facecolors=fc, edgecolors="none", linewidths=0, alpha=alpha, zorder=2)
    ax.add_collection3d(pc); return pc

if __name__=="__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ap=load_apophis(); print("loaded Apophis: scale %.0f m, cov cats"%ap["scale"], {int(c):int((ap["cov"]==c).sum()) for c in np.unique(ap["cov"])})
    R0=ap["Rser"][len(ap["Rser"])//2] if ap["Rser"] is not None else np.eye(3)
    fig=plt.figure(figsize=(12,6)); fig.patch.set_facecolor("#05070d")
    for i,mode in enumerate(["lit","coverage"]):
        ax=fig.add_subplot(1,2,i+1,projection="3d"); ax.set_facecolor("#05070d"); ax.set_axis_off()
        draw_apophis(ax,ap,R0,mode=mode)
        s=ap["scale"]*1.1; ax.set_xlim(-s,s);ax.set_ylim(-s,s);ax.set_zlim(-s,s)
        try: ax.set_box_aspect((1,1,1))
        except: pass
        ax.view_init(elev=18,azim=35+i*0); ax.set_title(mode,color="#8fb7ff")
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),"_apophis_cad.png"),facecolor="#05070d",dpi=90)
    print("saved _apophis_cad.png")
