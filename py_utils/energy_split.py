import numpy as np
import matplotlib.pyplot as plt
import glob, os, struct

MAGIC=b"RH3TBIN1"
HEADER=struct.Struct("<8sIIQQd")
files=sorted(glob.glob("data_new/solution_*.bin"))
if not files: raise RuntimeError("No data/solution_*.bin files found")

rows=[]
print(f"{'file':>18} {'t':>10} {'max|Ee-Ei|':>14} {'max|Ee-Er|':>14} {'max|Ei-Er|':>14} {'Ee-Er location':>24}")

for fn in files:
    with open(fn,"rb") as f: raw=f.read(HEADER.size)
    magic,version,nvar,nx,ny,t=HEADER.unpack(raw)
    if magic!=MAGIC or version!=1 or nvar!=8: raise RuntimeError(f"Bad format: {fn}")
    expected=HEADER.size+nx*ny*8*8
    if os.path.getsize(fn)!=expected: raise RuntimeError(f"Incomplete file: {fn}")
    a=np.memmap(fn,dtype="<f8",mode="r",offset=HEADER.size,shape=(ny,nx,8))
    x,y=a[:,:,0],a[:,:,1]
    ee,ei,er=a[:,:,5],a[:,:,6],a[:,:,7]
    valid=np.isfinite(ee)&np.isfinite(ei)&np.isfinite(er)
    d1=np.abs(ee-ei); d2=np.abs(ee-er); d3=np.abs(ei-er)
    def stats(d):
        m=np.where(valid,d,-np.inf)
        ij=np.unravel_index(int(np.argmax(m)),d.shape)
        return float(d[ij]),float(np.mean(d[valid])),float(np.sqrt(np.mean(d[valid]**2))),ij
    m1,a1,r1,i1=stats(d1); m2,a2,r2,i2=stats(d2); m3,a3,r3,i3=stats(d3)
    rows.append([t,m1,m2,m3,a1,a2,a3,r1,r2,r3,float(x[i2]),float(y[i2])])
    print(f"{os.path.basename(fn):>18} {t:10.4e} {m1:14.6e} {m2:14.6e} {m3:14.6e} ({x[i2]:.5f},{y[i2]:.5f})")

rows=np.asarray(rows)
header="time,max_Ee_Ei,max_Ee_Er,max_Ei_Er,mean_Ee_Ei,mean_Ee_Er,mean_Ei_Er,rms_Ee_Ei,rms_Ee_Er,rms_Ei_Er,x_max_Ee_Er,y_max_Ee_Er"
np.savetxt("energy_split_history.csv",rows,delimiter=",",header=header,comments="",fmt="%.16e")

fig,ax=plt.subplots(figsize=(9,5))
ax.semilogy(rows[:,0],rows[:,1],marker="o",label=r"$\max|E_e-E_i|$")
ax.semilogy(rows[:,0],rows[:,2],marker="o",label=r"$\max|E_e-E_r|$")
ax.semilogy(rows[:,0],rows[:,3],marker="o",label=r"$\max|E_i-E_r|$")
ax.set_xlabel("time"); ax.set_ylabel("maximum absolute energy split")
ax.set_title("Three-energy symmetry diagnostic"); ax.grid(True,which="both",alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig("energy_split_history.png",dpi=180)
plt.show()
print("Saved energy_split_history.csv and energy_split_history.png")