"""Check whether the narrow above-band profiles are due to the positivity bounds (R_inf, dR, L >= 0):
expected profile of log10 tau with and without bounds, against the exact linearized width."""
import numpy as np, sys, json
from scipy.optimize import nnls, minimize_scalar
src=open(sys.argv[1]).read().split('rng=np.random.default_rng(5)')[0]; ns={}; exec(src,ns)
cole=ns['cole']; stack=ns['stack']; Wlin=ns['Wlin']
def prof(w,Z0,sig,lt0,bounded):
    W=np.abs(Z0); b=stack(Z0/W)
    def rss(lt):
        def inner(al):
            A=np.column_stack([stack(np.ones_like(w)/W),stack(1/(1+(1j*w*10**lt)**al)/W),stack(1j*w/W)])
            if bounded: x,res=nnls(A,b); return res**2
            x,*_=np.linalg.lstsq(A,b,rcond=None); r=A@x-b; return r@r
        return minimize_scalar(inner,bounds=(0.3,1.0),method='bounded',options={'xatol':1e-7}).fun
    g=np.arange(lt0-4,lt0+4.001,0.025); L=np.array([rss(x) for x in g])/sig**2; L-=L.min(); ins=g[L<=3.84]
    return float(ins.max()-ins.min()), bool(L[0]>3.84 and L[-1]>3.84)
rng=np.random.default_rng(9); out=[]
for k in range(12):
    Ri=10**rng.uniform(np.log10(500),np.log10(5000)); dR=Ri*rng.uniform(1,3); al=rng.uniform(0.6,0.95); L=rng.uniform(0.4e-3,2.5e-3)
    fmax=100e3; f=np.logspace(3,5,20); w=2*np.pi*f
    for ratio in [1.5,3]:
        fc=fmax*ratio; tau=1/(2*np.pi*fc); Z0=cole(w,Ri,dR,tau,al,L); sig=0.005
        wl=Wlin(w,Ri,dR,tau,al,L,sig,np.abs(Z0)); wb,cb=prof(w,Z0,sig,np.log10(tau),True); wu,cu=prof(w,Z0,sig,np.log10(tau),False)
        out.append(dict(ratio=ratio,Wlin=wl,Wbounded=wb,closed_b=cb,Wunbounded=wu,closed_u=cu)); print(ratio,round(wl,2),round(wb,2),cb,round(wu,2),cu,flush=True)
json.dump(out,open(sys.argv[2],'w'))
