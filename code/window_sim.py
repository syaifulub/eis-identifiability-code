"""Two-sided identifiability window (dispersion above the band).
Identity: dR/(1+v) = dR - dR v + dR v^2/(1+v), v=(j w tau)^alpha, small when f_c >> f_max. The low-frequency asymptote is
R0 - K (j w)^alpha + j w L (R0 = R_inf + dR, K = dR tau^alpha), and tau enters through dR v^2 = K tau^alpha (j w)^(2 alpha),
which is the mirror image of the below-band case. The closed form is
  W'95 = 2 z sigma / (alpha ln10 D') (f_c/f_max)^alpha,  D' = ||P' h'||,  h'_k = (K (j w_k)^alpha/|Z_k|) (j w_k/w_max)^alpha,
P' projecting out the Jacobian of the asymptote w.r.t. (log10 R0, log10 K, alpha, log10 L); upper detection limit
Gamma' f_max with Gamma' = (alpha ln10 D' W*/(2 z sigma))^(1/alpha).
Validation on synthetic Cole+L spectra (tissue-like): closed form, exact linearized width W_lin (Eq. A6) and expected
profile likelihood (noise-free, threshold 3.84 sigma^2), with and without the positivity bounds of the linear
coefficients (NNLS vs unconstrained), for bands below the dispersion."""
import numpy as np, json, sys
from scipy.optimize import nnls, minimize_scalar
LN10=np.log(10); Z95=1.959964
def stack(c): return np.r_[c.real,c.imag]
def jac(fun,p,h=1e-6):
    r0=fun(p); J=np.zeros((r0.size,p.size))
    for i in range(p.size):
        d=np.zeros_like(p); d[i]=h*max(1,abs(p[i])); J[:,i]=(fun(p+d)-fun(p-d))/(2*d[i])
    return J
def proj(J): return np.eye(J.shape[0])-J@np.linalg.pinv(J)
def cole(w,Ri,dR,tau,al,L): return Ri+dR/(1+(1j*w*tau)**al)+1j*w*L
def closed_above(w,R0,K,al,L,fc,sig,Wt):
    def rn(p): return stack((10**p[0]-10**p[1]*(1j*w)**p[2]+1j*w*10**p[3])/Wt)
    J=jac(rn,np.array([np.log10(R0),np.log10(K),al,np.log10(L)])); P=proj(J)
    h=stack((K*(1j*w)**al)/Wt*(1j*w/w.max())**al); D=np.linalg.norm(P@h)
    W=2*Z95*sig/(al*LN10*D)*(2*np.pi*fc/w.max())**al; G=(al*LN10*D/(2*Z95*sig))**(1/al)
    return W,G,D,D/np.linalg.norm(h)
def Wlin(w,Ri,dR,tau,al,L,sig,Wt):
    def r5(q): return stack(cole(w,10**q[0],10**q[1],10**q[2],q[3],10**q[4])/Wt)
    J=jac(r5,np.array([np.log10(Ri),np.log10(dR),np.log10(tau),al,np.log10(L)])); C=np.linalg.inv(J.T@J)
    return 2*Z95*sig*np.sqrt(C[2,2])
def exp_profile(w,Z0,sig,lt0,bounded=True):
    W=np.abs(Z0); b=stack(Z0/W)
    def rss(lt):
        def inner(al):
            A=np.column_stack([stack(np.ones_like(w)/W),stack(1/(1+(1j*w*10**lt)**al)/W),stack(1j*w/W)])
            if bounded: x,res=nnls(A,b); return res**2
            x,*_=np.linalg.lstsq(A,b,rcond=None); r=A@x-b; return r@r
        return minimize_scalar(inner,bounds=(0.3,1.0),method='bounded',options={'xatol':1e-7}).fun
    g=np.arange(lt0-4,lt0+4.001,0.025); L=np.array([rss(x) for x in g])/sig**2; L-=L.min()
    ins=g[L<=3.84]; closed=(L[0]>3.84) and (L[-1]>3.84)
    return float(ins.max()-ins.min()), bool(closed)
rng=np.random.default_rng(5); rows=[]
for k in range(25):
    Ri=10**rng.uniform(np.log10(500),np.log10(5000)); dR=Ri*rng.uniform(1,3); al=rng.uniform(0.6,0.95); L=rng.uniform(0.4e-3,2.5e-3)
    for fmax in [10e3,30e3,100e3]:
        fmin=fmax/100; f=np.logspace(np.log10(fmin),np.log10(fmax),20); w=2*np.pi*f
        for ratio in [1.5,3,10,30]:
            fc=fmax*ratio; tau=1/(2*np.pi*fc); Z0=cole(w,Ri,dR,tau,al,L); Wt=np.abs(Z0)
            for sig in [0.005,0.025]:
                W8,G,D,sinth=closed_above(w,Ri+dR,dR*tau**al,al,L,fc,sig,Wt)
                wl=Wlin(w,Ri,dR,tau,al,L,sig,Wt); wp,cl=exp_profile(w,Z0,sig,np.log10(tau)); wu,cu=exp_profile(w,Z0,sig,np.log10(tau),False)
                rows.append(dict(truth=k,fmax=fmax,ratio=ratio,sigma=sig,alpha=al,r=ratio**(-al),W8=W8,Wlin=wl,Wprof=wp,closed=cl,
                                 Gamma_up=G,sinth=sinth,id8=bool(W8<1),idlin=bool(wl<1),idprof=bool(cl and wp<1),Wprof_unb=wu,closed_unb=cu,idprof_unb=bool(cu and wu<1)))
    print(k,flush=True)
json.dump(rows,open(sys.argv[1],'w'),default=float); print('done')
