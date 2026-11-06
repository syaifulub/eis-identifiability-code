"""Monte Carlo check of the asymptotic (Fisher) widths and of the plug-in detection limit.
Synthetic Cole+L spectra with the median KK-consistent asymptote; instrument band (10-100 kHz, N=10);
iid relative Gaussian errors. Fits: variable projection over (log10 tau, alpha) with NNLS for (Rinf, dR, L)."""
import numpy as np, json, sys
from scipy.optimize import nnls, minimize_scalar, minimize
rng=np.random.default_rng(20261029)
LN10=np.log(10); Z95=1.959964
f=np.arange(10e3,100.1e3,10e3); w=2*np.pi*f; N=f.size
Rinf,Q,a,L=2211.,7.4e-8,0.69,1.14e-3
def stack(c): return np.r_[c.real,c.imag]
def cole(Ri,dR,tau,al,Li): return Ri+dR/(1+(1j*w*tau)**al)+1j*w*Li
def jac(fun,p,h=1e-6):
    r0=fun(p); J=np.zeros((r0.size,p.size))
    for i in range(p.size):
        d=np.zeros_like(p); d[i]=h*max(1,abs(p[i])); J[:,i]=(fun(p+d)-fun(p-d))/(2*d[i])
    return J
def proj(J): return np.eye(J.shape[0])-J@np.linalg.pinv(J)

def fit_cole(Zo,lt0):
    W=np.abs(Zo); b=stack(Zo/W)
    def inner(lt,al):
        t=10**lt; A=np.column_stack([stack(np.ones(N)/W),stack(1/(1+(1j*w*t)**al)/W),stack(1j*w/W)])
        x,res=nnls(A,b); return res**2,x
    def prof(lt):
        o=minimize_scalar(lambda al: inner(lt,al)[0],bounds=(0.3,1.0),method='bounded',options={'xatol':1e-7}); return o.fun,o.x
    grid=np.linspace(-7,1,81); vals=[prof(g)[0] for g in grid]; g0=grid[int(np.argmin(vals))]
    o=minimize(lambda p: inner(p[0],p[1])[0],[g0,prof(g0)[1]],method='Nelder-Mead',options={'xatol':1e-7,'fatol':1e-14,'maxiter':4000})
    lt,al=o.x; rss,x=inner(lt,al); return lt,al,rss,x
def fit_rcpel(Zo):
    W=np.abs(Zo); b=stack(Zo/W)
    def inner(al):
        A=np.column_stack([stack(np.ones(N)/W),stack((1j*w)**(-al)/W),stack(1j*w/W)]); x,res=nnls(A,b); return res**2,x
    o=minimize_scalar(lambda al: inner(al)[0],bounds=(0.3,1.0),method='bounded',options={'xatol':1e-8}); rss,x=inner(o.x)
    return x[0],1/max(x[1],1e-300),o.x,x[2]
def D_of(Ri,Qv,al,Li,Zo):
    W=np.abs(Zo)
    def rn(p): return stack((10**p[0]+1/(10**p[1]*(1j*w)**p[2])+1j*w*10**p[3]-Zo)/W)
    J=jac(rn,np.array([np.log10(Ri),np.log10(Qv),al,np.log10(Li)])); P=proj(J)
    ht=stack(((1j*w)**(-al)/Qv)/W*(1j*w/w[0])**(-al)); return np.linalg.norm(P@ht)

out=[]
for r,sig in [(0.3,0.018),(0.5,0.018),(0.5,0.008),(0.7,0.008)]:
    fc=f[0]*r**(1/a); tau=1/(2*np.pi*fc); dR=tau**a/Q; Z0=cole(Rinf,dR,tau,a,L); W0=np.abs(Z0)
    # predicted widths: Eq. (8) and exact linearized (Schur complement, parameterization (Rinf,Q,alpha,L,tau))
    D=D_of(Rinf,Q,a,L,Z0); W8=2*Z95*sig/(a*LN10*D)*(f[0]/fc)**a
    def r5(p): return stack((cole(10**p[0],(10**p[4])**p[2]/10**p[1],10**p[4],p[2],10**p[3])-Z0)/W0)
    p5=np.array([np.log10(Rinf),np.log10(Q),a,np.log10(L),np.log10(tau)]); J5=jac(r5,p5)
    Pn=proj(J5[:,:4]); s=J5[:,4]; Wschur=2*Z95*sig/np.linalg.norm(Pn@s)
    Wwald=2*Z95*sig*np.sqrt(np.linalg.inv(J5.T@J5)[4,4])
    est=[];G=[];Gtrue=(a*LN10*D/(2*Z95*sig))**(1/a)
    for k in range(300):
        e=sig*rng.standard_normal(2*N); Zo=Z0+W0*(e[:N]+1j*e[N:])
        lt,al,rss,x=fit_cole(Zo,np.log10(tau)); est.append(lt)
        Ri,Qh,ah,Lh=fit_rcpel(Zo); sh=np.sqrt(rss/(2*N-5)); Dh=D_of(Ri,Qh,ah,Lh,Zo)
        G.append((ah*LN10*Dh/(2*Z95*sh))**(1/ah))
    est=np.array(est); G=np.array(G); err=est-np.log10(tau)
    sd=np.std(err,ddof=1); mad=1.4826*np.median(np.abs(err-np.median(err)))
    cov8=np.mean(np.abs(err)<=W8/2); covl=np.mean(np.abs(err)<=Wschur/2)
    lg=np.log10(f[0]/G)
    out.append(dict(r=r,sigma=sig,fc=fc,W8=W8,Wschur=Wschur,Wwald=Wwald,emp_2zsd=2*Z95*sd,emp_2zmad=2*Z95*mad,
        bias=float(np.median(err)),cov8=cov8,covlin=covl,at_bound=float(np.mean((est<-6.95)|(est>0.95))),
        DL_true=f[0]/Gtrue,DL_med=float(10**np.median(lg)),DL_q=[float(10**np.percentile(lg,5)),float(10**np.percentile(lg,95))],DL_sd_dec=float(np.std(lg,ddof=1))))
    print(out[-1])
json.dump(out,open(sys.argv[1],'w'),indent=1,default=float)
