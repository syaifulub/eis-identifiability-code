"""EE-relevance analyses (synthetic spectra only):
 (1) design nomogram: detection limit vs f_min and effective error, tissue-like asymptote (Table S45 medians)
 (2) sweep time vs f_min for stepped-sine analyzers
 (3) transfer test on a battery-type circuit R0 + ZARC + Warburg + L"""
import numpy as np, json, sys
from scipy.optimize import nnls, minimize_scalar, brentq
LN10=np.log(10); Z95=1.959964
def stack(c): return np.r_[c.real,c.imag]
def jac(fun,p,h=1e-6):
    r0=fun(p); J=np.zeros((r0.size,p.size))
    for i in range(p.size):
        d=np.zeros_like(p); d[i]=h*max(1,abs(p[i])); J[:,i]=(fun(p+d)-fun(p-d))/(2*d[i])
    return J
def proj(J): return np.eye(J.shape[0])-J@np.linalg.pinv(J)

# ---------- generic model: Z = R0 + dR/(1+(jw tau)^a) + extra(w; ext) + jwL ----------
def make(extra_fun, n_extra):
    def Zfun(f,R0,Q,a,tau,ext,L):
        w=2*np.pi*f; dR=tau**a/Q
        return R0+dR/(1+(1j*w*tau)**a)+extra_fun(w,ext)+1j*w*L
    def Zasym(f,R0,Q,a,ext,L):
        w=2*np.pi*f; return R0+1/(Q*(1j*w)**a)+extra_fun(w,ext)+1j*w*L
    return Zfun,Zasym
def no_extra(w,ext): return 0*w
def warburg(w,ext): return ext[0]/np.sqrt(1j*w)

def criterion(f,Zfun,Zasym,R0,Q,a,ext,L,fc,sig):
    tau=1/(2*np.pi*fc); Zo=Zfun(f,R0,Q,a,tau,ext,L); W=np.abs(Zo); w=2*np.pi*f
    ne=len(ext)
    def r_nuis(p):
        return stack((Zasym(f,10**p[0],10**p[1],p[2],10**p[3:3+ne],10**p[3+ne])-Zo)/W)
    p=np.r_[np.log10(R0),np.log10(Q),a,np.log10(ext),np.log10(L)]
    J=jac(r_nuis,p); P=proj(J)
    wk=(1j*w)**(-a)/Q; ht=stack(wk/W*(1j*w/w.min())**(-a)); D=np.linalg.norm(P@ht)
    W8=2*Z95*sig/(a*LN10*D)*(f.min()/fc)**a
    Gam=(a*LN10*D*1.0/(2*Z95*sig))**(1/a)
    def r_full(q):
        return stack((Zfun(f,10**q[0],10**q[1],q[2],10**q[-1],10**q[3:3+ne],10**q[3+ne])-Zo)/W)
    q=np.r_[p,np.log10(tau)]; J5=jac(r_full,q); C=np.linalg.inv(J5.T@J5)
    Wlin=2*Z95*sig*np.sqrt(C[-1,-1])
    return dict(W8=W8,Wlin=Wlin,Gamma=Gam,D=D,sinth=D/np.linalg.norm(ht),r=(f.min()/fc)**(-a))

def exp_profile(f,Zfun,R0,Q,a,ext,L,fc,sig,basis_extra,lo,hi,n=321):
    tau0=1/(2*np.pi*fc); Zo=Zfun(f,R0,Q,a,tau0,ext,L); W=np.abs(Zo); w=2*np.pi*f; b=stack(Zo/W)
    def rss(lt):
        t=10**lt
        def inner(al):
            cols=[stack(np.ones_like(f)/W),stack(1/(1+(1j*w*t)**al)/W),stack(1j*w/W)]+[stack(be(w)/W) for be in basis_extra]
            x,res=nnls(np.column_stack(cols),b); return res**2
        return minimize_scalar(inner,bounds=(0.3,1.0),method='bounded',options={'xatol':1e-6}).fun
    g=np.linspace(lo,hi,n); L_=np.array([rss(x) for x in g])/sig**2; L_-=L_.min()
    ins=g[L_<=3.84]; op=bool(L_[0]<=3.84 or L_[-1]<=3.84)
    return dict(width=float(ins.max()-ins.min()),open=op)

out={}
# (1) nomogram for the tissue-like asymptote (no extra element), bands f_min..100 kHz, N=10 log per band
Zt,Zta=make(no_extra,0)
tis=dict(R0=2211.,Q=7.4e-8,a=0.69,ext=np.array([]),L=1.14e-3)
def crit_t(f,fc,sig): 
    p=dict(tis); 
    return criterion(f,Zt,Zta,p['R0'],p['Q'],p['a'],np.array([1.0])[:0] if False else np.array([]),p['L'],fc,sig)
# handle empty ext: rewrite criterion call with ne=0
def crit_tissue(f,fc,sig):
    tau=1/(2*np.pi*fc); a=tis['a']; Q=tis['Q']
    Zo=Zt(f,tis['R0'],Q,a,tau,[],tis['L']); W=np.abs(Zo); w=2*np.pi*f
    def rn(p): return stack((Zta(f,10**p[0],10**p[1],p[2],[],10**p[3])-Zo)/W)
    p=np.r_[np.log10(tis['R0']),np.log10(Q),a,np.log10(tis['L'])]; J=jac(rn,p); P=proj(J)
    ht=stack(((1j*w)**(-a)/Q)/W*(1j*w/w.min())**(-a)); D=np.linalg.norm(P@ht)
    def rf(q): return stack((Zt(f,10**q[0],10**q[1],q[2],10**q[4],[],10**q[3])-Zo)/W)
    J5=jac(rf,np.r_[p,np.log10(tau)]); C=np.linalg.inv(J5.T@J5)
    return D, 2*Z95*sig*np.sqrt(C[4,4])
fmins=np.logspace(np.log10(100),np.log10(20e3),25); sigs=np.logspace(np.log10(0.002),np.log10(0.05),25)
DL8=np.zeros((len(sigs),len(fmins))); DLlin=np.zeros_like(DL8)
a=tis['a']
for j,fm in enumerate(fmins):
    f=np.logspace(np.log10(fm),5,10)
    # D independent of fc and sigma: compute once (fc arbitrary below band)
    D,_=crit_tissue(f,fm/5,0.01)
    for i,sg in enumerate(sigs):
        Gam=(a*LN10*D/(2*Z95*sg))**(1/a); DL8[i,j]=fm/Gam
        # exact linearized detection limit: solve Wlin(fc)=1 decade for fc<fmin
        g=lambda lfc: crit_tissue(f,10**lfc,sg)[1]-1.0
        lo,hi=np.log10(fm)-4,np.log10(fm)-1e-3
        try: DLlin[i,j]=10**brentq(g,lo,hi,xtol=1e-3) if g(lo)*g(hi)<0 else (np.nan)
        except Exception: DLlin[i,j]=np.nan
out['nomogram']=dict(fmins=fmins.tolist(),sigs=sigs.tolist(),DL8=DL8.tolist(),DLlin=DLlin.tolist())
# instrument point check (10 kHz, linear 10 points, sigma 0.018)
fin=np.arange(10e3,100.1e3,10e3); D,_=crit_tissue(fin,2e3,0.018); Gam=(a*LN10*D/(2*Z95*0.018))**(1/a)
out['instrument']=dict(Gamma=Gam,DL8=1e4/Gam)
# (2) sweep time: stepped sine, N log-spaced from f_min to 100 kHz, n_c cycles + n_s settling cycles per frequency
st=[]
for fm in [10e3,3e3,1e3,300.]:
    for N in [10,20]:
        f=np.logspace(np.log10(fm),5,N)
        for nc in [5,10]:
            st.append(dict(fmin=fm,N=N,cycles=nc,T_ms=1e3*nc*np.sum(1/f),share_lowest=float((1/f[0])/np.sum(1/f))))
out['sweep_time']=st
# (3) battery-type circuit: R0 + ZARC + Warburg + L ; band f_min..10 kHz, 10 points/decade
Zb,Zba=make(warburg,1)
bat=dict(R0=0.020,Q=None,a=0.80,ext=np.array([0.002]),L=0.5e-6)
fc_b=20.0; Rz=0.015; tau_b=1/(2*np.pi*fc_b); Qb=tau_b**bat['a']/Rz
rows=[]
for fm in [200.,100.,50.,20.,10.]:
    n=int(round(10*np.log10(1e4/fm)))+1; f=np.logspace(np.log10(fm),4,n)
    for sg in [0.002,0.005]:
        c=criterion(f,Zb,Zba,bat['R0'],Qb,bat['a'],bat['ext'],bat['L'],fc_b,sg)
        pr=exp_profile(f,Zb,bat['R0'],Qb,bat['a'],bat['ext'],bat['L'],fc_b,sg,[lambda w: 1/np.sqrt(1j*w)],lo=np.log10(tau_b)-4,hi=np.log10(tau_b)+4)
        rows.append(dict(fmin=fm,N=n,sigma=sg,r=c['r'],sinth=c['sinth'],W8=c['W8'],Wlin=c['Wlin'],prof=pr['width'],open=pr['open'],
                         id8=bool(c['W8']<1),idlin=bool(c['Wlin']<1),idprof=bool((not pr['open']) and pr['width']<1),DL8=fm/c['Gamma']))
out['battery']=rows
json.dump(out,open(sys.argv[1],'w'),default=float,indent=1)
print('instrument',out['instrument'])
for r in st: print(r)
for r in rows: print({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
