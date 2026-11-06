"""Prospective test on measured wide-band battery spectra (Zhang et al. 2020, Nature Communications 11:1706;
data Zenodo 10.5281/zenodo.3633835, 'EIS data.zip', CC BY 4.0). Commercial LCO/graphite coin cells, 60 frequencies
from 20 kHz to 0.02 Hz. Circuit: R0 + ZARC1 + ZARC2 + A_W/sqrt(jw) + jwL (NRMSE about 1.3%), where ZARC1 is the
low-frequency dispersion (f_c1 about 7-33 Hz) whose relaxation time tau1 is the target and ZARC2 (about 150-550 Hz),
the Warburg term and L are nuisance elements. Linear coefficients (R0, dR1, dR2, A_W, L) by NNLS; nonlinear
(log10 tau1, alpha1, log10 tau2, alpha2) by multistart TRF.
For each spectrum: full-band fit (reference); then the band is truncated from below (f_min = 1, 3, 10, 20, 30, 50, 100 Hz;
f_max = 20 kHz). On each truncated band, using only that band:
  observed : profile likelihood of log10 tau1 (nominal threshold 3.84; closed and narrower than one decade)
  closed   : closed form, Eqs. (8)-(9), with the asymptote R0 + 1/(Q1 (jw)^alpha1) + ZARC2 + A_W/sqrt(jw) + jwL implied by
             the fit on the band (Q1 = tau1^alpha1/dR1; a separate asymptote fit is degenerate because the CPE and
             Warburg power laws are nearly collinear on truncated bands)
  f_DL     : operational detection limit (exact linearized width of the full model along that asymptote)
  three-way rule; and whether the truncated-band interval contains the full-band tau1.
Usage: python battery_measured.py '<EIS data folder>' out.json [n_cycles_per_cell]"""
import numpy as np, json, sys, os, glob
from scipy.optimize import nnls, minimize_scalar, least_squares, brentq
LN10=np.log(10); Z95=1.959964; WSTAR=1.0; NPAR=9
LO=np.array([-7,0.3,-6,0.3]); HI=np.array([2,1,-2.6,1])  # ZARC2 restricted to f_c2 >= 63 Hz (full-band f_c2: 145-550 Hz) to prevent label switching
def stack(c): return np.r_[c.real,c.imag]
def zarc(w,lt,al): return 1/(1+(1j*w*10**lt)**al)
def design(w,p): return [np.ones_like(w,dtype=complex),zarc(w,p[0],p[1]),zarc(w,p[2],p[3]),1/np.sqrt(1j*w),1j*w]
def resid(w,Z,p,bounded=True):
    W=1/np.abs(Z); A=np.column_stack([stack(c*W) for c in design(w,p)]); b=stack(Z*W)
    if bounded: x,_=nnls(A,b)
    else: x,*_=np.linalg.lstsq(A,b,rcond=None)
    return A@x-b,x
def fit(w,Z,fixed_lt1=None,p0=None,bounded=True):
    """Global fit (grid of starts) or, with fixed_lt1, fit of (alpha1, lt2, alpha2) warm-started at p0."""
    if fixed_lt1 is None:
        starts=[]
        for lt1 in np.arange(-4,0.6,0.5):
            for lt2 in np.arange(-6,-2.59,0.5):
                if True:
                    for a in (0.6,0.9): r,_=resid(w,Z,[lt1,a,lt2,0.55],bounded); starts.append((r@r,[lt1,a,lt2,0.55]))
        starts=sorted(starts,key=lambda t:t[0])[:4]; best=None
        for _,s in starts:
            o=least_squares(lambda p: resid(w,Z,p,bounded)[0],s,bounds=(LO,HI),diff_step=1e-6,xtol=1e-10)
            r,x=resid(w,Z,o.x,bounded)
            if best is None or r@r<best['rss']: best=dict(p=o.x.copy(),x=x,rss=float(r@r))
        return best
    lo=np.array([0.3,-6,0.3]); hi=np.array([1,-2.6,1]); q0=np.clip(np.array(p0[1:]),lo+1e-6,hi-1e-6)
    f=lambda q: resid(w,Z,[fixed_lt1,q[0],q[1],q[2]],bounded)[0]
    o=least_squares(f,q0,bounds=(lo,hi),diff_step=1e-6,xtol=1e-9); r=f(o.x); p=np.r_[fixed_lt1,o.x]
    return dict(p=p,rss=float(r@r))
def profile_lt1(w,Z,full,bounded=True,half=3.0,step=0.05):
    """Profile of log10 tau1 by continuation from the optimum in both directions; refined crossings."""
    n=2*len(w); rmin=full['rss']; out={}
    for direction in (+1,-1):
        p=full['p'].copy(); g=[]; v=[]; lt=full['p'][0]
        while abs(lt-full['p'][0])<=half and LO[0]<=lt<=HI[0]:
            r=fit(w,Z,lt,p,bounded); p=r['p']; g.append(lt); v.append(r['rss']); lt+=direction*step
            if n*np.log(r['rss']/rmin)>20: break
        out[direction]=(np.array(g),np.array(v))
    g=np.r_[out[-1][0][::-1],out[+1][0][1:]]; v=np.r_[out[-1][1][::-1],out[+1][1][1:]]
    rmin=min(rmin,v.min()); lam=n*np.log(v/rmin); k=int(np.argmin(lam)); i=k; j=k
    while i>0 and lam[i-1]<=3.84: i-=1
    while j<len(g)-1 and lam[j+1]<=3.84: j+=1
    def cross(a,b): return g[a]+(3.84-lam[a])*(g[b]-g[a])/(lam[b]-lam[a]) if lam[b]!=lam[a] else g[a]
    clo=i>0 and lam[0]>3.84; chi=j<len(g)-1 and lam[-1]>3.84
    lo=cross(i-1,i) if i>0 else g[0]; hi=cross(j,j+1) if j<len(g)-1 else g[-1]
    closed=bool(clo and chi)
    return dict(lo=float(lo),hi=float(hi),width=float(hi-lo),closed=closed,ident=bool(closed and hi-lo<WSTAR))
def jac(fun,p,h=1e-6):
    r0=fun(p); J=np.zeros((r0.size,p.size))
    for i in range(p.size):
        d=np.zeros_like(p); d[i]=h*max(1,abs(p[i])); J[:,i]=(fun(p+d)-fun(p-d))/(2*d[i])
    return J
def asym_fit(w,Z):
    """R0 + 1/(Q1 (jw)^alpha1) + ZARC2 + A_W/sqrt(jw) + jwL, nonlinear (alpha1, lt2, alpha2)."""
    W=1/np.abs(Z); b=stack(Z*W)
    def rr(q):
        C=[np.ones_like(w,dtype=complex),(1j*w)**(-q[0]),zarc(w,q[1],q[2]),1/np.sqrt(1j*w),1j*w]
        A=np.column_stack([stack(c*W) for c in C]); x,_=nnls(A,b); return A@x-b,x
    best=None
    for a in (0.6,0.8,0.95):
        for lt2 in np.arange(-6,-2.59,0.5):
            o=least_squares(lambda q: rr(q)[0],[a,lt2,0.55],bounds=([0.3,-6,0.3],[1,-2.6,1]),diff_step=1e-6); r,x=rr(o.x)
            if best is None or r@r<best[0]: best=(r@r,o.x,x)
    _,q,x=best
    return dict(R0=max(x[0],1e-6),Q=1/max(x[1],1e-12),al=float(q[0]),dR2=max(x[2],1e-9),lt2=float(q[1]),al2=float(q[2]),AW=float(x[3]),L=max(x[4],1e-12))
def asym_from_fit(F):
    """Asymptote implied by the full-model fit on the band: Q1 = tau1^alpha1 / dR1 (well determined even when tau1 is not)."""
    p,x=F['p'],F['x']; tau=10**p[0]
    return dict(R0=max(x[0],1e-6),Q=tau**p[1]/max(x[1],1e-12),al=float(p[1]),dR2=max(x[2],1e-9),lt2=float(p[2]),al2=float(p[3]),AW=float(x[3]),L=max(x[4],1e-12))
def zasym(w,v,a):  # v = [log R0, log Q, al, log dR2, lt2, al2, AW, log L]
    return 10**v[0]+1/(10**v[1]*(1j*w)**v[2])+10**v[3]*zarc(w,v[4],v[5])+v[6]/np.sqrt(1j*w)+1j*w*10**v[7]
def vec(a): return np.array([np.log10(a['R0']),np.log10(a['Q']),a['al'],np.log10(a['dR2']),a['lt2'],a['al2'],a['AW'],np.log10(a['L'])])
def closed_form(w,a,sig,Wt):
    J=jac(lambda v: stack(zasym(w,v,a)/Wt),vec(a)); P=np.eye(J.shape[0])-J@np.linalg.pinv(J)
    ht=stack(((1j*w)**(-a['al'])/a['Q'])/Wt*(1j*w/w.min())**(-a['al'])); D=np.linalg.norm(P@ht)
    return D, D/np.linalg.norm(ht), (a['al']*LN10*D*WSTAR/(2*Z95*sig))**(1/a['al'])
def Wlin(w,a,fc,sig,Wt):
    def z(q):  # q = vec(a) + [log tau1]
        tau=10**q[8]; dR1=tau**q[2]/10**q[1]
        return 10**q[0]+dR1/(1+(1j*w*tau)**q[2])+10**q[3]*zarc(w,q[4],q[5])+q[6]/np.sqrt(1j*w)+1j*w*10**q[7]
    q=np.r_[vec(a),np.log10(1/(2*np.pi*fc))]; J=jac(lambda q: stack(z(q)/Wt),q)
    try: C=np.linalg.inv(J.T@J); return 2*Z95*sig*np.sqrt(max(C[8,8],0))
    except np.linalg.LinAlgError: return np.inf
def f_DL(w,a,sig,Wt):
    fmin=w.min()/(2*np.pi); g=lambda lf: np.log(Wlin(w,a,10**lf,sig,Wt))-np.log(WSTAR); lo,hi=np.log10(fmin)-5,np.log10(fmin)+0.5
    try:
        if g(hi)>0: return 10**hi,'above'
        if g(lo)<0: return 10**lo,'below'
        return float(10**brentq(g,lo,hi,xtol=1e-3)),'ok'
    except Exception: return np.nan,'failed'
if __name__=='__main__':
    folder,out=sys.argv[1],sys.argv[2]; ncyc=int(sys.argv[3]) if len(sys.argv)>3 else 3
    FMINS=[1,3,10,20,30,50,100]; rows=[]
    for fn in sorted(glob.glob(os.path.join(folder,'EIS_state_V_25C0[1-8].txt'))):
        d=np.loadtxt(fn,skiprows=1); cyc=np.unique(d[:,1]); pick=cyc[np.linspace(0,len(cyc)-1,ncyc).astype(int)]
        for cy in pick:
            s=d[d[:,1]==cy]; s=s[np.argsort(s[:,2])]; f=s[:,2]; Z=s[:,3]-1j*s[:,4]; w=2*np.pi*f
            full=fit(w,Z); fpr=profile_lt1(w,Z,full); fc1=1/(2*np.pi*10**full['p'][0])
            for fm in FMINS:
                m=f>=fm*0.999; wb,Zb=w[m],Z[m]; Wt=np.abs(Zb); ft=fit(wb,Zb); sig=np.sqrt(ft['rss']/(2*m.sum()-NPAR))
                pr=profile_lt1(wb,Zb,ft); pu=profile_lt1(wb,Zb,fit(wb,Zb,bounded=False),bounded=False)
                a=asym_from_fit(ft); D,sinth,G=closed_form(wb,a,sig,Wt); dl,fl=f_DL(wb,a,sig,Wt)
                fch=1/(2*np.pi*10**ft['p'][0]); pc=fch>f[m].min()/G; po=fch>dl
                rows.append(dict(file=os.path.basename(fn),cycle=float(cy),fmin=fm,N=int(m.sum()),fc1_full=fc1,alpha1_full=float(full['p'][1]),
                    fc2_full=float(1/(2*np.pi*10**full['p'][2])),full_ident=fpr['ident'],full_width=fpr['width'],nrmse_full=float(np.sqrt(full['rss']/len(w))),
                    sigma=float(sig),fc1_hat=float(fch),obs=pr['ident'],width=pr['width'],closed=pr['closed'],obs_unb=pu['ident'],width_unb=pu['width'],
                    truth_in=bool(pr['lo']<=full['p'][0]<=pr['hi']),DL_closed=float(f[m].min()/G),f_DL=float(dl),f_DL_flag=fl,Gamma=float(G),sinth=float(sinth),
                    pred_closed=bool(pc),pred_fDL=bool(po),three='likely' if (pc and po) else ('unlikely' if not(pc or po) else 'uncertain'),
                    r=float((fm/fc1)**(-full['p'][1]))))
            print(os.path.basename(fn),int(cy),'fc1 %.1f Hz (alpha %.2f) full-band width %.2f'%(fc1,full['p'][1],fpr['width']),flush=True)
    json.dump(rows,open(out,'w'),default=float); print('done',len(rows))
