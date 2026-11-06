"""Classification study of identifiability decision rules against profile likelihood (reference).
Synthetic Cole+L spectra with tissue-like parameters, as in Sect. 2.8 (independent re-simulation, own seeds).
Rules: majority class; Eq. (8)/(9) closed form (plug-in from R-CPE-L fit, f_c from Cole+L fit);
exact linearized width Eq. (A6) along the fitted asymptote ("asymptote-A8", same inputs as the closed form);
FIM/CRLB baseline (Wald width from the full Cole+L fit); oracle versions at the true parameters.
Also records timing. The Algorithm 2 test is in alg2_test.py."""
import numpy as np, json, sys, time
from scipy.optimize import nnls, minimize_scalar, brentq
LN10=np.log(10); Z95=1.959964; WSTAR=1.0
rng=np.random.default_rng(42)
def stack(c): return np.r_[c.real,c.imag]
def proj(J): return np.eye(J.shape[0])-J@np.linalg.pinv(J)
def cole(w,Ri,dR,tau,al,Li): return Ri+dR/(1+(1j*w*tau)**al)+1j*w*Li
def jac(fun,p,h=1e-6):
    p=np.nan_to_num(p,neginf=-12.0,posinf=12.0)
    r0=fun(p); J=np.zeros((r0.size,p.size))
    for i in range(p.size):
        d=np.zeros_like(p); d[i]=h*max(1,abs(p[i])); J[:,i]=(fun(p+d)-fun(p-d))/(2*d[i])
    return J
LT_LO,LT_HI=-7.0,1.0
def profile_fit(w,Zo,grid=np.arange(LT_LO,LT_HI+1e-9,0.05)):
    W=np.abs(Zo); b=stack(Zo/W); N=w.size
    def inner(lt,al):
        t=10**lt; A=np.column_stack([stack(np.ones(N)/W),stack(1/(1+(1j*w*t)**al)/W),stack(1j*w/W)])
        x,res=nnls(A,b); return res**2,x
    prof=[];als=[]
    for g in grid:
        o=minimize_scalar(lambda al: inner(g,al)[0],bounds=(0.3,1.0),method='bounded',options={'xatol':1e-6}); prof.append(o.fun); als.append(o.x)
    prof=np.array(prof); k=int(np.argmin(prof))
    # refine minimum
    o=minimize_scalar(lambda lt: minimize_scalar(lambda al: inner(lt,al)[0],bounds=(0.3,1.0),method='bounded').fun,
                      bounds=(grid[max(k-1,0)],grid[min(k+1,len(grid)-1)]),method='bounded',options={'xatol':1e-5})
    lt_hat=o.x; oa=minimize_scalar(lambda al: inner(lt_hat,al)[0],bounds=(0.3,1.0),method='bounded',options={'xatol':1e-7})
    rss_min,x=inner(lt_hat,oa.x); rss_min=min(rss_min,prof.min())
    n=2*N; Lam=n*np.log(prof/rss_min)
    ins=grid[Lam<=3.84]; closed=(Lam[0]>3.84) and (Lam[-1]>3.84)
    width=(ins.max()-ins.min()+0.05) if ins.size else 0.05
    return dict(lt=lt_hat,al=oa.x,R=x[0],dR=x[1],L=x[2],rss=rss_min,ident=bool(closed and width<WSTAR),width=float(width),closed=bool(closed))
def fit_rcpel(w,Zo):
    W=np.abs(Zo); b=stack(Zo/W); N=w.size
    def inner(al):
        A=np.column_stack([stack(np.ones(N)/W),stack((1j*w)**(-al)/W),stack(1j*w/W)]); x,res=nnls(A,b); return res**2,x
    o=minimize_scalar(lambda al: inner(al)[0],bounds=(0.3,1.0),method='bounded',options={'xatol':1e-8}); rss,x=inner(o.x)
    return x[0],1/max(x[1],1e-300),o.x,max(x[2],1e-12)
def D_asym(w,Ri,Q,al,Li,Wt):
    def rn(p): return stack((10**p[0]+1/(10**p[1]*(1j*w)**p[2])+1j*w*10**p[3])/Wt)
    J=jac(rn,np.array([np.log10(Ri),np.log10(Q),al,np.log10(Li)])); P=proj(J)
    ht=stack(((1j*w)**(-al)/Q)/Wt*(1j*w/w[0])**(-al)); return np.linalg.norm(P@ht)
def W8(w,Ri,Q,al,Li,fc,sig,Wt):
    return 2*Z95*sig/(al*LN10*D_asym(w,Ri,Q,al,Li,Wt))*(w[0]/(2*np.pi*fc))**al
def Wlin(w,Ri,Q,al,Li,fc,sig,Wt=None):
    """exact linearized width (Eq. A6) for a Cole+L spectrum with asymptote (Ri,Q,al,Li) and characteristic f_c"""
    tau=1/(2*np.pi*fc); Z0=cole(w,Ri,tau**al/Q,tau,al,Li); Wt=np.abs(Z0) if Wt is None else Wt
    def r5(p): return stack(cole(w,10**p[0],(10**p[4])**p[2]/10**p[1],10**p[4],p[2],10**p[3])/Wt)
    J5=jac(r5,np.array([np.log10(Ri),np.log10(Q),al,np.log10(Li),np.log10(tau)]))
    try: C=np.linalg.inv(J5.T@J5); return 2*Z95*sig*np.sqrt(max(C[4,4],0))
    except np.linalg.LinAlgError: return np.inf
def DL_A8(w,Ri,Q,al,Li,sig):
    """detection limit along the asymptote: largest f_c below f_min... solve Wlin(f_c)=W* for f_c < f_min"""
    fmin=w[0]/(2*np.pi); g=lambda lfc: np.log(Wlin(w,Ri,Q,al,Li,10**lfc,sig))-np.log(WSTAR)
    lo,hi=np.log10(fmin)-5,np.log10(fmin)+0.5
    try:
        if g(hi)>0: return 10**hi
        if g(lo)<0: return 10**lo
        return 10**brentq(g,lo,hi,xtol=1e-3)
    except Exception: return np.nan
def metrics(pred,obs):
    pred=np.asarray(pred,bool); obs=np.asarray(obs,bool)
    TP=int(np.sum(pred&obs)); TN=int(np.sum(~pred&~obs)); FP=int(np.sum(pred&~obs)); FN=int(np.sum(~pred&obs))
    sens=TP/(TP+FN) if TP+FN else np.nan; spec=TN/(TN+FP) if TN+FP else np.nan
    return dict(n=int(pred.size),TP=TP,TN=TN,FP=FP,FN=FN,acc=(TP+TN)/max(pred.size,1),sens=sens,spec=spec,
                bacc=np.nanmean([sens,spec]) if not(np.isnan(sens) and np.isnan(spec)) else np.nan)

NT=60; classes=[(100,1e3),(1e3,1e4),(1e4,3e4)]
truths=[]
for (lo,hi) in classes:
    for k in range(NT//3):
        Ri=10**rng.uniform(np.log10(500),np.log10(5000)); dR=Ri*rng.uniform(1,3); al=rng.uniform(0.60,0.95)
        L=rng.uniform(0.4e-3,2.5e-3); fc=10**rng.uniform(np.log10(lo),np.log10(hi)); tau=1/(2*np.pi*fc); Q=tau**al/dR
        truths.append(dict(Ri=Ri,dR=dR,al=al,L=L,fc=fc,Q=Q))
fmins=[10e3,3e3,1e3,300.]; noises=[0.01,0.025,0.05]; REPS=3
rows=[]; t0=time.time()
timing=dict(closed=[],asymA8=[],fim=[],profile=[])
for it,tr in enumerate(truths):
    tau=1/(2*np.pi*tr['fc'])
    for fmin in fmins:
        f=np.logspace(np.log10(fmin),5,20); w=2*np.pi*f
        Z0=cole(w,tr['Ri'],tr['dR'],tau,tr['al'],tr['L']); W0=np.abs(Z0)
        for sig in noises:
            o_W8=W8(w,tr['Ri'],tr['Q'],tr['al'],tr['L'],tr['fc'],sig,W0); o_Wl=Wlin(w,tr['Ri'],tr['Q'],tr['al'],tr['L'],tr['fc'],sig)
            r=(fmin/tr['fc'])**(-tr['al'])
            for rep in range(REPS):
                e=sig*rng.standard_normal(40); Zo=Z0+W0*(e[:20]+1j*e[20:])
                t1=time.perf_counter(); pf=profile_fit(w,Zo); timing['profile'].append(time.perf_counter()-t1)
                fc_hat=1/(2*np.pi*10**pf['lt']); sig_hat=np.sqrt(pf['rss']/(2*20-5)); Wobs=np.abs(Zo)
                # closed form (plug-in)
                t1=time.perf_counter(); Ri_,Q_,a_,L_=fit_rcpel(w,Zo); Ri_=max(Ri_,1e-4*np.min(Wobs))
                try: D=D_asym(w,Ri_,Q_,a_,L_,Wobs)
                except np.linalg.LinAlgError: D=np.nan
                Gam=(a_*LN10*D*WSTAR/(2*Z95*sig_hat))**(1/a_); DL8=fmin/Gam; timing['closed'].append(time.perf_counter()-t1)
                indom=(fmin>fc_hat) and (Gam>=2**(1/a_))
                pred8=fc_hat>DL8
                # asymptote-A8 detection limit (same inputs as closed form)
                t1=time.perf_counter(); DLa=DL_A8(w,Ri_,Q_,a_,L_,sig_hat); timing['asymA8'].append(time.perf_counter()-t1)
                predA=fc_hat>DLa
                # FIM/CRLB baseline at the full Cole+L fit
                t1=time.perf_counter(); Qf=(10**pf['lt'])**pf['al']/max(pf['dR'],1e-12)
                Wf=Wlin(w,max(pf['R'],1e-9),Qf,pf['al'],max(pf['L'],1e-12),fc_hat,sig_hat,Wobs); timing['fim'].append(time.perf_counter()-t1+timing['profile'][-1]*0)  # Jacobian only; fit cost reported separately
                predF=Wf<WSTAR
                rows.append(dict(truth=it,fmin=fmin,sigma=sig,rep=rep,r=r,fc=tr['fc'],fc_hat=fc_hat,obs=pf['ident'],prof_w=pf['width'],
                    oracle8=bool(o_W8<WSTAR),oracleLin=bool(o_Wl<WSTAR),Wlin_true=o_Wl,W8_true=o_W8,
                    pred8=bool(pred8),predA8=bool(predA),predFIM=bool(predF),indom=bool(indom),DL8=DL8,DLA8=DLa,Gam=Gam,alpha_hat=a_))
    print(it,round(time.time()-t0),flush=True)
    json.dump(dict(rows=rows,partial=True),open(sys.argv[1]+'.part','w'),default=float)
json.dump(dict(rows=rows,timing={k:float(np.median(v)) for k,v in timing.items()},truths=truths),open(sys.argv[1],'w'),default=float)
print('done',time.time()-t0)
