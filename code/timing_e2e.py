"""End-to-end computing time from the raw spectrum, same machine, single thread, same starting grid (reviewer P2).
240 spectra (60 truths x 4 bands, 2.5% noise). Pipelines:
  closed : R-CPE-L VarPro fit + sigma from its residuals + Eq. (9)
  fDL    : R-CPE-L VarPro fit + sigma + root finding of W_lin along the asymptote
  FIM    : Cole+L VarPro fit (log10 tau grid 0.25 decade on [-7,1], local refinement) + Wald width at the fit
  profile: Cole+L profile likelihood (grid 0.05 decade on [-7,1], 161 points)"""
import os; os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np, json, sys, time
from scipy.optimize import nnls, minimize_scalar
src=open('classify_sim.py').read().split('fmins=')[0]; ns={'__name__':'x','sys':sys}; exec(src,ns)
truths=ns['truths']; fit_rcpel=ns['fit_rcpel']; profile_fit=ns['profile_fit']; DL_A8=ns['DL_A8']; cole=ns['cole']; Wlin=ns['Wlin']
D_asym=ns['D_asym']; LN10=ns['LN10']; Z95=ns['Z95']; stack=ns['stack']
rng=np.random.default_rng(31)
def cole_fit(w,Zo):
    W=np.abs(Zo); b=stack(Zo/W); N=w.size
    def inner(lt,al):
        A=np.column_stack([stack(np.ones(N)/W),stack(1/(1+(1j*w*10**lt)**al)/W),stack(1j*w/W)]); x,res=nnls(A,b); return res**2,x
    prof=lambda lt: minimize_scalar(lambda al: inner(lt,al)[0],bounds=(0.3,1.0),method='bounded').fun
    g=np.arange(-7,1.001,0.25); v=[prof(x) for x in g]; k=int(np.argmin(v))
    o=minimize_scalar(prof,bounds=(g[max(k-1,0)],g[min(k+1,len(g)-1)]),method='bounded',options={'xatol':1e-5})
    oa=minimize_scalar(lambda al: inner(o.x,al)[0],bounds=(0.3,1.0),method='bounded'); rss,x=inner(o.x,oa.x)
    return o.x,oa.x,x,rss
T={k:[] for k in ['closed','fDL','FIM','profile']}
for tr in truths:
    tau=1/(2*np.pi*tr['fc'])
    for fm in [10e3,3e3,1e3,300.]:
        f=np.logspace(np.log10(fm),5,20); w=2*np.pi*f; Z0=cole(w,tr['Ri'],tr['dR'],tau,tr['al'],tr['L'])
        e=0.025*rng.standard_normal(40); Zo=Z0+np.abs(Z0)*(e[:20]+1j*e[20:])
        def asym():
            Ri,Q,a,L=fit_rcpel(w,Zo); Ri=max(Ri,1e-4*np.min(np.abs(Zo)))
            Za=Ri+1/(Q*(1j*w)**a)+1j*w*L; r=(Zo-Za)/np.abs(Zo); sh=np.sqrt(np.sum(stack(r)**2)/(40-4)); return Ri,Q,a,L,sh
        t=time.perf_counter(); Ri,Q,a,L,sh=asym(); D=D_asym(w,Ri,Q,a,L,np.abs(Zo)); G=(a*LN10*D/(2*Z95*sh))**(1/a); T['closed'].append(time.perf_counter()-t)
        t=time.perf_counter(); Ri,Q,a,L,sh=asym(); DL_A8(w,Ri,Q,a,L,sh); T['fDL'].append(time.perf_counter()-t)
        t=time.perf_counter(); lt,al,x,rss=cole_fit(w,Zo); s2=np.sqrt(rss/35); Qf=(10**lt)**al/max(x[1],1e-12)
        Wlin(w,max(x[0],1e-9),Qf,al,max(x[2],1e-12),1/(2*np.pi*10**lt),s2,np.abs(Zo)); T['FIM'].append(time.perf_counter()-t)
        t=time.perf_counter(); profile_fit(w,Zo); T['profile'].append(time.perf_counter()-t)
out={k:dict(median_ms=1e3*float(np.median(v)),q1_ms=1e3*float(np.percentile(v,25)),q3_ms=1e3*float(np.percentile(v,75))) for k,v in T.items()}
import platform; out['machine']=platform.processor() or platform.machine(); out['python']=platform.python_version()
json.dump(out,open(sys.argv[1],'w'),indent=1); print(json.dumps(out,indent=1))
