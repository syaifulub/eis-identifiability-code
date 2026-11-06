import numpy as np, json, sys
src=open('mc_rigor.py').read().split('out=[]')[0]
ns={}; exec(src,ns)
f=ns['f']; a=ns['a']; Q=ns['Q']; Rinf=ns['Rinf']; L=ns['L']; LN10=ns['LN10']; Z95=ns['Z95']; N=ns['N']
rng=np.random.default_rng(7)
res=[]
for r in [0.02,0.05,0.1,0.2,0.3,0.5]:
    sig=0.018; fc=f[0]*r**(1/a); tau=1/(2*np.pi*fc); dR=tau**a/Q; Z0=ns['cole'](Rinf,dR,tau,a,L); W0=np.abs(Z0)
    D=ns['D_of'](Rinf,Q,a,L,Z0); Gt=(a*LN10*D/(2*Z95*sig))**(1/a)
    # noise-free plug-in (bias of the plug-in itself)
    Ri,Qh,ah,Lh=ns['fit_rcpel'](Z0); Dn=ns['D_of'](Ri,Qh,ah,Lh,Z0); Gn=(ah*LN10*Dn/(2*Z95*sig))**(1/ah)
    lg=[];A=[]
    for k in range(200):
        e=sig*rng.standard_normal(2*N); Zo=Z0+W0*(e[:N]+1j*e[N:])
        lt,al,rss,x=ns['fit_cole'](Zo,np.log10(tau)); Ri,Qh,ah,Lh=ns['fit_rcpel'](Zo); sh=np.sqrt(rss/(2*N-5))
        Dh=ns['D_of'](Ri,Qh,ah,Lh,Zo); lg.append(np.log10(f[0]*(2*Z95*sh/(ah*LN10*Dh))**(1/ah))); A.append(ah)
    lg=np.array(lg)
    res.append(dict(r=r,fc=fc,DL_true=f[0]/Gt,DL_plugin_noisefree=f[0]/Gn,alpha_hat_noisefree=ah if False else float(ns['fit_rcpel'](Z0)[2]),
                    DL_med=float(10**np.median(lg)),DL_p5=float(10**np.percentile(lg,5)),DL_p95=float(10**np.percentile(lg,95)),sd_dec=float(np.std(lg,ddof=1)),alpha_hat_med=float(np.median(A))))
    print({k:(round(v,3) if isinstance(v,float) else v) for k,v in res[-1].items()})
json.dump(res,open(sys.argv[1],'w'),indent=1,default=float)
