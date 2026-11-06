"""Sensitivity of Algorithm 2 to conditions in the candidate band that differ from the pilot (reviewer P1).
Scenarios (reference = profile-likelihood identifiability in the candidate band, majority of 3 replicates):
  noise1 / noise5 : white noise 1% / 5% in the candidate band, pilot at 2.5% (reference rows from classify_sim)
  lowf            : noise rising below 10 kHz, sigma(f)=2.5%*(10 kHz/f)^0.3 (7.2% at 0.3 kHz); pilot 2.5% (10-100 kHz)
  pol10 / pol50   : unmodelled series electrode CPE (nu=0.6), |Z_pol|=10% / 50% of tissue |Z| at 1 kHz, present in
                    pilot and candidate band; 2.5% white noise; fitted model Cole+L (misspecified)
Decision: pilot-based operational detection limit f_DL (exact prior and +-0.5 decade) and closed form."""
import numpy as np, json, sys
src=open('classify_sim.py').read().split('fmins=')[0]; ns={'__name__':'x','sys':sys}; exec(src,ns)
truths=ns['truths']; fit_rcpel=ns['fit_rcpel']; profile_fit=ns['profile_fit']; DL_A8=ns['DL_A8']; cole=ns['cole']
D_asym=ns['D_asym']; LN10=ns['LN10']; Z95=ns['Z95']
rng=np.random.default_rng(23); fmins=[10e3,3e3,1e3,300.]; REPS=3
def zpol(w,tr,frac):
    if frac==0: return 0*w
    tau=1/(2*np.pi*tr['fc']); w1=2*np.pi*1e3; Zt1=abs(cole(np.array([w1]),tr['Ri'],tr['dR'],tau,tr['al'],tr['L'])[0])
    K=frac*Zt1*w1**0.6; return K/(1j*w)**0.6
def spec(w,tr,frac,sigf):
    tau=1/(2*np.pi*tr['fc']); Z0=cole(w,tr['Ri'],tr['dR'],tau,tr['al'],tr['L'])+zpol(w,tr,frac); n=w.size
    e=rng.standard_normal(2*n); s=sigf(w/(2*np.pi)); return Z0+np.abs(Z0)*s*(e[:n]+1j*e[n:])
out=[]
scen={'lowf':(0.0,lambda f:0.025*np.maximum(1,(1e4/f)**0.3)),'pol10':(0.10,lambda f:0.025+0*f),'pol50':(0.50,lambda f:0.025+0*f)}
for it,tr in enumerate(truths):
    for sc,(frac,sigf) in scen.items():
        fp=np.logspace(4,5,10); wp=2*np.pi*fp; Zp=spec(wp,tr,frac,lambda f:0.025+0*f)
        Ri,Q,a,L=fit_rcpel(wp,Zp); Ri=max(Ri,1e-4*np.min(np.abs(Zp))); pf=profile_fit(wp,Zp); sh=max(np.sqrt(pf['rss']/(2*10-5)),1e-4)
        for fm in fmins:
            f=np.logspace(np.log10(fm),5,20); w=2*np.pi*f
            dl=DL_A8(w,Ri,Q,a,L,sh); D=D_asym(w,Ri,Q,a,L,np.abs(Ri+1/(Q*(1j*w)**a)+1j*w*L)); G=(a*LN10*D/(2*Z95*sh))**(1/a)
            obs=[profile_fit(w,spec(w,tr,frac,sigf))['ident'] for _ in range(REPS)]
            out.append(dict(truth=it,scen=sc,fmin=fm,fc=tr['fc'],DLA8=dl,DL8=fm/G,pilot_ident=bool(pf['ident']),sig_hat=sh,obs=bool(np.mean(obs)>=0.5)))
    print(it,flush=True)
json.dump(out,open(sys.argv[1],'w'),default=float); print('done')
