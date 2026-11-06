"""Algorithm 2 test (M2): design decision from a pilot sweep, with and without a prior on f_c.
Pilot: 10-100 kHz, N=10, 2.5% noise. Asymptote (R-CPE-L) and sigma from the pilot.
Detection limit per candidate band (f_min..100 kHz, N=20) from the exact linearized width (Eq. A6) along the pilot asymptote.
Decision 'band adequate' if f_c^exp > DL (prior modes) ; evaluated against profile identifiability at sigma=2.5% (classify_sim rows)."""
import numpy as np, json, sys
src=open('classify_sim.py').read().split('fmins=')[0]; ns={'__name__':'x','sys':sys}; exec(src,ns)
truths=ns['truths']; fit_rcpel=ns['fit_rcpel']; profile_fit=ns['profile_fit']; DL_A8=ns['DL_A8']; cole=ns['cole']
rng=np.random.default_rng(7); SIG=0.025; fmins=[10e3,3e3,1e3,300.]
out=[]
for it,tr in enumerate(truths):
    tau=1/(2*np.pi*tr['fc']); fp=np.logspace(4,5,10); wp=2*np.pi*fp
    Z0=cole(wp,tr['Ri'],tr['dR'],tau,tr['al'],tr['L']); e=SIG*rng.standard_normal(20); Zp=Z0+np.abs(Z0)*(e[:10]+1j*e[10:])
    Ri,Q,a,L=fit_rcpel(wp,Zp); Ri=max(Ri,1e-4*np.min(np.abs(Zp)))
    pf=profile_fit(wp,Zp); sig_hat=max(np.sqrt(pf['rss']/(2*10-5)),1e-4)
    fc_pilot=1/(2*np.pi*10**pf['lt']) if pf['ident'] else np.nan
    for fm in fmins:
        f=np.logspace(np.log10(fm),5,20); w=2*np.pi*f
        dl=DL_A8(w,Ri,Q,a,L,sig_hat)
        # Eq. (9) closed-form DL along the same pilot asymptote
        D=ns['D_asym'](w,Ri,Q,a,L,np.abs(Ri+1/(Q*(1j*w)**a)+1j*w*L)); Gam=(a*ns['LN10']*D*1.0/(2*ns['Z95']*sig_hat))**(1/a)
        out.append(dict(truth=it,fmin=fm,fc=tr['fc'],DLA8=dl,DL8=fm/Gam,fc_pilot=fc_pilot,pilot_ident=bool(pf['ident']),sig_hat=sig_hat,alpha_pilot=a))
    print(it,flush=True)
json.dump(out,open(sys.argv[1],'w'),default=float)
print('done')
