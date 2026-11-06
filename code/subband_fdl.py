"""Operational detection limit f_DL and closed-form f_min/Gamma for the median KK-consistent asymptote
(R_inf 2211 Ohm, Q 7.4e-8, alpha 0.69, L 1.14 mH; sigma 0.018) on the measured band and its sub-bands (Table 6)."""
import numpy as np, sys, json
src=open('classify_sim.py').read().split('NT=60')[0]; ns={'sys':sys}; exec(src,ns)
Ri,Q,a,L,sig=2211.,7.4e-8,0.69,1.14e-3,0.018
out={}
for name,f in [('10-100',np.arange(10e3,100.1e3,10e3)),('20-100',np.arange(20e3,100.1e3,10e3)),('30-100',np.arange(30e3,100.1e3,10e3)),
               ('40-100',np.arange(40e3,100.1e3,10e3)),('10-70',np.arange(10e3,70.1e3,10e3))]:
    w=2*np.pi*f; Wa=np.abs(Ri+1/(Q*(1j*w)**a)+1j*w*L); D=ns['D_asym'](w,Ri,Q,a,L,Wa); G=(a*ns['LN10']*D/(2*ns['Z95']*sig))**(1/a)
    dl=ns['DL_A8'](w,Ri,Q,a,L,sig); out[name]=dict(N=f.size,Gamma=G,DL8=f[0]/G,fDL=dl,capped=bool(dl>=f[0]*10**0.5*0.999))
    print(name,f.size,'Gamma %.2f DL8 %.0f fDL %.0f'%(G,f[0]/G,dl),'(capped)' if out[name]['capped'] else '')
json.dump(out,open('subband_fdl.json','w'),indent=1)
