"""Synthetic readings in the tidy input format, for testing the pipeline without the measured data.
One spectrum per manifest row: Cole+L with alpha, f_c and relative error taken from the published Table S26 values
(R_inf, L and the CPE level drawn around the medians of the KK-consistent spectra); 1-6 ascending sweeps per
recording, 3 readings per frequency. The output is NOT the measured dataset."""
import numpy as np, pandas as pd, sys, os
man=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'spectra_manifest.csv'),dtype={'recording':str})
rng=np.random.default_rng(7); F=[10e3*k for k in range(1,11)]; rows=[]
for r in man.itertuples():
    al=float(np.clip(r.pub_alpha_m10,0.45,0.95)); fc=float(np.clip(r.pub_fc_hat_hz,30,3e4)); sig=float(min(r.pub_sigma,0.06))
    Ri=10**rng.normal(np.log10(2211),0.15); Q=10**rng.normal(np.log10(7.4e-8),0.2); L=10**rng.normal(np.log10(1.14e-3),0.1)
    tau=1/(2*np.pi*fc); dR=tau**al/Q; order=0
    for s in range(int(rng.integers(1,7))):
        for fq in F:
            w=2*np.pi*fq; Z=Ri+dR/(1+(1j*w*tau)**al)+1j*w*L
            for _ in range(3):
                e=sig*np.sqrt(3)*rng.standard_normal(2); z=Z+abs(Z)*(e[0]+1j*e[1])
                rows.append(dict(recording=r.recording,tissue=r.tissue,animal=r.animal,probe='scan',mode='scan',order=order,
                                 frequency_hz=fq,z_real=z.real,z_imag=z.imag)); order+=1
pd.DataFrame(rows).to_csv(sys.argv[1] if len(sys.argv)>1 else 'synthetic_readings.csv',index=False); print(len(rows),'readings')
