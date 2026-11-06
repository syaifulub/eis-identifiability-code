"""Convert registros_EIS.mat (Dell'Osa et al., Zenodo 10.5281/zenodo.15795175) to the CSV of wideband_test.py.
Usage: python -I prepare_dellosa.py registros_EIS.mat wideband_all.csv
Ayllon et al. 2016 subset (origin 4): column 1 is angular frequency (rad/s); IDs 1-50 error-free.
Gonzalez-Correa subset (origin 2): frequency in kHz, reactance stored positive (inverted here)."""
import sys, numpy as np, scipy.io as sio, pandas as pd
m=sio.loadmat(sys.argv[1]); R=m['registros'][0]
def g(r,k):
    v=r[k]; return str(np.ravel(v)[0].ravel()[0]) if v.dtype==object else str(np.ravel(v)[0])
rows=[]
for r in R:
    i=int(g(r,'id')); o=g(r,'origin'); e=np.asarray(r['eis'],float)
    if o=='4':   # Ayllon et al. 2016, SFB7 (Impedimed), tetrapolar, human; column 1 is angular frequency (rad/s)
        f=e[:,0]/(2*np.pi); re=e[:,1]; im=e[:,2]
        cls='error_free' if i<=50 else ('S2' if i<=390 else 'S3' if i<=610 else 'S4' if i<=660 else 'S5' if i<=1060 else 'S6' if i<=1200 else 'S7')
        src='Ayllon2016'
    elif o=='2': # Gonzalez-Correa 2018, Xitron 4001B, tetrapolar, human; frequency in kHz; reactance stored positive
        f=e[:,0]*1e3; re=e[:,1]; im=-e[:,2]
        cls='unlabelled'; src='GonzalezCorrea2018'
    else: continue
    for fi,a,b in zip(f,re,im): rows.append(dict(spectrum_id=f'ID{i:04d}',f_Hz=fi,re_ohm=a,im_ohm=b,source=src,sample_type=cls,device=g(r,'device'),electrodes='tetrapolar'))
d=pd.DataFrame(rows); d.to_csv(sys.argv[2],index=False)
print(d.groupby(['source','sample_type']).spectrum_id.nunique())
x=d[d.source=='GonzalezCorrea2018']; print(x.groupby('spectrum_id').im_ohm.median().describe()); print(x.head(3))
