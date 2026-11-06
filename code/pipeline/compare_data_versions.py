"""Compare the M11 NRMSE of the 20 spectra built from several readings files (e.g. PreparedData v2 and raw conversions) with Table S1. Usage: python compare_data_versions.py a.csv b.csv ..."""
import sys; sys.path.insert(0,'/home/claude/new/code/pipeline')
import pandas as pd, numpy as np, eis_pipeline as ep
F=[10e3*k for k in range(1,11)]
man=pd.read_csv('/home/claude/new/code/pipeline/spectra_manifest.csv',dtype={'recording':str})
out={}
for name in sys.argv[1:]:
    df=pd.read_csv(name,dtype={'recording':str}); sp=ep.spectra_from_readings(df,F); sw=ep.sweeps_from_readings(df,F); v=[]
    for r in man.itertuples():
        k=(r.recording,r.tissue)
        if k not in sp: v.append((np.nan,0)); continue
        f,Z=sp[k]; v.append((np.sqrt(ep.fit(ep.Spectrum(f,Z),'M11')['rss']/10),len(sw.get(k,[]))))
    out[name]=v
print('sp tissue        pub    '+'  '.join('%-14s'%n[:14] for n in out))
for i,r in enumerate(man.itertuples()):
    print('%2d %-12s %.4f '%(r.spectrum,r.tissue,r.pub_m11_nrmse)+'  '.join('%.4f(%d)%s     '%(v[i][0],v[i][1],'*' if abs(v[i][0]-r.pub_m11_nrmse)<5e-4 else ' ') for v in out.values()))
for n,v in out.items(): print(n,'matches',sum(abs(a-b)<5e-4 for (a,_),b in zip(v,man.pub_m11_nrmse)),'sweeps',sum(s for _,s in v))
