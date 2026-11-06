"""Compare pipeline output with the published values (Tables S1 and S26) stored in spectra_manifest.csv."""
import pandas as pd, numpy as np, sys, os
res=sys.argv[1] if len(sys.argv)>1 else 'results'
man=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'spectra_manifest.csv'),dtype={'recording':str})
qc=pd.read_csv(os.path.join(res,'spectra_qc.csv')); cr=pd.read_csv(os.path.join(res,'criterion.csv'))
m=man.merge(qc,on=['spectrum','tissue']).merge(cr,on=['spectrum','tissue'],suffixes=('','_c'))
num=lambda s: pd.to_numeric(s.astype(str).str.replace('>1e5','1e5'),errors='coerce')
rows=[('KK RMS (%)','pub_kk_rms_pct','kk_rms_pct'),('M11 NRMSE','pub_m11_nrmse','m11_nrmse'),('sigma','pub_sigma','sigma'),
      ('alpha (M11)','pub_alpha_m11','alpha_m11'),('Gamma','pub_gamma','Gamma'),('closed-form DL (Hz)','pub_dl_nominal_hz','DL_closed_nominal'),
      ('fc_hat (Hz)','pub_fc_hat_hz','fc_hat'),('fc upper (Hz)','pub_fc_upper_hz','fc_upper')]
print('%-22s %10s %12s %12s'%('quantity','median pub','median new','median |rel diff|'))
for lab,p,q in rows:
    a=num(m[p]); b=m[q].astype(float); rd=np.abs(b-a)/np.abs(a).replace(0,np.nan)
    print('%-22s %10.4g %12.4g %12.2f'%(lab,np.nanmedian(a),np.nanmedian(b),np.nanmedian(rd)))
agree=lambda p,q:int((m[p].astype(str).str.lower().isin(['yes','true'])==m[q].astype(bool)).sum())
print('KK classification agreement: %d/%d'%(agree('pub_kk_consistent','kk_consistent'),len(m)))
print('tau identifiable (observed) agreement: %d/%d'%(agree('pub_ident_obs','ident_observed'),len(m)))
print('closed-form prediction agreement: %d/%d'%(agree('pub_ident_pred','ident_closed'),len(m)))
print('\nPer spectrum: tissue, fc_hat, closed-form DL, f_DL (operational), prediction closed / f_DL / three-way, observed')
for r in m.itertuples():
    print('%2d %-12s %8.0f %8.0f %8.0f %-5s %-5s %-9s %s'%(r.spectrum,r.tissue,r.fc_hat,r.DL_closed_nominal,r.f_DL_nominal,r.ident_closed,r.ident_fDL,r.three_way,r.ident_observed))
