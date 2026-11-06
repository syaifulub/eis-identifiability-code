import json, numpy as np, pandas as pd, sys
R=pd.DataFrame(json.load(open(sys.argv[1] if len(sys.argv)>1 else 'battery_measured.json')))
R['below']=R.fmin>R.fc1_full; R['tw_dec']=R.three!='uncertain'; R['tw_pred']=R.three=='likely'
def cm(d,k,ref):
    p=d[k].values.astype(bool); o=d[ref].values.astype(bool)
    TP=(p&o).sum();TN=(~p&~o).sum();FP=(p&~o).sum();FN=(~p&o).sum(); se=TP/max(TP+FN,1); sp=TN/max(TN+FP,1)
    return dict(n=int(len(d)),prev=round(float(o.mean()),3),FP=int(FP),FN=int(FN),acc=round(float((TP+TN)/max(len(d),1)),3),bacc=round(float((se+sp)/2),3))
out={'n_spectra':int(R.groupby(['file','cycle']).ngroups),'fc1_full_range':[float(R.fc1_full.min()),float(R.fc1_full.max())],
     'alpha1_range':[float(R.alpha1_full.min()),float(R.alpha1_full.max())],'fc2_range':[float(R.fc2_full.min()),float(R.fc2_full.max())],
     'nrmse_full_median':float(R.groupby(['file','cycle']).nrmse_full.first().median()),'sigma_band_median':float(R.sigma.median()),
     'full_ident':int(R.groupby(['file','cycle']).full_ident.first().sum())}
for ref in ['obs_unb','obs']:
    for nm,d in [('all',R),('dispersion below band',R[R.below]),('dispersion within band',R[~R.below])]:
        for k in ['pred_closed','pred_fDL']: out[f'{ref}|{nm}|{k}']=cm(d,k,ref)
        D=d[d.tw_dec]; out[f'{ref}|{nm}|three_way']=dict(cm(D,'tw_pred',ref),referred=int((~d.tw_dec).sum()),ref_ident=round(float(d[~d.tw_dec][ref].mean()),3) if (~d.tw_dec).sum() else None)
for k,v in out.items(): print(k,v)
# correctness of identified tau: truncated interval contains full-band tau
I=R[R.obs]; print('identified (bounded) intervals containing full-band tau1: %d/%d'%(I.truth_in.sum(),len(I)))
print('identifiable (unbounded) by fmin/fc1 bins:'); R['ratio']=R.fmin/R.fc1_full
print(R.groupby(pd.cut(R.ratio,[0,0.3,1,2,4,100]),observed=True)[['obs_unb','obs','pred_closed','pred_fDL']].mean().round(2))
json.dump(out,open('battery_measured_metrics.json','w'),indent=1,default=float)
# --- accuracy-based reference: identified AND |log10(fc1_hat/fc1_full)| <= 0.5 decade
R['bias']=np.log10(R.fc1_hat/R.fc1_full)
for ref in ['obs_unb','obs']:
    R[ref+'_acc']=R[ref]&(R.bias.abs()<=0.5)
    for nm,d in [('all',R),('dispersion below band',R[R.below])]:
        for k in ['pred_closed','pred_fDL']: out[f'{ref}_acc|{nm}|{k}']=cm(d,k,ref+'_acc')
        D=d[d.tw_dec]; out[f'{ref}_acc|{nm}|three_way']=dict(cm(D,'tw_pred',ref+'_acc'),referred=int((~d.tw_dec).sum()))
        out[f'{ref}|{nm}|biased_identified']=int((d[ref]&(d.bias.abs()>0.5)).sum())
        print(ref+'_acc',nm,{k:out[f'{ref}_acc|{nm}|{k}'] for k in ['pred_closed','pred_fDL','three_way']},'biased identified',out[f'{ref}|{nm}|biased_identified'])
out['bias_median_by_fmin']=R.groupby('fmin').bias.median().round(3).to_dict()
json.dump(out,open('battery_measured_metrics.json','w'),indent=1,default=float)
