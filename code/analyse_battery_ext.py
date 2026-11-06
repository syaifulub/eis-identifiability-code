"""Summary of the extended battery test (states of charge I-IX, 25/35/45 C; two cycles per cell: first and last).
Reference: unbounded profile likelihood of log10 tau1 on each truncated band (as for the state-V test).
Usage: python analyse_battery_ext.py battery_ext_all.json"""
import json, sys, numpy as np, pandas as pd
R = pd.read_json(sys.argv[1] if len(sys.argv) > 1 else 'battery_ext_all.json')
def cm(d, k, ref='obs_unb'):
    p = d[k].values.astype(bool); o = d[ref].values.astype(bool)
    TP = (p & o).sum(); TN = (~p & ~o).sum(); FP = (p & ~o).sum(); FN = (~p & o).sum()
    return dict(n=len(d), identifiable=int(o.sum()), FP=int(FP), FN=int(FN), BA=round(((TP / max(TP + FN, 1)) + (TN / max(TN + FP, 1))) / 2, 3))
def summary(d):
    d = d.copy(); d['dec'] = d.three != 'uncertain'; d['tw'] = d.three == 'likely'
    d['bias'] = np.log10(d.fc1_hat / d.fc1_full); I = d[d.obs_unb]; acc = I.bias.abs() <= 0.5; below = I.fmin > I.fc1_full; rule = I.fmin <= I.fc1_hat
    S = d.groupby(['file', 'cycle']).first()
    return dict(spectra=len(S), full_band_identifiable=int(S.full_ident.sum()), nrmse_median=round(float(S.nrmse_full.median()), 4),
                closed=cm(d, 'pred_closed'), f_DL=cm(d, 'pred_fDL'), three_way=dict(cm(d[d.dec], 'tw'), referred=int((~d.dec).sum())),
                identified=len(I), accurate=int(acc.sum()), within_band=f'{int(acc[~below].sum())}/{int((~below).sum())}',
                below_band=f'{int(acc[below].sum())}/{int(below.sum())}', rule_rejects_inaccurate=f'{int((~rule & ~acc).sum())}/{int((~acc).sum())}',
                rule_keeps_accurate=f'{int((rule & acc).sum())}/{int(acc.sum())}')
out = {'all': summary(R)}
for t, d in R.groupby('temp'): out[t] = summary(d)
S = R.groupby(['file', 'cycle']).first().reset_index()
good = S[(S.nrmse_full <= 0.03) & S.full_ident & (S.fc2_full > 64) & (S.fc2_full < 150000)][['file', 'cycle']]
out['good_fit_subset (NRMSE<=3%, tau1 identifiable on full band, ZARC2 not at a bound)'] = summary(R.merge(good, on=['file', 'cycle']))
out['nrmse_median_by_state'] = S.groupby('state').nrmse_full.median().round(4).to_dict()
for k, v in out.items(): print(k, v)
json.dump(out, open('battery_ext_metrics.json', 'w'), indent=1, default=str)
