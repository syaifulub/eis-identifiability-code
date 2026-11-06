import json,numpy as np
A=json.load(open('alg2_out.json')); R=json.load(open('class_out.json'))['rows']
obs={}
for x in R:
    if x['sigma']==0.025: obs.setdefault((x['truth'],x['fmin']),[]).append(x['obs'])
obs={k:np.mean(v)>=0.5 for k,v in obs.items()}
rng=np.random.default_rng(3); sgn={t:rng.choice([-1,1]) for t in range(60)}
def m(p,o):
    p=np.asarray(p,bool);o=np.asarray(o,bool);TP=int((p&o).sum());TN=int((~p&~o).sum());FP=int((p&~o).sum());FN=int((~p&o).sum())
    se=TP/(TP+FN);sp=TN/(TN+FP);return (TP,TN,FP,FN,round((TP+TN)/len(p),3),round(se,3),round(sp,3),round((se+sp)/2,3))
o=[obs[(a['truth'],a['fmin'])] for a in A]
print('n',len(A),'prevalence',np.mean(o),'pilot ident',np.mean([a['pilot_ident'] for a in A[::4]]))
res={}
for name,fcp in [('exact',lambda a:a['fc']),('+-0.5 dec',lambda a:a['fc']*10**(0.5*sgn[a['truth']])),('+-1 dec',lambda a:a['fc']*10**(1.0*sgn[a['truth']])),
                 ('pilot estimate',lambda a:a['fc_pilot'])]:
    for rule in ['DLA8','DL8']:
        sel=[(fcp(a)>a[rule]) if np.isfinite(fcp(a)) else None for a in A]
        idx=[i for i,s in enumerate(sel) if s is not None]
        r=m([sel[i] for i in idx],[o[i] for i in idx]); res[name+'|'+rule]=dict(n=len(idx),cm=r); print('%-15s %-5s n=%d'%(name,rule,len(idx)),r)
# conservative margin: require f_c^exp*10^-delta > DL with delta = prior uncertainty
for d,name in [(0.5,'+-0.5 dec, margin'),(1.0,'+-1 dec, margin')]:
    sel=[a['fc']*10**(d*sgn[a['truth']])*10**(-d)>a['DLA8'] for a in A]; r=m(sel,o); res[name]=dict(n=len(A),cm=r); print('%-20s'%name,r)
# no prior: report DL; pilot-identifiable truths
print('median sig_hat',np.median([a['sig_hat'] for a in A[::4]]))
print('DL A8/DL8 ratio median, IQR',np.percentile([a['DLA8']/a['DL8'] for a in A if np.isfinite(a['DL8']) and a['DL8']>0],[25,50,75]))
json.dump(res,open('alg2_metrics.json','w'),indent=1,default=float)
print('\n--- Algorithm 2 (revised): pilot-identified -> adequate; else f_c^exp > DL_A8 ---')
res2={}
for name,fcp in [('exact',lambda a:a['fc']),('+-0.5 dec',lambda a:a['fc']*10**(0.5*sgn[a['truth']])),('+-1 dec',lambda a:a['fc']*10**(1.0*sgn[a['truth']]))]:
    for rule in ['DLA8','DL8']:
        sel=[True if a['pilot_ident'] else (fcp(a)>a[rule]) for a in A]
        r=m(sel,o); res2[name+'|'+rule]=r; print('%-10s %-5s'%(name,rule),r)
# subset where pilot is not identified (design actually needed)
idx=[i for i,a in enumerate(A) if not a['pilot_ident']]
print('pilot not identified: n',len(idx),'prevalence',np.mean([o[i] for i in idx]))
for name,fcp in [('exact',lambda a:a['fc']),('+-0.5 dec',lambda a:a['fc']*10**(0.5*sgn[a['truth']])),('+-1 dec',lambda a:a['fc']*10**(1.0*sgn[a['truth']]))]:
    for rule in ['DLA8','DL8']:
        r=m([fcp(A[i])>A[i][rule] for i in idx],[o[i] for i in idx]); res2['nopilot|'+name+'|'+rule]=r; print('  %-10s %-5s'%(name,rule),r)
json.dump(dict(raw=res,revised=res2),open('alg2_metrics.json','w'),indent=1,default=float)
