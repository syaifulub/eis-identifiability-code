"""Three-way rule (reviewer P1): 'likely' if both the closed form and f_DL say identifiable, 'unlikely' if both say
not identifiable, otherwise 'uncertain -> refer' (profile likelihood or measurement). Classification study and Algorithm 2."""
import json,numpy as np
R=json.load(open('class_out.json'))['rows']; A=json.load(open('alg2_out.json'))
def tw(pairs):
    # pairs: (closed, oper, obs)
    yes=[o for c,p,o in pairs if c and p]; no=[o for c,p,o in pairs if (not c) and (not p)]; ref=[o for c,p,o in pairs if c!=p]
    TP=sum(yes); FP=len(yes)-TP; TN=len(no)-sum(no); FN=sum(no); n=len(pairs)
    se=TP/max(TP+FN,1); sp=TN/max(TN+FP,1)
    return dict(n=n,decided=len(yes)+len(no),referred=len(ref),ref_frac=len(ref)/n,ref_ident=float(np.mean(ref)) if ref else np.nan,
                TP=TP,TN=TN,FP=FP,FN=FN,sens=se,spec=sp,bacc=(se+sp)/2,acc=(TP+TN)/max(len(yes)+len(no),1))
res={}
for nm,f in [('in-domain',lambda x:x['indom']),('near threshold',lambda x:0.5<=x['Wlin_true']<=2),('all',lambda x:True)]:
    S=[x for x in R if f(x)]; r=tw([(x['pred8'],x['predA8'],x['obs']) for x in S]); res['class|'+nm]=r
    print(nm,{k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
obs={}
for x in R:
    if x['sigma']==0.025: obs.setdefault((x['truth'],x['fmin']),[]).append(x['obs'])
obs={k:np.mean(v)>=0.5 for k,v in obs.items()}
r2=np.random.default_rng(3); sgn={t:r2.choice([-1,1]) for t in range(60)}
for d in [0,0.5,1.0]:
    P=[]
    for a in A:
        if a['pilot_ident']: continue
        fe=a['fc']*10**(d*sgn[a['truth']]); P.append((fe>a['DL8'],fe>a['DLA8'],obs[(a['truth'],a['fmin'])]))
    r=tw(P); res['alg2|%.1f'%d]=r; print('alg2 delta',d,{k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
json.dump(res,open('threeway_out.json','w'),indent=1,default=float)
