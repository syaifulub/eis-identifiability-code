"""Cluster-bootstrap (over the 60 truths) 95% intervals for balanced accuracy and FP/FN rates (Table 5 and Sect. 3.6)."""
import json,numpy as np
R=json.load(open('class_out.json'))['rows']; A=json.load(open('alg2_out.json'))
rng=np.random.default_rng(5); B=2000
def ba(p,o):
    p=np.asarray(p,bool);o=np.asarray(o,bool)
    se=(p&o).sum()/max(o.sum(),1); sp=(~p&~o).sum()/max((~o).sum(),1); return (se+sp)/2
rules={'closed':'pred8','oper':'predA8','fim':'predFIM','closed_true':'oracle8','wlin_true':'oracleLin'}
subs={'all':lambda x:True,'indom':lambda x:x['indom'],'near':lambda x:0.5<=x['Wlin_true']<=2}
by={t:[x for x in R if x['truth']==t] for t in range(60)}
out={}
for sn,sf in subs.items():
    for rn,k in rules.items():
        pt=ba([x[k] for x in R if sf(x)],[x['obs'] for x in R if sf(x)]); bs=[]
        for _ in range(B):
            ts=rng.integers(0,60,60); S=[x for t in ts for x in by[t] if sf(x)]
            bs.append(ba([x[k] for x in S],[x['obs'] for x in S]))
        out[sn+'|'+rn]=(pt,*np.percentile(bs,[2.5,97.5])); print(sn,rn,'BA %.2f (%.2f-%.2f)'%out[sn+'|'+rn])
# paired difference FIM - operational, all and indom
for sn in ['all','indom']:
    d=[]
    for _ in range(B):
        ts=rng.integers(0,60,60); S=[x for t in ts for x in by[t] if subs[sn](x)]; o=[x['obs'] for x in S]
        d.append(ba([x['predFIM'] for x in S],o)-ba([x['predA8'] for x in S],o))
    print(sn,'BA FIM - operational: %.2f (%.2f to %.2f)'%(np.mean(d),*np.percentile(d,[2.5,97.5])))
# Algorithm 2
obs={}
for x in R:
    if x['sigma']==0.025: obs.setdefault((x['truth'],x['fmin']),[]).append(x['obs'])
obs={k:np.mean(v)>=0.5 for k,v in obs.items()}
r2=np.random.default_rng(3); sgn={t:r2.choice([-1,1]) for t in range(60)}
byA={t:[a for a in A if a['truth']==t] for t in range(60)}
for d in [0,0.5,1.0]:
    pr=lambda a:True if a['pilot_ident'] else a['fc']*10**(d*sgn[a['truth']])>a['DLA8']
    pt=ba([pr(a) for a in A],[obs[(a['truth'],a['fmin'])] for a in A]); bs=[]
    for _ in range(B):
        ts=rng.integers(0,60,60); S=[a for t in ts for a in byA[t]]
        bs.append(ba([pr(a) for a in S],[obs[(a['truth'],a['fmin'])] for a in S]))
    print('alg2 delta',d,'BA %.2f (%.2f-%.2f)'%(pt,*np.percentile(bs,[2.5,97.5])))
