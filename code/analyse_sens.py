import json,numpy as np
S=json.load(open('alg2_sens.json')); A=json.load(open('alg2_out.json')); R=json.load(open('class_out.json'))['rows']
r2=np.random.default_rng(3); sgn={t:r2.choice([-1,1]) for t in range(60)}
def m(p,o):
    p=np.asarray(p,bool);o=np.asarray(o,bool);TP=int((p&o).sum());TN=int((~p&~o).sum());FP=int((p&~o).sum());FN=int((~p&o).sum())
    se=TP/max(TP+FN,1);sp=TN/max(TN+FP,1);return dict(n=len(p),prev=float(o.mean()),TP=TP,TN=TN,FP=FP,FN=FN,sens=se,spec=sp,bacc=(se+sp)/2)
rows=[]
# white-noise mismatch: predictions from the 2.5% pilot (alg2_out), reference at 1% / 5%
for sig,nm in [(0.025,'matched (2.5%)'),(0.01,'band noise 1%'),(0.05,'band noise 5%')]:
    obs={}
    for x in R:
        if x['sigma']==sig: obs.setdefault((x['truth'],x['fmin']),[]).append(x['obs'])
    obs={k:np.mean(v)>=0.5 for k,v in obs.items()}
    for a in A: a['_o_'+nm]=obs[(a['truth'],a['fmin'])]
    rows.append((nm,A,'_o_'+nm))
for sc,nm in [('lowf','noise rising below 10 kHz'),('pol10','electrode CPE 10%'),('pol50','electrode CPE 50%')]:
    rows.append((nm,[x for x in S if x['scen']==sc],'obs'))
out={}
for nm,L,ok in rows:
    for d in [0,0.5]:
        pred=[True if a['pilot_ident'] else a['fc']*10**(d*sgn[a['truth']])>a['DLA8'] for a in L]
        r=m(pred,[a[ok] for a in L]); 
        # three-way among unsettled
        U=[a for a in L if not a['pilot_ident']]; yes=no=ref=0; err=0
        for a in U:
            fe=a['fc']*10**(d*sgn[a['truth']]); c=fe>a['DL8']; p=fe>a['DLA8']
            if c and p: yes+=1; err+= (not a[ok])
            elif not c and not p: no+=1; err+= bool(a[ok])
            else: ref+=1
        r.update(tw_decided=yes+no,tw_ref=ref,tw_err=err,pilot_ident=sum(a['pilot_ident'] for a in L)//4, sig_hat=float(np.median([a['sig_hat'] for a in L])))
        out[nm+'|%.1f'%d]=r
        print('%-28s d=%.1f prev %.2f FP %3d FN %3d sens %.2f spec %.2f BA %.2f | 3-way decided %d ref %d err %d | sig_hat %.3f'%(nm,d,r['prev'],r['FP'],r['FN'],r['sens'],r['spec'],r['bacc'],r['tw_decided'],r['tw_ref'],r['tw_err'],r['sig_hat']))
json.dump(out,open('alg2_sens_metrics.json','w'),indent=1,default=float)
