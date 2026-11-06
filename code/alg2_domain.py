"""Domain check for Algorithm 2 (added after the measured tetrapolar test): refer a band choice when the R-CPE-L
exponent of the pilot lies at a bound (0.3 or 1.0), i.e. when the pilot asymptote is not estimable.
Re-evaluates the existing simulation output (alg2_out.json, class_out.json) without new simulations."""
import json, numpy as np
A=json.load(open('alg2_out.json')); R=json.load(open('class_out.json'))['rows']
obs={}
for x in R:
    if x['sigma']==0.025: obs.setdefault((x['truth'],x['fmin']),[]).append(x['obs'])
obs={k:np.mean(v)>=0.5 for k,v in obs.items()}
rng=np.random.default_rng(3); sgn={t:rng.choice([-1,1]) for t in range(60)}
def m(p,o):
    p=np.asarray(p,bool);o=np.asarray(o,bool);TP=int((p&o).sum());TN=int((~p&~o).sum());FP=int((p&~o).sum());FN=int((~p&o).sum())
    se=TP/(TP+FN) if TP+FN else np.nan;sp=TN/(TN+FP) if TN+FP else np.nan
    return dict(n=int(p.size),TP=TP,TN=TN,FP=FP,FN=FN,sens=se,spec=sp,BA=(se+sp)/2)
o=np.array([obs[(a['truth'],a['fmin'])] for a in A]); bound=np.array([a['alpha_pilot']<=0.3+1e-3 or a['alpha_pilot']>=1-1e-3 for a in A])
pid=np.array([a['pilot_ident'] for a in A])
out=dict(n=len(A),at_bound=int(bound.sum()),at_bound_lower=int(sum(a['alpha_pilot']<=0.301 for a in A)),
         at_bound_but_pilot_identified=int((bound&pid).sum()),identifiable_among_referred=float(o[bound&~pid].mean()) if (bound&~pid).any() else None)
for name,s in [('exact',0.0),('+-0.5 dec',0.5),('+-1 dec',1.0)]:
    sel=np.array([True if a['pilot_ident'] else (a['fc']*10**(s*sgn[a['truth']])>a['DLA8']) for a in A])
    keep=~(bound&~pid)
    out[name]=dict(all=m(sel,o),decided=m(sel[keep],o[keep]),referred=int((~keep).sum()))
print(json.dumps(out,indent=1,default=float)); json.dump(out,open('alg2_domain.json','w'),indent=1,default=float)
