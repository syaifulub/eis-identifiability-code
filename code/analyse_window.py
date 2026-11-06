import json, numpy as np, pandas as pd, sys
R=pd.DataFrame(json.load(open(sys.argv[1] if len(sys.argv)>1 else 'window_out2.json')))
def cm(d,k,ref):
    p=d[k].values.astype(bool); o=d[ref].values.astype(bool)
    TP=(p&o).sum();TN=(~p&~o).sum();FP=(p&~o).sum();FN=(~p&o).sum()
    se=TP/max(TP+FN,1); sp=TN/max(TN+FP,1); return dict(n=len(d),prev=round(o.mean(),3),FP=int(FP),FN=int(FN),acc=round((TP+TN)/len(d),3),bacc=round((se+sp)/2,3))
out={}
for ref in ['idprof_unb','idprof']:
    for nm,d in [('all',R),('r<=0.5',R[R.r<=0.5]),('r>0.5',R[R.r>0.5])]:
        for k in ['id8','idlin']:
            out[f'{ref}|{nm}|{k}']=cm(d,k,ref); print(ref,nm,k,out[f'{ref}|{nm}|{k}'])
for lo,hi in [(0,0.1),(0.1,0.3),(0.3,0.5),(0.5,1)]:
    d=R[(R.r>lo)&(R.r<=hi)&R.closed_unb]
    if len(d): print('r %.1f-%.1f closed-unb n=%d  W8/Wunb %.2f  Wlin/Wunb %.2f'%(lo,hi,len(d),(d.W8/d.Wprof_unb).median(),(d.Wlin/d.Wprof_unb).median()))
print('ident (bounded / unbounded) by ratio:'); print(R.groupby('ratio')[['idprof','idprof_unb']].mean().round(2))
print('bounded-closed but unbounded-not identifiable:',int((R.idprof & ~R.idprof_unb).sum()),'of',int(R.idprof.sum()))
json.dump(out,open('window_metrics.json','w'),indent=1)
