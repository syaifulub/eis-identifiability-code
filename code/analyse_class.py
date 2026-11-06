import json,numpy as np,sys
d=json.load(open('class_out.json')); R=d['rows']; T=d['timing']
def m(pred,obs):
    p=np.asarray(pred,bool);o=np.asarray(obs,bool)
    TP=int((p&o).sum());TN=int((~p&~o).sum());FP=int((p&~o).sum());FN=int((~p&o).sum())
    se=TP/(TP+FN) if TP+FN else np.nan; sp=TN/(TN+FP) if TN+FP else np.nan
    return dict(n=len(p),TP=TP,TN=TN,FP=FP,FN=FN,acc=(TP+TN)/len(p),sens=se,spec=sp,bacc=np.nanmean([se,sp]))
obs=lambda S:[x['obs'] for x in S]
rules=[('Majority (non-ident.)',lambda x:False),('Eq. (9) plug-in',lambda x:x['pred8']),('A8 asymptote (proposed)',lambda x:x['predA8']),
       ('FIM/CRLB at Cole+L fit',lambda x:x['predFIM']),('Eq. (8) oracle',lambda x:x['oracle8']),('A8 oracle',lambda x:x['oracleLin'])]
# majority class = most frequent observed class overall
maj=np.mean(obs(R))>=0.5; rules[0]=('Majority (%s)'%('ident.' if maj else 'non-ident.'),lambda x,maj=maj:maj)
subsets=[('All',R),('In-domain (f_min>f_c_hat, Gamma>=2^(1/a))',[x for x in R if x['indom']]),
         ('Out-of-domain',[x for x in R if not x['indom']]),('Near threshold (0.5<=W_lin,true<=2)',[x for x in R if 0.5<=x['Wlin_true']<=2]),
         ('sigma 1%',[x for x in R if x['sigma']==0.01]),('sigma 2.5%',[x for x in R if x['sigma']==0.025]),('sigma 5%',[x for x in R if x['sigma']==0.05])]
out={}
for sn,S in subsets:
    print('\n==',sn,len(S),'prevalence %.2f'%np.mean(obs(S)))
    out[sn]={}
    for rn,fn in rules:
        mm=m([fn(x) for x in S],obs(S)); out[sn][rn]=mm
        print('  %-26s TP %3d TN %3d FP %3d FN %3d acc %.3f sens %.3f spec %.3f bacc %.3f'%(rn,mm['TP'],mm['TN'],mm['FP'],mm['FN'],mm['acc'],mm['sens'],mm['spec'],mm['bacc']))
print('\ntiming (median s):',T)
# detection-limit accuracy: true detection limit = oracle A8 at true asymptote? compare DL8 and DLA8 to observed transition per truth/sigma
json.dump(out,open('class_metrics.json','w'),default=float,indent=1)
