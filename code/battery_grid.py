"""Expanded synthetic transfer test (M13): R0 + ZARC (+ optional Warburg) + L over a grid of
f_c, alpha, Warburg magnitude, f_min and noise; Eq. (8) and exact linearized width vs expected profile."""
import numpy as np, json, sys, itertools
src=open('ee_relevance.py').read().split('out={}')[0]
ns={}; exec(src,ns)
make=ns['make']; criterion=ns['criterion']; exp_profile=ns['exp_profile']; warburg=ns['warburg']
Zb,Zba=ns['make'](warburg,1)
rows=[]
for fc,a,AW,fm,sg in itertools.product([5.,20.,80.],[0.6,0.8,0.95],[1e-6,2e-3],[10.,30.,100.,300.],[0.002,0.005]):
    R0=0.020; Rz=0.015; L=0.5e-6; tau=1/(2*np.pi*fc); Q=tau**a/Rz
    n=int(round(10*np.log10(1e4/fm)))+1; f=np.logspace(np.log10(fm),4,n)
    c=criterion(f,Zb,Zba,R0,Q,a,np.array([AW]),L,fc,sg)
    pr=exp_profile(f,Zb,R0,Q,a,np.array([AW]),L,fc,sg,[lambda w: 1/np.sqrt(1j*w)],lo=np.log10(tau)-4,hi=np.log10(tau)+4,n=241)
    idp=(not pr['open']) and pr['width']<1
    rows.append(dict(fc=fc,alpha=a,AW=AW,fmin=fm,sigma=sg,r=c['r'],sinth=c['sinth'],W8=c['W8'],Wlin=c['Wlin'],prof=pr['width'],open=pr['open'],
                     id8=bool(c['W8']<1),idlin=bool(c['Wlin']<1),idprof=bool(idp)))
json.dump(rows,open(sys.argv[1],'w'),default=float)
R=rows; dom=[x for x in R if x['r']<=0.5]
for name,sub in [('all',R),('in-domain r<=0.5',dom),('out r>0.5',[x for x in R if x['r']>0.5])]:
    if not sub: continue
    a8=np.mean([x['id8']==x['idprof'] for x in sub]); al=np.mean([x['idlin']==x['idprof'] for x in sub]); pos=np.mean([x['idprof'] for x in sub])
    print(name,len(sub),'agree Eq8 %.3f Wlin %.3f prevalence %.2f'%(a8,al,pos))
