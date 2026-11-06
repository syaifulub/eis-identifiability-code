"""Stability of the bootstrap critical value c* (95th percentile of the profile statistic at the true value)
with respect to the number of bootstrap replicates B. R-CPE-L (M11) truth = median KK-consistent descriptors;
instrument band; errors N(0, Sigma) with the structure estimated from the 51 sweeps (Table S47), scaled to
an rms relative error of 0.018 per stacked component."""
import numpy as np, json, sys
from scipy.optimize import nnls, minimize_scalar
rng=np.random.default_rng(11)
f=np.arange(10e3,100.1e3,10e3); w=2*np.pi*f; N=10; n=2*N
R0,Q0,a0,L0=2211.,7.4e-8,0.69,1.14e-3
Z0=R0+1/(Q0*(1j*w)**a0)+1j*w*L0; W=np.abs(Z0)
def stack(c): return np.r_[c.real,c.imag]
s0,g,rf,rri,k=0.110,0.009,0.180,-0.194,0.459
Rf=rf**np.abs(np.subtract.outer(np.arange(N),np.arange(N))); sd=s0*(f/f[0])**g; V=np.diag(np.r_[sd,k*sd])
Sig=V@np.kron(np.array([[1,rri],[rri,1]]),Rf)@V; Sig*=0.018**2/(np.trace(Sig)/n); Lc=np.linalg.cholesky(Sig)
def cols(al): return [stack(np.ones(N)/W),stack((1j*w)**(-al)/W),stack(1j*w/W)]
def rss_free(b):
    def inner(al):
        x,res=nnls(np.column_stack(cols(al)),b); return res**2
    return minimize_scalar(inner,bounds=(0.3,1.0),method='bounded',options={'xatol':1e-8}).fun
def rss_fix_alpha(b): x,res=nnls(np.column_stack(cols(a0)),b); return res**2
def rss_fix_Q(b):
    def inner(al):
        c=cols(al); bb=b-c[1]*(1/Q0); x,res=nnls(np.column_stack([c[0],c[2]]),bb); return res**2
    return minimize_scalar(inner,bounds=(0.3,1.0),method='bounded',options={'xatol':1e-8}).fun
def rss_fix_R(b):
    def inner(al):
        c=cols(al); bb=b-c[0]*R0; x,res=nnls(np.column_stack([c[1],c[2]]),bb); return res**2
    return minimize_scalar(inner,bounds=(0.3,1.0),method='bounded',options={'xatol':1e-8}).fun
B=10000; lam={'alpha':[],'log10Q':[],'log10Rinf':[]}
for i in range(B):
    e=Lc@rng.standard_normal(n); b=stack(Z0/W)+e
    r0=rss_free(b)
    lam['alpha'].append(n*np.log(rss_fix_alpha(b)/r0)); lam['log10Q'].append(n*np.log(rss_fix_Q(b)/r0)); lam['log10Rinf'].append(n*np.log(rss_fix_R(b)/r0))
out={}
for kk,v in lam.items():
    v=np.maximum(np.array(v),0); ref=np.percentile(v,95)
    res={'ref_B10000':ref}
    for b_ in [200,500,1000,2000]:
        qs=[np.percentile(rng.choice(v,b_,replace=False),95) for _ in range(400)]
        res[f'B{b_}_mean']=float(np.mean(qs)); res[f'B{b_}_sd']=float(np.std(qs,ddof=1)); res[f'B{b_}_rel_sd']=float(np.std(qs,ddof=1)/ref)
        # coverage implication: P(chi-like stat <= c*) for c* at mean+-1sd
        res[f'B{b_}_cov_lo']=float(np.mean(v<=np.mean(qs)-np.std(qs))); res[f'B{b_}_cov_hi']=float(np.mean(v<=np.mean(qs)+np.std(qs)))
    out[kk]=res; print(kk,{a:round(b,3) for a,b in res.items()})
json.dump(out,open(sys.argv[1],'w'),indent=1)
