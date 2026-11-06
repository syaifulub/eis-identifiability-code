"""Numerical verification of the closed-form band criterion (Eqs. 8-11, Appendix A).
Representative R-CPE-L asymptote: medians of the 12 KK-consistent spectra (Table S45c).
Error models: Table S47 (all 51 sweeps; KK-only), perturbed rho_f (+0.2), AR(1) rho=0.6.
No measured spectra are used; all spectra are synthetic Cole+L spectra."""
import numpy as np, itertools, json, sys
from scipy.optimize import nnls, minimize_scalar
LN10=np.log(10); Z95=1.959964

def cole(f,Rinf,dR,tau,a,L):
    w=2*np.pi*f; return Rinf+dR/(1+(1j*w*tau)**a)+1j*w*L
def stack(c): return np.r_[c.real,c.imag]

def jac(fun,p,h=1e-6):
    r0=fun(p); J=np.zeros((r0.size,p.size))
    for i in range(p.size):
        d=np.zeros_like(p); d[i]=h*max(1,abs(p[i]))
        J[:,i]=(fun(p+d)-fun(p-d))/(2*d[i])
    return J
def proj(J): return np.eye(J.shape[0])-J@np.linalg.pinv(J)

def setup(f,Rinf,Q,a,L,fc):
    tau=1/(2*np.pi*fc); dR=tau**a/Q
    Zo=cole(f,Rinf,dR,tau,a,L); W=np.abs(Zo)
    # R-CPE-L Jacobian (nuisance: log10 Rinf, log10 Q, alpha, log10 L) at the asymptote
    def r4(p):
        R,q,al,l=10**p[0],10**p[1],p[2],10**p[3]
        Z=R+1/(q*(1j*2*np.pi*f)**al)+1j*2*np.pi*f*l
        return stack((Z-Zo)/W)
    p4=np.array([np.log10(Rinf),np.log10(Q),a,np.log10(L)])
    J4=jac(r4,p4)
    # Cole+L Jacobian (log10 Rinf, log10 Q, alpha, log10 L, log10 tau), Q held as parameter
    def r5(p):
        R,q,al,l,t=10**p[0],10**p[1],p[2],10**p[3],10**p[4]
        Z=cole(f,R,t**al/q,t,al,l); return stack((Z-Zo)/W)
    p5=np.r_[p4,np.log10(tau)]
    J5=jac(r5,p5)
    om=2*np.pi*f; w=(1j*om)**(-a)/Q; u=(1j*om*tau)**(-a)
    sc=(f.min()/fc)**(-a)
    ht=stack(w/W*(1j*om/om.min())**(-a))            # first order, Eq. (8)
    hx=stack((w*u/(1+u)**2)/W)/sc                    # exact derivative wrt ln(eta), same scaling
    return dict(tau=tau,dR=dR,Zo=Zo,W=W,J4=J4,J5=J5,ht=ht,hx=hx,u=u,sc=sc)

def widths(f,Rinf,Q,a,L,fc,sig):
    S=setup(f,Rinf,Q,a,L,fc); P=proj(S['J4']); r=(f.min()/fc)**(-a)
    D=np.linalg.norm(P@S['ht']); Dx=np.linalg.norm(P@S['hx'])
    W8=2*Z95*sig/(a*LN10*D)*(f.min()/fc)**a
    Wnc=2*Z95*sig/(a*LN10*Dx)*(f.min()/fc)**a          # refined derivative, no coupling
    C=np.linalg.inv(S['J5'].T@S['J5']); Ww=2*Z95*sig*np.sqrt(C[4,4])  # exact linearized (Wald)
    sinth=D/np.linalg.norm(S['ht'])
    bound=r*(2+r)/sinth; relD=abs(Dx-D)/D
    rem=np.max(np.abs(S['u'])/np.abs(1+S['u']))
    return dict(r=r,W8=W8,Wnc=Wnc,Wwald=Ww,sinth=sinth,relD=relD,bound=bound,rem=rem,S=S)

def expected_profile(f,Rinf,Q,a,L,fc,sig,lo=-7,hi=1,n=321):
    """Asymptotic expected profile: noise-free truth, Lambda(tau)=min RSS/sigma^2 (alpha free, R,dR,L >=0)."""
    tau0=1/(2*np.pi*fc); Zo=cole(f,Rinf,tau0**a/Q,tau0,a,L); W=np.abs(Zo); om=2*np.pi*f
    b=stack(Zo/W)
    def rss(lt):
        t=10**lt
        def inner(al):
            A=np.c_[stack(np.ones_like(f)/W),stack(1/(1+(1j*om*t)**al)/W),stack(1j*om/W)]
            x,res=nnls(A,b); return res**2
        o=minimize_scalar(inner,bounds=(0.3,1.0),method='bounded',options={'xatol':1e-6})
        return o.fun
    g=np.linspace(lo,hi,n); L_=np.array([rss(x) for x in g])/sig**2
    L_=L_-L_.min(); inside=g[L_<=3.84]
    openlo=L_[0]<=3.84; openhi=L_[-1]<=3.84
    return dict(width=inside.max()-inside.min(), open=bool(openlo or openhi))

def Sigma(N,f,s0,g,rf,rri,k):
    Rf=rf**np.abs(np.subtract.outer(np.arange(N),np.arange(N)))
    Rri=np.array([[1,rri],[rri,1]])
    sd=s0*(f/f[0])**g; V=np.diag(np.r_[sd,k*sd])
    return V@np.kron(Rri,Rf)@V

def kappas(f,Rinf,Q,a,L,fc,Sig):
    S=setup(f,Rinf,Q,a,L,fc); N=f.size
    P4=proj(S['J4']); P5=proj(S['J5'])
    sig2=np.trace(P5@Sig)/(2*N-5)                     # E[RSS]/(2N-5): the sigma^2 used in Eq. (8)
    h=S['ht']; Ph=P4@h
    kS=np.sqrt(Ph@Sig@Ph/(sig2*(Ph@Ph)))              # Eq. (A6)
    Li=np.linalg.inv(np.linalg.cholesky(Sig))
    PS=proj(Li@S['J4']); DS=np.linalg.norm(PS@Li@h)
    gls_ratio=(1/DS)/(np.sqrt(sig2)/np.linalg.norm(Ph))   # W_GLS / W_Eq8
    J5=S['J5']; A=np.linalg.inv(J5.T@J5)
    sand=A@J5.T@Sig@J5@A; kfull=np.sqrt(sand[4,4]/(sig2*A[4,4]))   # full-model sandwich / iid Wald
    return dict(kS=kS,kfull=kfull,gls=gls_ratio)

out={}
fin=np.arange(10e3,100.1e3,10e3)                     # instrument band, N=10
base=dict(Rinf=2211.,Q=7.4e-8,a=0.69,L=1.14e-3); sig=0.018
# 1) accuracy of the closed form vs exact linearized width and expected profile (instrument band)
rows=[]
for r in [0.05,0.1,0.2,0.3,0.4,0.5,0.7,0.9]:
    fc=fin.min()*r**(1/base['a'])
    w=widths(fin,fc=fc,sig=sig,**base); pr=expected_profile(fin,fc=fc,sig=sig,**base)
    rows.append(dict(r=r,fc=round(fc),W8=w['W8'],Wnc=w['Wnc'],Wwald=w['Wwald'],prof=pr['width'],open=pr['open'],
        sinth=w['sinth'],relD=w['relD'],bound=w['bound'],rem=w['rem'],
        etaSE=base['a']*LN10*w['Wwald']/(2*Z95)))
out['accuracy_instrument']=rows
# 2) remainder bound over alpha and r (Step 1)
worst=0
for a in np.linspace(0.3,1.0,15):
    for r in np.linspace(0.01,0.9,40):
        u=r*np.exp(-1j*a*np.pi/2)*np.logspace(0,-1.5,50)**a
        worst=max(worst,np.max((np.abs(u)/np.abs(1+u))/r))
out['remainder_ratio_max']=worst
# 3) sandwich and GLS factors for the error models (instrument band)
models={'estimated (51 sweeps)':(0.110,0.009,0.180,-0.194,0.459),
        'KK-only (27 sweeps)':(0.012,-0.063,-0.057,0.300,1.904),
        'rho_f + 0.2':(0.110,0.009,0.380,-0.194,0.459),
        'AR(1), rho=0.6':(0.110,0.0,0.6,0.0,1.0),
        'white, equal':(0.110,0.0,0.0,0.0,1.0)}
qs=dict(Rinf=[1396.,2505.],Q=[3.13e-8,18.2e-8],a=[0.652,0.843],L=[0.92e-3,1.64e-3])
kres={}
for name,(s0,g,rf,rri,k) in models.items():
    Sg=Sigma(10,fin,s0,g,rf,rri,k); vals=[]
    for R,Qv,a,L in itertools.product(*qs.values()):
        for r in [0.1,0.3,0.5]:
            fc=fin.min()*r**(1/a); vals.append(kappas(fin,R,Qv,a,L,fc,Sg))
    med=kappas(fin,fc=fin.min()*0.3**(1/0.69),Sig=Sg,**base)
    kres[name]=dict(kS_med=med['kS'],kS_rng=[min(v['kS'] for v in vals),max(v['kS'] for v in vals)],
                   kfull_med=med['kfull'],kfull_rng=[min(v['kfull'] for v in vals),max(v['kfull'] for v in vals)],
                   gls_med=med['gls'],gls_rng=[min(v['gls'] for v in vals),max(v['gls'] for v in vals)])
out['sandwich']=kres
# 4) band-design bands (log grid, N=20): first-order vs exact linearized width at r=0.3 and 0.5
bd=[]
for fmin in [10e3,3e3,1e3,0.3e3]:
    fl=np.logspace(np.log10(fmin),5,20)
    for r in [0.3,0.5]:
        fc=fmin*r**(1/0.69); w=widths(fl,fc=fc,sig=0.025,**base)
        bd.append(dict(fmin=fmin,r=r,ratio8=w['W8']/w['Wwald'],rationc=w['Wnc']/w['Wwald'],sinth=w['sinth']))
out['band_design']=bd
json.dump(out,open(sys.argv[1] if len(sys.argv)>1 else 'verify_out.json','w'),indent=1,default=float)
for k,v in out.items(): print(k); print(json.dumps(v,indent=0,default=lambda x: round(float(x),3))[:3000])
