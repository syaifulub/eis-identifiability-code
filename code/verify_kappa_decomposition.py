import numpy as np, importlib.util, sys
spec=importlib.util.spec_from_file_location('v','verify_criteria.py')
src=open('verify_criteria.py').read().split('out={}')[0]
ns={}; exec(src,ns)
fin=np.arange(10e3,100.1e3,10e3); base=dict(Rinf=2211.,Q=7.4e-8,a=0.69,L=1.14e-3)
fc=fin.min()*0.3**(1/0.69)
for name,par in {'full estimated':(0.11,0.009,0.18,-0.194,0.459),'only kappa=0.459':(0.11,0,0,0,0.459),'only rho_f=0.18':(0.11,0,0.18,0,1),'only rho_RI=-0.194':(0.11,0,0,-0.194,1),'only gamma':(0.11,0.009,0,0,1)}.items():
    k=ns['kappas'](fin,fc=fc,Sig=ns['Sigma'](10,fin,*par),**base); print(name, {a:round(b,3) for a,b in k.items()})
S=ns['setup'](fin,fc=fc,**base); P=ns['proj'](S['J4']); Ph=P@S['ht']; print('share of ||Ph||^2 in real part', round(float(np.sum(Ph[:10]**2)/np.sum(Ph**2)),3))
