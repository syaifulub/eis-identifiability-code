import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker as mt
plt.rcParams.update({'font.family':'Liberation Sans','mathtext.fontset':'custom','mathtext.rm':'Liberation Sans','mathtext.it':'Liberation Sans:italic','mathtext.bf':'Liberation Sans:bold','font.size':8,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,
  'axes.edgecolor':'#555','xtick.color':'#333','ytick.color':'#333','axes.labelcolor':'#222','pdf.fonttype':42})
import sys
d=json.load(open(sys.argv[1] if len(sys.argv)>1 else 'ee_relevance_output.json')); n=d['nomogram']
fm=np.array(n['fmins'])/1e3; sg=np.array(n['sigs'])*100; DLl=np.array(n['DLlin'],dtype=float); DL8=np.array(n['DL8'])
B='#2a78d6'; O='#eb6834'; INK='#222'; lev=[30,100,300,1000,3000]
# --- main: nomogram only, single column
fig,ax=plt.subplots(figsize=(3.0,2.75),constrained_layout=True)
ax.contour(fm,sg,DLl,levels=lev,colors=B,linewidths=1.4); ax.contour(fm,sg,DL8,levels=lev,colors=O,linewidths=1.0,linestyles='--')
lab={30:'30 Hz',100:'100 Hz',300:'300 Hz',1000:'1 kHz',3000:'3 kHz'}; rows={30:4.3,100:3.0,300:4.3,1000:3.0,3000:4.3}
for v in lev:
    iy=np.argmin(abs(sg-rows[v])); row=np.log10(DLl[iy]); ok=~np.isnan(row); x=np.interp(np.log10(v),row[ok],np.log10(fm[ok]))
    ax.text(10**x,sg[iy],lab[v],fontsize=7,color=INK,ha='center',va='center',bbox=dict(boxstyle='round,pad=0.12',fc='white',ec='none'))
ax.plot(10,1.8,'o',ms=5.5,mfc='white',mec=INK,mew=1.1)
ax.annotate('this study',(10,1.8),xytext=(-58,-26),textcoords='offset points',fontsize=8,color=INK,bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none'),arrowprops=dict(arrowstyle='-',lw=0.6,color='#555'))
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Lower band limit $f_{\min}$ (kHz)'); ax.set_ylabel(r'Effective relative error $\sigma$ (%)')
ax.set_xticks([0.1,0.3,1,3,10]); ax.set_xticklabels(['0.1','0.3','1','3','10']); ax.set_yticks([0.2,0.5,1,2,5]); ax.set_yticklabels(['0.2','0.5','1','2','5'])
ax.xaxis.set_minor_formatter(mt.NullFormatter()); ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.plot([],[],color=B,lw=1.4,label=r'operational, $W_{\mathrm{lin}}$'); ax.plot([],[],color=O,lw=1.0,ls='--',label='closed form, Eq. (9)')
ax.legend(loc='lower left',fontsize=8,frameon=False,title='Detection limit',title_fontsize=8)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.savefig('Fig6_nomogram.pdf'); fig.savefig('Fig6.eps'); fig.savefig('Fig6_nomogram.png',dpi=220)
# --- supplement: time trade-off
st={(r['fmin'],r['N'],r['cycles']):r['T_ms'] for r in d['sweep_time']}
tb=[st[(f,20,10)] for f in [10e3,3e3,1e3,300.]]; ib=[47,70,85,97]; t10=st[(10e3,20,10)]; sigs=[5,2.5,1,0.5]; ta=[t10*(2.5/s)**2 for s in sigs]; ia=[30,47,60,68]
fig,ax=plt.subplots(figsize=(4.2,2.8),constrained_layout=True)
ax.plot(tb,ib,'-o',color=B,lw=2,ms=6,mec='white',mew=1.2,label=r'Lower $f_{\min}$ (2.5% noise)')
ax.plot(ta,ia,'-s',color=O,lw=2,ms=6,mec='white',mew=1.2,label=r'Average more ($f_{\min}$ = 10 kHz)')
for t,i,l in zip(tb,ib,['10 kHz','3','1','0.3 kHz']): ax.annotate(l,(t,i),(4,-10),textcoords='offset points',fontsize=8,color=INK)
for t,i,l in zip(ta,ia,['5%','2.5%','1%','0.5%']): ax.annotate(l,(t,i),(4,4),textcoords='offset points',fontsize=8,color=INK)
ax.set_xscale('log'); ax.set_xticks([2,5,10,20,50,100,200]); ax.set_xticklabels(['2','5','10','20','50','100','200']); ax.xaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel('Sweep time (ms; 20 frequencies, 10 cycles each)'); ax.set_ylabel(r'Identifiable $\log_{10}\tau$ (% of truths)'); ax.set_ylim(20,102)
ax.grid(True,axis='y',color='#ddd',lw=0.5); ax.legend(loc='lower right',fontsize=8,frameon=False)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.savefig('FigS17_time_tradeoff.png',dpi=220)
