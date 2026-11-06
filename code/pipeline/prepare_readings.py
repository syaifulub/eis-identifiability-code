"""Convert a table of individual impedance readings from the public dataset (Falagario et al., Zenodo
record 20408128, PreparedData.zip, CC BY 4.0) into the tidy input format of run_pipeline.py.

The column names of the source files are passed as options, so the script does not assume a file layout:
  python prepare_readings.py SOURCE.csv readings.csv --recording <col> --tissue <col> --frequency <col>
         (--real <col> --imag <col> | --magnitude <col> --phase-deg <col>) [--order <col>] [--mode <col>]
         [--animal <col>] [--probe <col>] [--frequency-scale 1000]
Several source files can be given (comma-separated); they are concatenated. Rows keep their file order if no
--order column is given (the acquisition order is needed to split complete sweeps)."""
import argparse, numpy as np, pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument('source'); ap.add_argument('out')
for c in ['recording','tissue','frequency','real','imag','magnitude','phase-deg','order','mode','animal','probe']: ap.add_argument('--'+c)
ap.add_argument('--frequency-scale',type=float,default=1.0,help='multiply the frequency column to obtain Hz')
a=ap.parse_args()
df=pd.concat([pd.read_csv(s) for s in a.source.split(',')],ignore_index=True)
out=pd.DataFrame({'recording':df[a.recording].astype(str),'tissue':df[a.tissue].astype(str).str.lower(),
                  'frequency_hz':df[a.frequency].astype(float)*a.frequency_scale})
if a.real and a.imag: out['z_real']=df[a.real].astype(float); out['z_imag']=df[a.imag].astype(float)
else:
    ph=np.deg2rad(df[a.phase_deg].astype(float)); m=df[a.magnitude].astype(float); out['z_real']=m*np.cos(ph); out['z_imag']=m*np.sin(ph)
out['order']=df[a.order].values if a.order else np.arange(len(df))
out['mode']=df[a.mode].astype(str) if a.mode else 'scan'
for c in ['animal','probe']:
    out[c]=df[getattr(a,c)].values if getattr(a,c) else ''
out.to_csv(a.out,index=False); print(len(out),'readings ->',a.out)
