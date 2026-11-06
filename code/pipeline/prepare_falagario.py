"""Convert the public dataset (Falagario et al., Zenodo record 20408128, PreparedData.zip, CC BY 4.0) into the tidy
input format of run_pipeline.py.

PreparedData/byOriginalSubset/subset <X>/pr_<timestamp>_SP_<probe>_<config>_<tissue>.txt contain one reading per
line: device state; scan frequency (kHz); |Z| (Ohm); phase (deg); resistance (Ohm); reactance (Ohm); scaled time
stamp; tissue name. Lines starting with '%' are headers. The analysis uses the yellow bipolar frequency-scan
recordings of subsets A (first animal) and D (second animal). The acquisition order is the line order (the scaled
time stamp is not monotone in subset A).

Usage: python prepare_falagario.py <path to PreparedData> readings.csv"""
import sys, os, re, glob
import pandas as pd
root = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else 'readings.csv'
SUBSETS = {'subset A': 1, 'subset D': 2}
rows = []
for sub, animal in SUBSETS.items():
    for fn in sorted(glob.glob(os.path.join(root, 'byOriginalSubset', sub, 'pr_*_SP_yellow_bipolar_*.txt'))):
        m = re.match(r'pr_(\d{12})_SP_(\w+?)_(\w+?)_(.+)\.txt$', os.path.basename(fn)); rec, probe, cfg, tissue = m.groups()
        k = 0
        for line in open(fn, encoding='utf-8', errors='replace'):
            if line.startswith('%') or not line.strip(): continue
            p = line.strip().split(';')
            rows.append(dict(recording=rec, tissue=tissue.lower(), animal=animal, probe=probe, mode='scan', order=k,
                             frequency_hz=float(p[1]) * 1e3, z_real=float(p[4]), z_imag=float(p[5]), device_state=p[0]))
            k += 1
df = pd.DataFrame(rows); df.to_csv(out, index=False)
print(len(df), 'readings from', df.groupby(['recording', 'tissue']).ngroups, 'recordings ->', out)
