#!/bin/bash
# Extended battery test: all states of charge (I-IX) at 25 C except V, and all states at 35 and 45 C; first and last cycle.
# Usage: bash run_battery_ext.sh "<EIS data folder>"   (folder 'EIS data' from Zenodo 10.5281/zenodo.3633835)
D="$1"
for st in I II III IV VI VII VIII IX; do python battery_measured_ext.py "$D" ext25_$st.json 2 "EIS_state_${st}_25C*.txt"; done
for t in 35C 45C; do python battery_measured_ext.py "$D" ext_$t.json 2 "EIS_state_*_${t}*.txt"; done
python - <<'PY'
import json, glob, pandas as pd
rows = []
for f in sorted(glob.glob('ext25_*.json') + glob.glob('ext_*.json')): rows += json.load(open(f))
R = pd.DataFrame(rows); R['state'] = R.file.str.extract(r'state_([IVX]+)_')[0]; R['temp'] = R.file.str.extract(r'_(\d+C)')[0]
R.to_json('battery_ext_all.json', orient='records')
PY
python analyse_battery_ext.py battery_ext_all.json
