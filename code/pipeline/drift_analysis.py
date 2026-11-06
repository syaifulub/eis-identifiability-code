"""Between-sweep drift and its effect on the Kramers-Kronig test (design lesson, Supplementary Note S4.9).
Uses the raw SmartProbe frequency-scan files (Supplementary.zip, OriginalData/SmartProbe), converted with the gain and
system phase of Data_Freq_Processing.m after moving-median outlier removal (raw_to_impedance.py), because only these
files carry the acquisition time (ms). For each of the 20 recordings: complete sweeps, sweep duration, KK residual of
each sweep, of the pooled per-frequency median and of the median of the per-sweep spectra, and the between-sweep
drift of |Z| and phase per second. The time column of the raw files is the duration of each reading (ms; longer at
low frequency); elapsed time is its cumulative sum, which excludes any pause between readings.
Usage: python drift_analysis.py <OriginalData/SmartProbe> drift.csv"""
import sys, os, re, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eis_pipeline as ep, raw_to_impedance as rz
root, out = sys.argv[1], sys.argv[2]
F = np.array([10e3 * k for k in range(1, 11)])
man = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectra_manifest.csv'), dtype={'recording': str})
rows = []
for r in man.itertuples():
    fn = glob.glob(os.path.join(root, 'Pig*', 'bipolarScan', 'frequencyScan', '**', '%s_SP_yellow_bipolar_%s.txt' % (r.recording, r.tissue)), recursive=True)[0]
    V = rz.read_raw(fn); bad = np.zeros(len(V), bool)
    for c in range(V.shape[1]): bad |= rz.movmedian_outliers(V[:, c])
    V = V[~bad]; zr, zi = rz.convert(V); f = V[:, 1] * 1e3; t = np.cumsum(V[:, 4]) / 1e3; Z = zr + 1j * zi  # column 5 = duration of each reading (ms)
    runs = []; s = 0
    for i in range(1, len(f) + 1):
        if i == len(f) or f[i] < f[i - 1]: runs.append(np.arange(s, i)); s = i
    sw = [ix for ix in runs if set(F) <= set(f[ix])]
    spec = []; tmid = []; dur = []
    for ix in sw:
        spec.append(np.array([np.median(Z[ix][f[ix] == q].real) + 1j * np.median(Z[ix][f[ix] == q].imag) for q in F]))
        tmid.append(np.median(t[ix])); dur.append(t[ix].max() - t[ix].min())
    pooled = np.array([np.median(Z[f == q].real) + 1j * np.median(Z[f == q].imag) for q in F])
    kk_sw = [ep.kk_test(F, S)['rms'] for S in spec]
    rec = dict(spectrum=r.spectrum, tissue=r.tissue, n_sweeps=len(sw), sweep_duration_s=float(np.median(dur)) if dur else np.nan,
               kk_pooled=ep.kk_test(F, pooled)['rms'], kk_sweep_max=float(max(kk_sw)) if kk_sw else np.nan,
               kk_sweep_median=ep.kk_test(F, np.median(np.real(spec), 0) + 1j * np.median(np.imag(spec), 0))['rms'] if spec else np.nan)
    if len(spec) >= 2:
        S = np.array(spec); dt = np.diff(tmid)
        dmag = np.abs(np.diff(np.log(np.abs(S)), axis=0)).max(1) / dt          # largest relative |Z| change per second
        dph = np.abs(np.diff(np.degrees(np.angle(S)), axis=0)).max(1) / dt     # largest phase change per second
        rng_mag = (np.log(np.abs(S)).max(0) - np.log(np.abs(S)).min(0)).max()  # largest spread of |Z| across sweeps
        rec.update(drift_mag_pct_per_s=float(100 * np.median(dmag)), drift_phase_deg_per_s=float(np.median(dph)), spread_mag_pct=float(100 * rng_mag),
                   interval_s=float(np.median(dt)))
    rows.append(rec)
D = pd.DataFrame(rows); D.to_csv(out, index=False); print(D.round(3).to_string())
