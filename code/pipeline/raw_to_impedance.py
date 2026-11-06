"""Convert the raw SmartProbe frequency-scan files of Supplementary.zip (OriginalData/SmartProbe/Pig*/bipolarScan/
frequencyScan/*.txt; columns: device state, scan frequency (kHz), raw real, raw imaginary, time (ms)) to impedance,
following Data_Freq_Processing.m of the dataset's Tissue Analysis Scripts (AD5933 gain factor and system phase
from the 20 kOhm calibration). Optional outlier removal mimics MATLAB rmoutliers(T,'movmedian',50) applied to all
columns (row removed if any column deviates more than 3 scaled MADs from its moving median), as in
SmartProbe_EBIData_Preparation.m.
Usage: python raw_to_impedance.py <OriginalData/SmartProbe> readings.csv [--rmoutliers]"""
import sys, os, re, glob
import numpy as np, pandas as pd
GAIN = np.array([2.90738653334637e-07, 2.99823456365641e-07, 2.92931222769893e-07, 2.90231670035123e-07, 2.85507769080134e-07,
                 2.78597409670134e-07, 2.72258436903450e-07, 2.65419313499278e-07, 2.60462285122552e-07, 2.54863652675660e-07])
PHSYS = np.array([-1.49110776969620, -1.47849961102297, -1.42836072466419, -1.35525965047324, -1.31089463876321, -1.25018226569347,
                  -1.19111403087276, -1.1869839948261, -1.05340121871913, -0.979407835388207])

def convert(V):
    n = (V[:, 1] / 10).astype(int) - 1; re_, im_ = V[:, 2], V[:, 3]
    mag = 1 / (np.hypot(re_, im_) * GAIN[n])
    at = np.arctan(im_ / re_)
    ph = np.where(re_ > 0, np.where(im_ > 0, at, at + 2 * np.pi), at + np.pi) - PHSYS[n]
    ph = np.degrees(ph)
    ph = np.where(ph >= 360, ph - 360 * np.trunc(ph / 360), np.where(ph >= 270, ph - 360, ph))
    return mag * np.cos(np.radians(ph)), mag * np.sin(np.radians(ph))

def movmedian_outliers(x, k=50):
    """MATLAB movmedian/movmad window for even k: k/2 before and k/2-1 after; truncated at the ends."""
    N = len(x); lo = k // 2; hi = k // 2 - 1; out = np.zeros(N, bool)
    for i in range(N):
        w = x[max(0, i - lo): min(N, i + hi + 1)]; med = np.median(w); mad = 1.482602218505602 * np.median(np.abs(w - med))
        out[i] = abs(x[i] - med) > 3 * mad
    return out

def read_raw(fn):
    rows = [l.split() for l in open(fn, encoding='utf-8', errors='replace') if l.strip() and not l.startswith('%')]
    return np.array([[float(v) for v in r[:5]] for r in rows if len(r) >= 5])

if __name__ == '__main__':
    root, out = sys.argv[1], sys.argv[2]; rmo = '--rmoutliers' in sys.argv
    rows = []
    for fn in sorted(glob.glob(os.path.join(root, 'Pig*', 'bipolarScan', 'frequencyScan', '**', '*_SP_yellow_bipolar_*.txt'), recursive=True)):
        m = re.match(r'(\d{12})_SP_(\w+?)_(\w+?)_(.+)\.txt$', os.path.basename(fn)); rec, probe, cfg, tissue = m.groups()
        V = read_raw(fn)
        if rmo:
            bad = np.zeros(len(V), bool)
            for c in range(V.shape[1]): bad |= movmedian_outliers(V[:, c])
            V = V[~bad]
        zr, zi = convert(V); animal = 1 if '/Pig1/' in fn else 2
        for k in range(len(V)):
            rows.append(dict(recording=rec, tissue=tissue.lower(), animal=animal, probe=probe, mode='scan', order=k,
                             frequency_hz=V[k, 1] * 1e3, z_real=zr[k], z_imag=zi[k]))
    pd.DataFrame(rows).to_csv(out, index=False); print(len(rows), 'readings ->', out)
