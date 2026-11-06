"""Examples of the eisvarpro API (run from this folder: python run_examples.py)."""
import numpy as np, eisvarpro as ev

def read(name):
    a = np.genfromtxt(name, delimiter=',', names=True); return a['f_Hz'], a['re_ohm'] + 1j * a['im_ohm']

# 1. Algorithm 1 + decision rule on the band of the article (10-100 kHz), dispersion at 1 kHz below the band
f, Z = read('synthetic_narrow_band.csv'); r = ev.analyze(f, Z)          # nominal threshold; B=200 to calibrate
print('narrow band:', r['report']['status'], r['report']['report'], 'f_c_hat = %.0f Hz' % r['fits']['M10']['fc'],
      'f_DL = %.0f Hz' % r['criteria']['f_DL'], 'f_min/Gamma = %.0f Hz' % r['criteria']['DL_closed'])

# 2. Same tissue, band extended to 0.3 kHz: the Cole parameters become reportable
f, Z = read('synthetic_wide_band.csv'); r = ev.analyze(f, Z)
print('wide band:  ', r['report']['status'], r['report']['report'])

# 3. Algorithm 2: screen candidate bands from a 10-100 kHz pilot with a prior f_c = 1 kHz
rng = np.random.default_rng(1); fp = np.linspace(10e3, 100e3, 10)
Zp = 50 + 400 / (1 + (1j * fp / 1e3)**0.7) + 1j * 2 * np.pi * fp * 1e-6
rng.standard_normal(20); Zp = Zp + np.abs(Zp) * 0.025 * (rng.standard_normal(10) + 1j * rng.standard_normal(10))
out = ev.screen_bands(fp, Zp, [(10e3, 100e3, 20), (3e3, 100e3, 20), (300, 100e3, 20)], fc_prior=1e3)
for b in out['bands']:
    dl = ('> %.0f Hz (none resolvable)' if b['f_DL_flag'] == 'above' else '%.0f Hz') % b['f_DL']
    print('band %.0f Hz-%.0f kHz: f_DL %s -> %s' % (b['f_min'], b['f_max'] / 1e3, dl, b['classification']))

# 4. Domain check on a measured tetrapolar pilot (top decade of a public SFB7 spectrum, CC BY 4.0)
f, Z = read('tetrapolar_SFB7_ID0001.csv'); top = f >= f.max() / 10 * (1 - 1e-9)
print('tetrapolar pilot:', ev.screen_bands(f[top], Z[top], [(3e3, 1e6, 27)], fc_prior=44e3)['status'])

# 5. Sweeps: per-sweep median and drift budget
print('drift budget at sigma 1.8%%, drift 1.1%%/s: %.1f s' % ev.sweeps.drift_budget(0.018, 0.011))
