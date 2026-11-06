import os, sys, importlib.util
import numpy as np
import pytest
import eisvarpro as ev
from eisvarpro import criteria as K, sweeps as S, _core as C

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, '..', 'examples')


def cole(f, Ri, dR, fc, al, L):
    w = 2 * np.pi * f
    return Ri + dR / (1 + (1j * w / (2 * np.pi * fc))**al) + 1j * w * L


def read(name):
    a = np.genfromtxt(os.path.join(EX, name), delimiter=',', names=True)
    return a['f_Hz'], a['re_ohm'] + 1j * a['im_ohm']


def test_varpro_recovers_noise_free_cole():
    f = np.logspace(2.5, 5, 20); Z = cole(f, 50, 400, 1e3, 0.7, 1e-6)
    r = ev.fit_models(f, Z)
    p = r['M10']['params']
    assert abs(10**p['log10Rinf'] - 50) / 50 < 1e-3
    assert abs(10**p['log10dR'] - 400) / 400 < 1e-3
    assert abs(r['M10']['fc'] - 1e3) / 1e3 < 1e-3
    assert abs(p['alpha'] - 0.7) < 1e-4


def test_kk_consistent_for_cole():
    f = np.logspace(4, 5, 10); Z = cole(f, 50, 400, 1e3, 0.7, 1e-6)
    assert C.kk_test(f, Z)['rms'] < 0.5


def test_closed_form_identities():
    D, al, sig, fmin = 0.12, 0.69, 0.018, 10e3
    G = K.gamma(D, al, sig)
    assert np.isclose(K.width_closed(fmin, fmin / G, D, al, sig), 1.0)          # Eq. (8) at the limit of Eq. (9)
    fc = 2e3
    assert np.isclose(K.width_closed(fmin, fc, D, al, K.sigma_max(fmin, fc, D, al)), 1.0)   # Eq. (11)
    assert np.isclose(K.gamma(D, al, sig / 2) / G, 2**(1 / al))                 # halving sigma: factor 2^(1/alpha)


def test_f_DL_solves_W_lin_equal_Wstar():
    f = np.linspace(10e3, 100e3, 10); a = ev.fit_models(f, cole(f, 50, 400, 1e3, 0.7, 1e-6))['M11']['asymptote']
    dl, flag = K.f_DL(f, a['Rinf'], a['Q'], a['alpha'], a['L'], 0.005)
    assert flag == 'ok'
    assert abs(K.W_lin(f, a['Rinf'], a['Q'], a['alpha'], a['L'], dl, 0.005) - 1.0) < 0.01
    # with a larger effective error no dispersion up to f_min 10^0.5 is resolvable on this band
    assert K.f_DL(f, a['Rinf'], a['Q'], a['alpha'], a['L'], 0.018)[1] == 'above'


def test_three_way():
    assert K.three_way(True, True) == 'likely'
    assert K.three_way(False, False) == 'unlikely'
    assert K.three_way(True, False) == 'uncertain'


def test_decision_rule_examples():
    f, Z = read('synthetic_narrow_band.csv'); r = ev.analyze(f, Z)
    assert r['report']['status'] == 'R-CPE-L' and not r['report']['fc_in_band']
    assert 'M11.log10Q' in r['report']['report'] and 'M11.alpha' in r['report']['report']
    f, Z = read('synthetic_wide_band.csv'); r = ev.analyze(f, Z)
    assert r['report']['status'] == 'Cole' and 'M10.log10tau' in r['report']['report']
    assert r['wald_width_full_fit'] < 1.0


def test_kk_flag():
    f = np.linspace(10e3, 100e3, 10); Z = cole(f, 50, 400, 1e3, 0.7, 1e-6)
    rep = ev.decision_rule(5.0, f, 1e3, {'M10': {}, 'M11': {}})
    assert rep['status'] == 'KK-inconsistent'


def test_algorithm2_screen_and_domain_check():
    # tissue-like pilot on 10-100 kHz (N=10, 2.5% noise as in the article), dispersion at 1 kHz
    rng = np.random.default_rng(1)
    fp = np.linspace(10e3, 100e3, 10); Zp = cole(fp, 50, 400, 1e3, 0.7, 1e-6)
    rng.standard_normal(20)   # same draws as the documented example
    Zp = Zp + np.abs(Zp) * 0.025 * (rng.standard_normal(10) + 1j * rng.standard_normal(10))
    out = ev.screen_bands(fp, Zp, [(10e3, 100e3, 20), (3e3, 100e3, 20), (300, 100e3, 20)], fc_prior=1e3,
                          check_pilot_profile=False)
    assert out['status'] == 'screened'
    assert [b['classification'] for b in out['bands']] == ['unlikely', 'uncertain', 'likely']
    fdl = [b['f_DL'] for b in out['bands']]
    assert fdl[2] < fdl[1] < fdl[0]
    # measured tetrapolar pilot (top decade, dispersion just below it): exponent at bound -> all uncertain
    f, Z = read('tetrapolar_SFB7_ID0001.csv'); top = f >= f.max() / 10 * (1 - 1e-9)
    out = ev.screen_bands(f[top], Z[top], [(3e3, 1e6, 27), (10e3, 1e6, 20)], fc_prior=44e3)
    assert out['status'].startswith('domain check')
    assert all(b['classification'] == 'uncertain' for b in out['bands'])


def test_sweeps_and_drift_budget():
    f = np.linspace(10e3, 100e3, 10); Z0 = cole(f, 50, 400, 1e3, 0.7, 1e-6)
    sw = np.array([Z0 * (1 + 0.011 * k * 9.5) for k in range(3)])      # 1.1 %/s drift, 9.5 s per sweep
    assert np.allclose(S.sweep_median(sw), sw[1])
    rate = S.drift_rate(sw, [0, 9.5, 19.0])
    assert 0.009 < rate < 0.012
    assert abs(S.drift_budget(0.018, 0.011) - 1.636) < 0.01


def test_above_band_window():
    f = np.logspace(3, 4.5, 15); Z = cole(f, 50, 400, 100e3, 0.7, 1e-7)
    a = K.fit_lowfreq_asymptote(f, Z)
    out = K.above_band(f, a, 0.01)
    assert out['Gamma_prime'] > 1 and out['DL_upper'] > f.max()


def test_core_identical_to_pipeline():
    path = os.path.join(HERE, '..', '..', 'code', 'pipeline', 'eis_pipeline.py')
    if not os.path.exists(path):
        pytest.skip('pipeline of the repository not present')
    spec_ = importlib.util.spec_from_file_location('eis_pipeline_ref', path)
    ep = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(ep)
    f, Z = read('synthetic_narrow_band.csv')
    for m in ('M10', 'M11'):
        a = C.fit(C.Spectrum(f, Z), m); b = ep.fit(ep.Spectrum(f, Z), m)
        assert a['rss'] == b['rss'] and np.array_equal(a['xi'], b['xi'])
    r = C.fit(C.Spectrum(f, Z), 'M11'); p = C.params('M11', r)
    args = (f, 10**p['log10Rinf'], 10**p['log10Q'], p['alpha'], 10**p['log10L'], 0.01)
    assert C.f_DL(*args) == ep.f_DL(*args)
