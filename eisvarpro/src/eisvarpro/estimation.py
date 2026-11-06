"""Algorithm 1 (identifiability-aware estimation) and the reporting decision rule (Fig. 5)."""
from __future__ import annotations
import numpy as np
from . import _core as C
from . import criteria as K

WSTAR = 1.0          # identifiability threshold for log10-scale parameters (decades)
WSTAR_ALPHA = 0.2    # identifiability threshold for alpha
KK_LIMIT = 3.0       # % RMS, linear Kramers-Kronig test
COLE_PARAMS = ('log10Rinf', 'log10dR', 'log10tau', 'alpha')
DESCRIPTORS = ('log10Rinf', 'log10Q', 'alpha')


def as_spectrum(f, Z):
    f = np.asarray(f, float); Z = np.asarray(Z, complex); o = np.argsort(f)
    return C.Spectrum(f[o], Z[o])


def effective_sigma(rss, n, p=5):
    """Effective relative error sigma = [RSS/(2N - p)]^(1/2) from the Cole+L fit (p = 5)."""
    return float(np.sqrt(rss / (n - p)))


def fit_models(f, Z, n_starts=5):
    """VarPro fits of M10 (Cole+L) and M11 (R-CPE-L). Returns parameters, RSS, NRMSE, effective sigma and the
    R-CPE-L asymptote in SI units."""
    spec = as_spectrum(f, Z); out = {}
    for m in ('M10', 'M11'):
        r = C.fit(spec, m, n_starts=n_starts); p = C.params(m, r)
        out[m] = dict(params=p, rss=r['rss'], nrmse=float(np.sqrt(r['rss'] / spec.N)), _res=r)
    p10 = out['M10']['params']; out['M10']['fc'] = float(1 / (2 * np.pi * 10**p10['log10tau']))
    out['sigma'] = effective_sigma(out['M10']['rss'], spec.n)
    p11 = out['M11']['params']
    out['M11']['asymptote'] = dict(Rinf=10**p11['log10Rinf'], Q=10**p11['log10Q'], alpha=p11['alpha'], L=10**p11['log10L'])
    out['M11']['alpha_at_bound'] = bool(p11['alpha'] <= C.BOUNDS['M11']['alpha'][0] + 1e-3 or
                                         p11['alpha'] >= C.BOUNDS['M11']['alpha'][1] - 1e-3)
    out['spectrum'] = spec
    return out


def profiles(spec, fits, names=None, psi=None, B=0, rng=None, crit=None):
    """Profile likelihood intervals. psi: error-model parameters (Eq. 5); with B > 0 the thresholds are
    calibrated by parametric bootstrap (B = 200 for screening, >= 1,000 for a reported interval), otherwise the
    nominal chi2 threshold 3.84 is used. Returns {model: {name: interval dict}}."""
    names = names or {'M10': COLE_PARAMS, 'M11': DESCRIPTORS}
    out = {}
    for m, nm in names.items():
        res = fits[m]['_res']; thr = {n: C.CHI95 for n in nm}
        if B and psi is not None:
            thr = C.calibrated_threshold(spec, m, list(nm), res, psi, B=B, rng=rng or np.random.default_rng(0))
        out[m] = {}
        for n in nm:
            g, lam = C.profile(spec, m, n, rss_min=res['rss'])
            wmax = WSTAR_ALPHA if n == 'alpha' else WSTAR
            it = C.interval(g, lam, thr[n], wmax); it['threshold'] = float(thr[n]); out[m][n] = it
    return out


def analyze(f, Z, psi=None, B=0, rng=None, reproducibility=None):
    """Algorithm 1 followed by the decision rule of Fig. 5 for one spectrum.

    Returns a dict with: kk (RMS %), fits, sigma, intervals, criteria (D, sin theta, Gamma, f_min/Gamma, f_DL,
    descriptor bias), the three-way class for the fitted f_c, and `report`: which parameters may be reported and
    why. `reproducibility` (optional dict name -> half-width) is attached to the report."""
    f = np.asarray(f, float); Z = np.asarray(Z, complex)
    kk = C.kk_test(np.sort(f), Z[np.argsort(f)])
    fits = fit_models(f, Z); spec = fits['spectrum']; sig = fits['sigma']
    iv = profiles(spec, fits, psi=psi, B=B, rng=rng)
    asym = fits['M11']['asymptote']
    crit = K.below_band(spec.f, asym, sig, np.abs(spec.Z))
    fc = fits['M10']['fc']; pred = K.predict_below(fc, crit)
    bias = None   # Eq. (10) applies only to a dispersion below the band, r = (f_min/f_c)^-alpha <= 0.5
    if fc < spec.f.min() and (spec.f.min() / fc)**(-asym['alpha']) <= 0.5:
        try:
            bias = K.descriptor_bias(spec.f, asym['Rinf'], asym['Q'], asym['alpha'], asym['L'], fc, np.abs(spec.Z))
        except Exception:
            bias = None
    rep = decision_rule(kk['rms'], spec.f, fc, iv)
    if reproducibility: rep['reproducibility'] = reproducibility
    rep['detection_limit_f_DL'] = crit['f_DL']
    rep['descriptor_bias'] = bias['delta'] if bias else 'not applicable (dispersion not below the band, r > 0.5)'
    return dict(kk=kk, fits={m: {k: v for k, v in fits[m].items() if k != '_res'} for m in ('M10', 'M11')},
                sigma=sig, intervals=iv, criteria=crit, prediction_at_fitted_fc=pred,
                wald_width_full_fit=K.wald_width_full_fit(spec.f, spec.Z, fits['M10']['params'], sig), report=rep)


def decision_rule(kk_rms, f, fc_hat, intervals):
    """Fig. 5. KK-inconsistent: flag, no physical interpretation. Otherwise report the Cole parameters that meet
    the identifiability criterion if f_min <= f_c_hat <= f_max; else report the R-CPE-L descriptors that meet it
    (band-specific) and do not interpret tau, dR, R0."""
    f = np.asarray(f, float)
    if kk_rms >= KK_LIMIT:
        return dict(status='KK-inconsistent', report=[], note='Flag: do not interpret parameters physically; '
                    're-measure, or report as KK-inconsistent with reproducibility limits.')
    in_band = bool(f.min() <= fc_hat <= f.max())
    cole_ok = [n for n, it in intervals['M10'].items() if it['identifiable']]
    if in_band and cole_ok:
        return dict(status='Cole', fc_in_band=True, report=['M10.' + n for n in cole_ok],
                    left_out=['M10.' + n for n in intervals['M10'] if n not in cole_ok],
                    note='Report these Cole parameters with conditionally calibrated intervals.')
    desc_ok = [n for n, it in intervals['M11'].items() if it['identifiable']]
    return dict(status='R-CPE-L', fc_in_band=in_band, report=['M11.' + n for n in desc_ok],
                note='Band-specific descriptors; state band, correction model and probe; do not interpret tau, dR, R0.'
                     + ('' if in_band else ' Fitted f_c lies outside the band.'))
