"""Algorithm 2: preliminary frequency-band screening from a pilot spectrum, with or without a prior on f_c.

Steps (as in the article):
 1. Fit M10 and M11 to the pilot; sigma = [RSS/(2N-5)]^(1/2) from M10, times kappa_S if a correlated error model is
    known; if a larger error is expected in the candidate band, pass it as `sigma_candidate`.
 2. If the pilot profile already identifies tau: every band containing the pilot band is adequate; stop.
 3. Domain check (added after the measured tetrapolar test, Sect. 3.8): if the M11 exponent of the pilot lies at a
    bound, the asymptote is not estimable (pilot too close to the dispersion): every band is 'uncertain'; stop.
 4. For each candidate band: f_DL along the pilot asymptote on the frequencies of the band; with a prior
    f_c^exp +- delta decades: likely if f_c^exp 10^-delta > f_DL, unlikely if f_c^exp 10^delta < f_min/Gamma
    (Eq. 9), otherwise uncertain (refer); without a prior, report f_DL (the band can identify tau only for
    dispersions above it).
 5. Confirm the chosen band by measurement and a profile-likelihood check; use tetrapolar electrodes if electrode
    polarization may be appreciable at f_min.
This is a screen, not a validated design optimization.
"""
from __future__ import annotations
import numpy as np
from . import _core as C
from . import criteria as K
from .estimation import fit_models, WSTAR


def log_band(fmin, fmax, N):
    return np.logspace(np.log10(fmin), np.log10(fmax), int(N))


def screen_bands(f_pilot, Z_pilot, candidates, fc_prior=None, delta=0.0, kappa_S=1.0, sigma_candidate=None,
                 check_pilot_profile=True, Wstar=WSTAR):
    """candidates: list of frequency arrays, or of (f_min, f_max, N) tuples (log-spaced).
    Returns dict(pilot=..., status=..., bands=[...])."""
    fits = fit_models(f_pilot, Z_pilot); spec = fits['spectrum']
    sig = fits['sigma'] * kappa_S if sigma_candidate is None else float(sigma_candidate)
    pilot = dict(sigma=sig, alpha_M11=fits['M11']['params']['alpha'], alpha_at_bound=fits['M11']['alpha_at_bound'],
                 fc_M10=fits['M10']['fc'])
    bands = [np.asarray(c, float) if not (isinstance(c, tuple) and len(c) == 3) else log_band(*c) for c in candidates]
    if check_pilot_profile:
        g, lam = C.profile(spec, 'M10', 'log10tau', rss_min=fits['M10']['rss'])
        it = C.interval(g, lam, C.CHI95, Wstar); pilot['tau_identifiable'] = it['identifiable']
        if it['identifiable']:
            out = []
            for b in bands:
                contains = b.min() <= spec.f.min() and b.max() >= spec.f.max()
                out.append(dict(f_min=float(b.min()), f_max=float(b.max()), N=int(b.size),
                                classification='adequate (contains pilot)' if contains else 'not assessed'))
            return dict(pilot=pilot, status='pilot identifies tau', bands=out)
    if pilot['alpha_at_bound']:
        return dict(pilot=pilot, status='domain check: pilot asymptote not estimable (exponent at bound)',
                    bands=[dict(f_min=float(b.min()), f_max=float(b.max()), N=int(b.size), classification='uncertain')
                           for b in bands])
    a = fits['M11']['asymptote']; out = []
    for b in bands:
        cr = K.below_band(b, a, sig, None, Wstar)
        row = dict(f_min=float(b.min()), f_max=float(b.max()), N=int(b.size), f_DL=cr['f_DL'],
                   f_DL_flag=cr['f_DL_flag'], DL_closed=cr['DL_closed'], Gamma=cr['Gamma'], D=cr['D'])
        if fc_prior is not None:
            lo, hi = fc_prior * 10**(-delta), fc_prior * 10**delta
            if cr['f_DL_flag'] in ('ok', 'below') and lo > cr['f_DL']: row['classification'] = 'likely'
            elif hi < cr['DL_closed']: row['classification'] = 'unlikely'
            else: row['classification'] = 'uncertain'
        else:
            row['classification'] = 'report f_DL'
        out.append(row)
    return dict(pilot=pilot, status='screened', bands=out)
