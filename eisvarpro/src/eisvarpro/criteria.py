"""Band criteria of the article (Sect. 2.7, Appendix A).

Below the band (dispersion below f_min), from the R-CPE-L asymptote:
  closed form            W95~ = 2 z sigma (f_min/f_c)^alpha / (alpha ln10 D)            Eq. (8)
  band ratio             Gamma = (alpha ln10 D W* / (2 z sigma))^(1/alpha), limit f_min/Gamma   Eq. (9)
  descriptor bias        delta theta = b (f_min/f_c)^(-alpha)                            Eq. (10)
  tolerable error        sigma_max = alpha ln10 D W* (f_min/f_c)^(-alpha) / (2 z)       Eq. (11)
  operational rule       W_lin(f_c) (Eq. A6) and f_DL: W_lin(f_DL) = W*
Above the band (dispersion above f_max), from the low-frequency asymptote R0 - K (j w)^alpha + j w L:
  W95~' = 2 z sigma (f_c/f_max)^alpha / (alpha ln10 D'), Gamma' (Eq. 12); window f_min/Gamma < f_c < Gamma' f_max.
Also: the Fisher (Wald) width of log10 tau at a full Cole+L fit (the post-fit baseline of Table 7) and the
three-way screening rule (likely / uncertain / unlikely).

The closed form is an analytical scaling law and is anti-conservative as a numerical interval (coverage 58-72%);
decisions use W_lin / f_DL (Sect. 2.7). Domain: Cole (or ZARC) element, dispersion clearly outside the band
(r = (f_min/f_c)^-alpha <= 0.5), estimable asymptote, no strong electrode polarization.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq, minimize_scalar, nnls
from . import _core as C

LN10, Z95 = C.LN10, C.Z95


# ------------------------------------------------------------------------------------------- below the band
def band_factor(f, Rinf, Q, alpha, L, Zabs=None):
    """D = ||P h~|| and sin(theta) = D / ||h~|| (Eq. 8). Zabs: |Z_obs| used as weights (default: |asymptote|)."""
    f = np.asarray(f, float); w = 2 * np.pi * f
    Wt = np.abs(Rinf + 1 / (Q * (1j * w)**alpha) + 1j * w * L) if Zabs is None else np.asarray(Zabs, float)
    return C.band_factor(f, Rinf, Q, alpha, L, Wt)


def gamma(D, alpha, sigma, Wstar=1.0, z=Z95):
    """Band ratio Gamma of Eq. (9)."""
    return float((alpha * LN10 * D * Wstar / (2 * z * sigma))**(1 / alpha))


def width_closed(fmin, fc, D, alpha, sigma, z=Z95):
    """First-order 95% width of log10 tau, Eq. (8) (scaling law; anti-conservative)."""
    return float(2 * z * sigma / (alpha * LN10 * D) * (fmin / fc)**alpha)


def sigma_max(fmin, fc, D, alpha, Wstar=1.0, z=Z95):
    """Largest tolerable effective relative error, Eq. (11)."""
    return float(alpha * LN10 * D * Wstar / (2 * z) * (fmin / fc)**(-alpha))


def descriptor_bias(f, Rinf, Q, alpha, L, fc, Zabs=None):
    """Bias of the R-CPE-L descriptors (log10 Rinf, log10 Q, alpha, log10 L) when the data are a Cole+L spectrum
    with characteristic frequency fc below the band, Eq. (10): returns dict with b and delta = b (f_min/f_c)^-alpha."""
    f = np.asarray(f, float); w = 2 * np.pi * f
    Wt = np.abs(Rinf + 1 / (Q * (1j * w)**alpha) + 1j * w * L) if Zabs is None else np.asarray(Zabs, float)
    def rn(p): return C.stack((10**p[0] + 1 / (10**p[1] * (1j * w)**p[2]) + 1j * w * 10**p[3]) / Wt)
    p0 = np.array([np.log10(Rinf), np.log10(Q), alpha, np.log10(L)])
    J = C._jac(rn, p0)
    ht = C.stack(((1j * w)**(-alpha) / Q) / Wt * (1j * w / w.min())**(-alpha))
    b = -np.linalg.lstsq(J, ht, rcond=None)[0]
    r = (f.min() / fc)**(-alpha)
    names = ('log10Rinf', 'log10Q', 'alpha', 'log10L')
    return dict(b=dict(zip(names, map(float, b))), delta=dict(zip(names, map(float, b * r))), r=float(r))


def W_lin(f, Rinf, Q, alpha, L, fc, sigma, z=Z95):
    """Exact linearized width of log10 tau (Eq. A6) at a hypothesised fc on the R-CPE-L asymptote."""
    return C.W_lin(f, Rinf, Q, alpha, L, fc, sigma, z)


def f_DL(f, Rinf, Q, alpha, L, sigma, Wstar=1.0, z=Z95):
    """Operational detection limit: f_c at which W_lin = W*. Returns (value, flag); flag 'above' means that no
    dispersion up to f_min*10^0.5 is resolvable along this asymptote, 'below' that all down to f_min*1e-5 are."""
    return C.f_DL(f, Rinf, Q, alpha, L, sigma, z, Wstar)


def wald_width_full_fit(f, Zobs, p10, sigma, z=Z95):
    """Fisher (Wald) 95% width of log10 tau at a full Cole+L fit (post-fit baseline, Table 7).
    p10: parameters of M10 as returned by fit_models()['M10']['params']."""
    f = np.asarray(f, float); w = 2 * np.pi * f; Wt = C.weights(np.asarray(Zobs, complex))
    def r(q): return C.stack((10**q[0] + 10**q[1] / (1 + (1j * w * 10**q[2])**q[3]) + 1j * w * 10**q[4]) * Wt)
    q0 = np.array([p10['log10Rinf'], p10['log10dR'], p10['log10tau'], p10['alpha'], p10['log10L']])
    J = C._jac(r, q0)
    try:
        Cv = np.linalg.inv(J.T @ J); return float(2 * z * sigma * np.sqrt(max(Cv[2, 2], 0)))
    except np.linalg.LinAlgError:
        return float('inf')


def below_band(f, asym, sigma, Zabs=None, Wstar=1.0, z=Z95):
    """All below-band quantities for one band and one asymptote.
    asym: dict(Rinf, Q, alpha, L) in SI units (see fit_models()['M11']['asymptote'])."""
    Rinf, Q, al, L = asym['Rinf'], asym['Q'], asym['alpha'], asym['L']
    D, sinth = band_factor(f, Rinf, Q, al, L, Zabs)
    G = gamma(D, al, sigma, Wstar, z)
    dl, flag = f_DL(f, Rinf, Q, al, L, sigma, Wstar, z)
    return dict(D=D, sin_theta=sinth, Gamma=G, DL_closed=float(np.min(f) / G), f_DL=float(dl), f_DL_flag=flag)


def predict_below(fc, crit):
    """Closed-form and operational predictions for a dispersion at fc, and the three-way class."""
    pc = bool(fc > crit['DL_closed'])
    po = bool(crit['f_DL_flag'] in ('ok', 'below') and fc > crit['f_DL'])
    return dict(closed_form=pc, operational=po, three_way=three_way(pc, po))


def three_way(pred_closed, pred_operational):
    """Three-way rule (Sect. 3.7): likely if both predict identifiability, unlikely if neither, else uncertain."""
    if pred_closed and pred_operational: return 'likely'
    if not pred_closed and not pred_operational: return 'unlikely'
    return 'uncertain'


# ------------------------------------------------------------------------------------------- above the band
def fit_lowfreq_asymptote(f, Zobs):
    """Fit R0 - K (j w)^alpha + j w L (K >= 0) by variable projection (grid + bounded scalar search in alpha)."""
    f = np.asarray(f, float); w = 2 * np.pi * f; Z = np.asarray(Zobs, complex); W = C.weights(Z)
    def inner(al):
        A = np.column_stack([np.ones_like(w, dtype=complex), -(1j * w)**al, 1j * w]) * W[:, None]
        Ar = np.vstack([A.real, A.imag]); br = C.stack(Z * W)
        c, res = nnls(Ar, br); return res**2, c
    g = np.arange(0.30, 1.0001, 0.01); v = [inner(a)[0] for a in g]; k = int(np.argmin(v))
    o = minimize_scalar(lambda a: inner(a)[0], bounds=(g[max(k - 1, 0)], g[min(k + 1, g.size - 1)]), method='bounded')
    al = o.x if o.fun <= v[k] else g[k]; rss, c = inner(al)
    return dict(R0=float(c[0]), K=float(max(c[1], 1e-300)), alpha=float(al), L=float(max(c[2], 1e-12)), rss=float(rss))


def above_band(f, asym_lf, sigma, Zabs=None, Wstar=1.0, z=Z95):
    """Mirror-image closed form for a dispersion above the band, Eq. (12): D', sin(theta'), Gamma' and the upper
    detection limit Gamma' f_max. asym_lf: output of fit_lowfreq_asymptote."""
    f = np.asarray(f, float); w = 2 * np.pi * f
    R0, K, al, L = asym_lf['R0'], asym_lf['K'], asym_lf['alpha'], asym_lf['L']
    Wt = np.abs(R0 - K * (1j * w)**al + 1j * w * L) if Zabs is None else np.asarray(Zabs, float)
    def rn(p): return C.stack((10**p[0] - 10**p[1] * (1j * w)**p[2] + 1j * w * 10**p[3]) / Wt)
    J = C._jac(rn, np.array([np.log10(R0), np.log10(K), al, np.log10(L)]))
    P = np.eye(J.shape[0]) - J @ np.linalg.pinv(J)
    ht = C.stack(K * (1j * w)**al / Wt * (1j * w / w.max())**al)
    D = float(np.linalg.norm(P @ ht)); G = gamma(D, al, sigma, Wstar, z)
    return dict(D_prime=D, sin_theta_prime=float(D / np.linalg.norm(ht)), Gamma_prime=G, DL_upper=float(np.max(f) * G))
