"""Measured-data pipeline for the manuscript
"Identifiability-Aware Parameter Estimation and Frequency-Band Assessment of Equivalent-Circuit Models".

Implements, from a tidy table of readings, the steps of Sects. 2.1-2.7 and Algorithm 1:
  quality control (median spectra, complete sweeps), linear Kramers-Kronig test, variable-projection (VarPro)
  fits of M10 (Cole+L) and M11 (R-CPE-L), structured error model (Eq. 5), profile likelihood with nominal and
  bootstrap-calibrated thresholds, closed-form criterion (Eqs. 8-9), operational detection limit f_DL (Eq. A6)
  and band truncation (Table 6).

This is a self-contained re-implementation of the Methods as written in the manuscript. Where the text leaves a
detail open, the choice made here is stated in the docstring of the function concerned (search "CHOICE").
Numbers may differ slightly from the published tables; compare_to_published.py reports the differences.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import nnls, minimize_scalar, least_squares, minimize, brentq

LN10 = np.log(10.0)
Z95 = 1.959964
CHI95 = 3.841459

# ----------------------------------------------------------------------------------------------- utilities
def stack(c):
    return np.r_[np.real(c), np.imag(c)]

def weights(Zobs):
    """Residual weights 1/max(|Z|, 1 Ohm) (Table S42)."""
    return 1.0 / np.maximum(np.abs(Zobs), 1.0)

# Bounds (log10 of value in SI units; Table S42)
BOUNDS = {
    'M10': dict(log10Rinf=(-3, 7), log10dR=(-3, 8), log10tau=(-7, 1), alpha=(0.3, 1.0), log10L=(-9, 0)),
    'M11': dict(log10Rinf=(-3, 7), log10Q=(-12, 0), alpha=(0.3, 1.0), log10L=(-9, 0)),
}

# ----------------------------------------------------------------------------------------------- models
def basis(model, w, xi):
    """Columns of the separable model. M10: xi=(log10 tau, alpha), c=(Rinf, dR, L).
    M11: xi=(alpha,), c=(Rinf, 1/Q, L)."""
    jw = 1j * w
    if model == 'M10':
        lt, al = xi
        return np.column_stack([np.ones_like(w, dtype=complex), 1.0 / (1.0 + (jw * 10.0**lt)**al), jw])
    if model == 'M11':
        (al,) = xi
        return np.column_stack([np.ones_like(w, dtype=complex), jw**(-al), jw])
    raise ValueError(model)

def lin_bounds(model):
    b = BOUNDS[model]
    if model == 'M10':
        lo = [b['log10Rinf'][0], b['log10dR'][0], b['log10L'][0]]; hi = [b['log10Rinf'][1], b['log10dR'][1], b['log10L'][1]]
    else:  # coefficient is 1/Q, so its log-bounds are -bounds of Q
        lo = [b['log10Rinf'][0], -b['log10Q'][1], b['log10L'][0]]; hi = [b['log10Rinf'][1], -b['log10Q'][0], b['log10L'][1]]
    return 10.0**np.array(lo, float), 10.0**np.array(hi, float)

def nl_bounds(model):
    b = BOUNDS[model]
    return (np.array([b['log10tau'][0], b['alpha'][0]]), np.array([b['log10tau'][1], b['alpha'][1]])) if model == 'M10' \
        else (np.array([b['alpha'][0]]), np.array([b['alpha'][1]]))

def bounded_lsq(A, b, lo, hi, fixed=None):
    """min ||A c - b|| s.t. lo <= c <= hi, with optional fixed coefficients {index: value}.
    CHOICE: shifted NNLS (c = lo + c', c' >= 0); coefficients exceeding the upper bound are fixed at the bound and
    the problem is re-solved (active-set loop), which reproduces the KKT solution for these well-conditioned bounds."""
    n = A.shape[1]; fixed = dict(fixed or {})
    for _ in range(n + 1):
        free = [k for k in range(n) if k not in fixed]
        rhs = b - (A[:, list(fixed)] @ np.array(list(fixed.values())) if fixed else 0.0)
        c = np.zeros(n)
        for k, v in fixed.items(): c[k] = v
        if free:
            Af = A[:, free]; cp, _ = nnls(Af, rhs - Af @ lo[free], maxiter=50 * n)
            c[free] = lo[free] + cp
        over = [k for k in free if c[k] > hi[k]]
        if not over: break
        for k in over: fixed[k] = hi[k]
    r = A @ c - b
    return c, float(r @ r)

class Spectrum:
    """Weighted design for one spectrum: f (Hz), Zobs (complex, Ohm)."""
    def __init__(self, f, Zobs, G=None):
        self.f = np.asarray(f, float); self.w = 2 * np.pi * self.f; self.Z = np.asarray(Zobs, complex)
        self.W = weights(self.Z); self.b = stack(self.Z * self.W); self.N = self.f.size; self.n = 2 * self.N
        self.G = G  # optional whitening matrix L^{-1} (GLS)
    def A(self, model, xi):
        Ac = basis(model, self.w, xi) * self.W[:, None]
        A = np.vstack([Ac.real, Ac.imag])
        return A if self.G is None else self.G @ A
    def rhs(self):
        return self.b if self.G is None else self.G @ self.b
    def inner(self, model, xi, fixed=None):
        lo, hi = lin_bounds(model)
        return bounded_lsq(self.A(model, xi), self.rhs(), lo, hi, fixed)

# ----------------------------------------------------------------------------------------------- fitting
def fit(spec: Spectrum, model: str, fixed_lin=None, fixed_nl=None, n_starts=5):
    """Global VarPro fit. M10: grid over (log10 tau, alpha) then TRF (scipy least_squares) from the best n_starts
    grid points; M11: dense grid over alpha then bounded scalar refinement. fixed_lin {index: value} fixes linear
    coefficients (profiles of linear parameters); fixed_nl {'log10tau' or 'alpha': value} fixes a nonlinear one."""
    fixed_nl = dict(fixed_nl or {})
    if model == 'M11':
        if 'alpha' in fixed_nl:
            c, rss = spec.inner(model, (fixed_nl['alpha'],), fixed_lin); return dict(xi=np.array([fixed_nl['alpha']]), c=c, rss=rss)
        g = np.arange(0.30, 1.0001, 0.01); v = [spec.inner(model, (a,), fixed_lin)[1] for a in g]; k = int(np.argmin(v))
        o = minimize_scalar(lambda a: spec.inner(model, (a,), fixed_lin)[1], bounds=(g[max(k - 1, 0)], g[min(k + 1, g.size - 1)]),
                            method='bounded', options={'xatol': 1e-7})
        a = o.x if o.fun <= v[k] else g[k]; c, rss = spec.inner(model, (a,), fixed_lin)
        return dict(xi=np.array([a]), c=c, rss=rss)
    # M10
    lo, hi = nl_bounds('M10')
    if 'log10tau' in fixed_nl or 'alpha' in fixed_nl:
        which = 0 if 'log10tau' in fixed_nl else 1; val = fixed_nl['log10tau' if which == 0 else 'alpha']
        other_lo, other_hi = lo[1 - which], hi[1 - which]
        def f1(x):
            xi = (val, x) if which == 0 else (x, val); return spec.inner('M10', xi, fixed_lin)[1]
        g = np.linspace(other_lo, other_hi, 15 if which == 0 else 33); v = [f1(x) for x in g]; k = int(np.argmin(v))
        o = minimize_scalar(f1, bounds=(g[max(k - 1, 0)], g[min(k + 1, g.size - 1)]), method='bounded', options={'xatol': 1e-7})
        x = o.x if o.fun <= v[k] else g[k]; xi = (val, x) if which == 0 else (x, val); c, rss = spec.inner('M10', xi, fixed_lin)
        return dict(xi=np.array(xi), c=c, rss=rss)
    G1 = np.arange(lo[0], hi[0] + 1e-9, 0.25); G2 = np.arange(0.35, 1.0001, 0.1)
    cand = sorted(((spec.inner('M10', (a, b), fixed_lin)[1], a, b) for a in G1 for b in G2))[:n_starts]
    best = None
    def resid(xi):
        c, _ = spec.inner('M10', xi, fixed_lin); return spec.A('M10', xi) @ c - spec.rhs()
    for _, a, b in cand:
        try:
            o = least_squares(resid, x0=[a, b], bounds=(lo, hi), method='trf', diff_step=1e-6, xtol=1e-10, ftol=1e-12)
            xi = o.x
        except Exception:
            xi = np.array([a, b])
        c, rss = spec.inner('M10', xi, fixed_lin)
        if best is None or rss < best['rss']: best = dict(xi=np.array(xi), c=c, rss=rss)
    return best

def params(model, res):
    c, xi = res['c'], res['xi']
    if model == 'M10':
        return dict(log10Rinf=np.log10(max(c[0], 1e-300)), log10dR=np.log10(max(c[1], 1e-300)), log10tau=xi[0], alpha=xi[1],
                    log10L=np.log10(max(c[2], 1e-300)))
    return dict(log10Rinf=np.log10(max(c[0], 1e-300)), log10Q=-np.log10(max(c[1], 1e-300)), alpha=xi[0], log10L=np.log10(max(c[2], 1e-300)))

def model_Z(model, f, p):
    w = 2 * np.pi * np.asarray(f); jw = 1j * w
    if model == 'M10':
        return 10**p['log10Rinf'] + 10**p['log10dR'] / (1 + (jw * 10**p['log10tau'])**p['alpha']) + jw * 10**p['log10L']
    return 10**p['log10Rinf'] + 1 / (10**p['log10Q'] * jw**p['alpha']) + jw * 10**p['log10L']

def info_criteria(rss, n, p):
    """CHOICE: Gaussian AICc/BIC with k = p + 1 (variance counted)."""
    k = p + 1; ll = n * np.log(rss / n)
    return dict(AICc=ll + 2 * k + 2 * k * (k + 1) / (n - k - 1), BIC=ll + k * np.log(n))

NPAR = {'M10': 5, 'M11': 4}
LINIDX = {'M10': {'log10Rinf': 0, 'log10dR': 1, 'log10L': 2}, 'M11': {'log10Rinf': 0, 'log10Q': 1, 'log10L': 2}}

def fit_fixed(spec, model, name, value):
    """Best fit with parameter `name` fixed at `value` (others re-optimized globally through VarPro)."""
    if name in LINIDX[model]:
        k = LINIDX[model][name]; v = 10.0**(-value if name == 'log10Q' else value)
        return fit(spec, model, fixed_lin={k: v})
    return fit(spec, model, fixed_nl={name: value})

# ----------------------------------------------------------------------------------------------- profiles
def profile(spec, model, name, rss_min=None, step=None, refine=True):
    """Profile Lambda(theta) = n log(RSS(theta)/RSS_min) on a grid spanning the bounds (step 0.05 decade, 0.005 for
    alpha), refined with 60 extra points over the region where Lambda <= 30 so that narrow intervals (e.g. of
    log10 R_inf) are resolved. CHOICE: RSS_min is the smaller of the global fit and the profile minimum."""
    lo, hi = BOUNDS[model][name]
    step = step or (0.005 if name == 'alpha' else 0.05)
    g = np.arange(lo, hi + 1e-9, step)
    rss = np.array([fit_fixed(spec, model, name, x)['rss'] for x in g])
    if rss_min is None: rss_min = fit(spec, model)['rss']
    if refine:
        lam = spec.n * np.log(rss / min(rss_min, rss.min())); k = int(np.argmin(lam)); i = k; j = k
        while i > 0 and lam[i - 1] <= 30: i -= 1
        while j < g.size - 1 and lam[j + 1] <= 30: j += 1
        a, b = g[max(i - 1, 0)], g[min(j + 1, g.size - 1)]
        if (b - a) < 40 * step:
            gf = np.linspace(a, b, 61)[1:-1]
            rf = np.array([fit_fixed(spec, model, name, x)['rss'] for x in gf])
            g = np.r_[g, gf]; rss = np.r_[rss, rf]; o = np.argsort(g); g, rss = g[o], rss[o]
    rss_min = min(rss_min, rss.min())
    return g, spec.n * np.log(rss / rss_min)

def interval(g, lam, crit, width_max):
    """Connected region {Lambda <= crit} around the minimum, linearly interpolated at the crossings.
    Closed if it does not reach either end of the grid (bounds); identifiable if closed and narrower than width_max."""
    k = int(np.argmin(lam)); i = k
    while i > 0 and lam[i - 1] <= crit: i -= 1
    j = k
    while j < g.size - 1 and lam[j + 1] <= crit: j += 1
    closed_lo = i > 0; closed_hi = j < g.size - 1
    def cross(a, b):
        if lam[b] == lam[a]: return g[a]
        return g[a] + (crit - lam[a]) * (g[b] - g[a]) / (lam[b] - lam[a])
    lo = cross(i - 1, i) if closed_lo else g[0]; hi = cross(j, j + 1) if closed_hi else g[-1]
    closed = closed_lo and closed_hi
    return dict(lo=float(lo), hi=float(hi), width=float(hi - lo), closed=bool(closed),
                identifiable=bool(closed and (hi - lo) < width_max), est=float(g[k]))

# ----------------------------------------------------------------------------------------------- error model
def Sigma(psi, f):
    """Eq. (5): Sigma = V (R_RI kron R_f) V, stacking real then imaginary parts."""
    s0, gam, rf, rri, kap = psi; N = len(f); f = np.asarray(f, float)
    Rf = rf**np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    R = np.kron(np.array([[1.0, rri], [rri, 1.0]]), Rf)
    sd = s0 * (f / f[0])**gam; V = np.diag(np.r_[sd, kap * sd])
    return V @ R @ V

PSI_PUBLISHED = (0.110, 0.009, 0.180, -0.194, 0.459)   # all 51 sweeps (Table S46)

def estimate_error_model(residuals, f, p=4):
    """ML estimate of psi from pooled residual vectors (one per sweep, from M11 fits).
    Plain Gaussian ML (no degrees-of-freedom rescaling); this reproduces the published psi of Table S46 for the 51
    sweeps of the public data (sigma0 0.110, gamma 0.009, rho_f 0.180, rho_RI -0.194, kappa 0.459)."""
    R = np.asarray(residuals); n = R.shape[1]
    def nll(q):
        s0, gam, rf, rri, kap = np.exp(q[0]), q[1], np.tanh(q[2]), np.tanh(q[3]), np.exp(q[4])
        S = Sigma((s0, gam, rf, rri, kap), f)
        sign, ld = np.linalg.slogdet(S)
        if sign <= 0: return 1e30
        Si = np.linalg.inv(S); return 0.5 * (R.shape[0] * ld + np.einsum('ij,jk,ik->', R, Si, R))
    q0 = np.array([np.log(np.std(R) + 1e-6), 0.0, 0.1, 0.0, 0.0])
    o = minimize(nll, q0, method='Nelder-Mead', options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10))
    q = o.x; s0 = np.exp(q[0])
    return (float(s0), float(q[1]), float(np.tanh(q[2])), float(np.tanh(q[3])), float(np.exp(q[4])))

# ----------------------------------------------------------------------------------------------- calibration
def calibrated_threshold(spec, model, names, res_fit, psi, B=200, rng=None):
    """Parametric bootstrap: spectra from the fitted model with relative residuals ~ N(0, lambda Sigma(psi)),
    lambda = RSS / (tr(Sigma) (1 - p/n)); c*_j = 95th percentile of Lambda_j at the simulating value."""
    rng = rng or np.random.default_rng(0)
    p = params(model, res_fit); Zfit = model_Z(model, spec.f, p)
    S = Sigma(psi, spec.f); lam = res_fit['rss'] / (np.trace(S) * (1 - NPAR[model] / spec.n))
    Lc = np.linalg.cholesky(lam * S + 1e-15 * np.eye(spec.n))
    out = {nm: [] for nm in names}
    for _ in range(B):
        e = Lc @ rng.standard_normal(spec.n); Zs = Zfit + np.abs(Zfit) * (e[:spec.N] + 1j * e[spec.N:])
        sp = Spectrum(spec.f, Zs); rmin = fit(sp, model)['rss']
        for nm in names:
            rf = fit_fixed(sp, model, nm, p[nm])['rss']; out[nm].append(sp.n * np.log(max(rf / rmin, 1.0)))
    return {nm: float(np.percentile(v, 95)) for nm, v in out.items()}

# ----------------------------------------------------------------------------------------------- KK test
def kk_test(f, Z, n_voigt=5):
    """Linear KK test: R + jwL + 1/(jwC) + 5 Voigt elements with fixed log-spaced time constants between
    1/(2 pi f_max) and 1/(2 pi f_min), fitted by linear least squares to the |Z|-weighted complex spectrum.
    Returns RMS and maximum (percent) of the stacked real and imaginary relative residuals (Z - Z_KK)/|Z|; this
    definition reproduces Table S1 exactly for the spectra whose data are unchanged."""
    w = 2 * np.pi * np.asarray(f); Z = np.asarray(Z, complex); W = 1 / np.abs(Z)
    taus = np.logspace(np.log10(1 / (2 * np.pi * np.max(f))), np.log10(1 / (2 * np.pi * np.min(f))), n_voigt)
    cols = [np.ones_like(w, dtype=complex), 1j * w, 1 / (1j * w)] + [1 / (1 + 1j * w * t) for t in taus]
    A = np.column_stack(cols) * W[:, None]; Ar = np.vstack([A.real, A.imag]); br = stack(Z * W)
    c, *_ = np.linalg.lstsq(Ar, br, rcond=None); Zk = np.column_stack(cols) @ c
    rel = stack((Z - Zk) / np.abs(Z))
    return dict(rms=float(100 * np.sqrt(np.mean(rel**2))), max=float(100 * np.abs(rel).max()), df=int(2 * len(f) - len(cols)))

# ----------------------------------------------------------------------------------------------- criteria
def _jac(fun, p, h=1e-6):
    r0 = fun(p); J = np.zeros((r0.size, p.size))
    for i in range(p.size):
        d = np.zeros_like(p); d[i] = h * max(1, abs(p[i])); J[:, i] = (fun(p + d) - fun(p - d)) / (2 * d[i])
    return J

def band_factor(f, Rinf, Q, al, L, Wt):
    """D = ||P h~|| and sin(theta) at the R-CPE-L asymptote (Eq. 8)."""
    w = 2 * np.pi * np.asarray(f)
    def rn(p): return stack((10**p[0] + 1 / (10**p[1] * (1j * w)**p[2]) + 1j * w * 10**p[3]) / Wt)
    J = _jac(rn, np.array([np.log10(Rinf), np.log10(Q), al, np.log10(L)])); P = np.eye(J.shape[0]) - J @ np.linalg.pinv(J)
    ht = stack(((1j * w)**(-al) / Q) / Wt * (1j * w / w[0])**(-al)); D = np.linalg.norm(P @ ht)
    return float(D), float(D / np.linalg.norm(ht))

def W_lin(f, Rinf, Q, al, L, fc, sig, z=Z95, Wt=None):
    """Exact linearized (Fisher-information) width of log10 tau, Eq. (A6), for the Cole+L spectrum with the given
    asymptote and characteristic frequency fc."""
    w = 2 * np.pi * np.asarray(f); tau = 1 / (2 * np.pi * fc)
    def cole(q): return 10**q[0] + (10**q[4])**q[2] / 10**q[1] / (1 + (1j * w * 10**q[4])**q[2]) + 1j * w * 10**q[3]
    q0 = np.array([np.log10(Rinf), np.log10(Q), al, np.log10(L), np.log10(tau)])
    Wt = np.abs(cole(q0)) if Wt is None else Wt
    J = _jac(lambda q: stack(cole(q) / Wt), q0)
    try:
        C = np.linalg.inv(J.T @ J); return float(2 * z * sig * np.sqrt(max(C[4, 4], 0)))
    except np.linalg.LinAlgError:
        return np.inf

def f_DL(f, Rinf, Q, al, L, sig, z=Z95, Wstar=1.0, span=(5.0, 0.5)):
    """Operational detection limit: f_c at which W_lin = W* along the asymptote, searched on
    [f_min 10^-5, f_min 10^0.5]. Returns (value, flag) with flag 'ok', 'above' (W_lin > W* at the upper end: no
    dispersion below the upper end is resolvable) or 'below'."""
    fmin = float(np.min(f)); g = lambda lf: np.log(W_lin(f, Rinf, Q, al, L, 10**lf, sig, z)) - np.log(Wstar)
    lo, hi = np.log10(fmin) - span[0], np.log10(fmin) + span[1]
    try:
        if g(hi) > 0: return 10**hi, 'above'
        if g(lo) < 0: return 10**lo, 'below'
        return float(10**brentq(g, lo, hi, xtol=1e-3)), 'ok'
    except Exception:
        return np.nan, 'failed'

def criterion(spec, p11, sigma, z=Z95, Wstar=1.0):
    """Closed form (Eqs. 8-9) and operational detection limit for one spectrum."""
    Rinf, Q, al, L = 10**p11['log10Rinf'], 10**p11['log10Q'], p11['alpha'], 10**p11['log10L']
    D, sinth = band_factor(spec.f, Rinf, Q, al, L, np.abs(spec.Z))
    Gam = (al * LN10 * D * Wstar / (2 * z * sigma))**(1 / al)
    dl, flag = f_DL(spec.f, Rinf, Q, al, L, sigma, z, Wstar)
    return dict(D=D, sin_theta=sinth, Gamma=float(Gam), DL_closed=float(np.min(spec.f) / Gam), f_DL=dl, f_DL_flag=flag)

# ----------------------------------------------------------------------------------------------- data handling
def spectra_from_readings(df, freqs, by=('recording', 'tissue'), method='sweep_median'):
    """Spectrum per (recording, tissue); complete spectra only.
    method='sweep_median' (default): per-frequency median over the per-sweep median spectra of the complete sweeps
      (real and imaginary parts separately), so that every sweep counts once and a few readings cannot switch the
      spectrum between drifting sweeps; recordings without a complete sweep fall back to 'pooled'.
    method='pooled': per-frequency median over all readings (the construction of the earlier analysis)."""
    out = {}
    if method == 'sweep_median':
        sw = sweeps_from_readings(df, freqs, by)
        for key, lst in sw.items():
            if lst:
                Zs = np.array(lst); out[key] = (np.asarray(freqs, float), np.median(Zs.real, 0) + 1j * np.median(Zs.imag, 0))
        rest = df[[tuple(k) not in out for k in df[list(by)].itertuples(index=False, name=None)]]
        if len(rest): out.update(spectra_from_readings(rest, freqs, by, 'pooled'))
        return out
    for key, g in df.groupby(list(by)):
        med = g.groupby('frequency_hz')[['z_real', 'z_imag']].median()
        if all(fq in med.index for fq in freqs):
            m = med.loc[freqs]; out[key] = (np.asarray(freqs, float), m['z_real'].values + 1j * m['z_imag'].values)
    return out

def sweeps_from_readings(df, freqs, by=('recording', 'tissue')):
    """Complete sweeps: runs of readings (acquisition order) in which frequency does not decrease and every
    frequency of the band occurs; per-frequency median within the run."""
    out = {}
    for key, g in df.groupby(list(by)):
        g = g.sort_values('order'); fr = g['frequency_hz'].values; runs = []; start = 0
        for i in range(1, len(fr) + 1):
            if i == len(fr) or fr[i] < fr[i - 1]:
                runs.append(g.iloc[start:i]); start = i
        sw = []
        for r in runs:
            med = r.groupby('frequency_hz')[['z_real', 'z_imag']].median()
            if all(fq in med.index for fq in freqs):
                m = med.loc[freqs]; sw.append(m['z_real'].values + 1j * m['z_imag'].values)
        out[key] = sw
    return out
