"""Repeated sweeps (Sects. 3.1, 4.4): spectrum construction, drift and the drift budget of the sweep duration."""
from __future__ import annotations
import numpy as np


def sweep_median(sweeps):
    """Median of the per-sweep spectra (real and imaginary parts separately). sweeps: array (n_sweeps, N) complex.
    Recommended instead of pooling readings across sweeps, which mixes drifting sweeps."""
    S = np.asarray(sweeps, complex)
    return np.median(S.real, 0) + 1j * np.median(S.imag, 0)


def drift_rate(sweeps, t_start):
    """Median over consecutive sweep pairs of the largest relative change of |Z| over the band per second.
    t_start: start time (s) of each sweep."""
    S = np.abs(np.asarray(sweeps, complex)); t = np.asarray(t_start, float)
    if S.shape[0] < 2: return float('nan')
    r = [np.max(np.abs(S[i + 1] - S[i]) / S[i]) / (t[i + 1] - t[i]) for i in range(S.shape[0] - 1) if t[i + 1] > t[i]]
    return float(np.median(r)) if r else float('nan')


def drift_budget(sigma, rate):
    """Longest sweep duration T (s) for which within-sweep drift stays below the effective error: T <~ sigma/rate."""
    return float(sigma / rate) if rate > 0 else float('inf')
