"""Command line: eisvarpro analyze spectrum.csv | eisvarpro screen pilot.csv --bands 0.3e3:100e3:20 ...
CSV columns: f_Hz, re_ohm, im_ohm (im negative for capacitive behaviour)."""
from __future__ import annotations
import argparse, json, sys
import numpy as np
from .estimation import analyze
from .design import screen_bands
from ._core import PSI_PUBLISHED


def _read(path):
    a = np.genfromtxt(path, delimiter=',', names=True)
    return a['f_Hz'], a['re_ohm'] + 1j * a['im_ohm']


def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items() if k != 'spectrum'}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(prog='eisvarpro'); sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze', help='Algorithm 1 + decision rule for one spectrum')
    a.add_argument('csv'); a.add_argument('--B', type=int, default=0, help='bootstrap replicates (0: nominal threshold)')
    a.add_argument('--psi', default=None, help='error model sigma0,gamma,rho_f,rho_RI,kappa (default: published)')
    s = sub.add_parser('screen', help='Algorithm 2 from a pilot spectrum')
    s.add_argument('csv'); s.add_argument('--bands', nargs='+', required=True, help='fmin:fmax:N (log-spaced)')
    s.add_argument('--fc-prior', type=float, default=None); s.add_argument('--delta', type=float, default=0.0)
    s.add_argument('--kappa-s', type=float, default=1.0); s.add_argument('--sigma-candidate', type=float, default=None)
    o = ap.parse_args(argv)
    f, Z = _read(o.csv)
    if o.cmd == 'analyze':
        psi = tuple(map(float, o.psi.split(','))) if o.psi else PSI_PUBLISHED
        res = analyze(f, Z, psi=psi, B=o.B)
    else:
        bands = [tuple(float(x) if i < 2 else int(x) for i, x in enumerate(b.split(':'))) for b in o.bands]
        res = screen_bands(f, Z, bands, fc_prior=o.fc_prior, delta=o.delta, kappa_S=o.kappa_s,
                           sigma_candidate=o.sigma_candidate)
    json.dump(_clean(res), sys.stdout, indent=1); print()


if __name__ == '__main__':
    main()
