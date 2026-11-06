"""Run the measured-data analysis end to end.

Input : a tidy CSV of individual readings (see README.md, "Input format"), e.g. produced by
        prepare_readings.py from the public dataset, and spectra_manifest.csv (the 20 analysed spectra).
Output: CSV/JSON tables in --out:
  spectra_qc.csv        Table S1   KK residuals, M11 NRMSE, number of complete sweeps
  error_model.json      Table S46  psi estimated from the complete sweeps (all / KK-consistent only)
  fits.csv              M10 and M11 parameters, RSS, AICc, BIC
  profiles.csv          Table 5 / S15  nominal and calibrated intervals, closure, identifiability
  criterion.csv         Table S26 + operational detection limit f_DL (per spectrum)
  truncation.csv        Table 6 / S50  per spectrum and sub-band, incl. f_DL
  truncation_summary.csv Table 6     medians over KK-consistent spectra
  run_info.json         versions, options, timing

Usage: python run_pipeline.py readings.csv --out results [--B 200] [--psi published|estimated] [--quick]
"""
import argparse, json, time, sys, platform, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eis_pipeline as ep

FREQS = [10e3 * k for k in range(1, 11)]
SUBBANDS = {'10-100': FREQS, '20-100': FREQS[1:], '30-100': FREQS[2:], '40-100': FREQS[3:], '10-70': FREQS[:7]}

def process(r, spec_fZ, kkok, psi, B, step, seed):
    fits = []; prof = []; crit = []; trunc = []; rng = np.random.default_rng(seed)
    names = {'M10': ['log10tau', 'log10dR'], 'M11': ['log10Rinf', 'log10Q', 'alpha']}
    if True:
        f, Z = spec_fZ; sp = ep.Spectrum(f, Z)
        res = {m: ep.fit(sp, m) for m in ('M10', 'M11')}; P = {m: ep.params(m, res[m]) for m in res}
        for m in res:
            fits.append(dict(spectrum=r.spectrum, tissue=r.tissue, model=m, rss=res[m]['rss'], nrmse=np.sqrt(res[m]['rss'] / sp.N),
                             **ep.info_criteria(res[m]['rss'], sp.n, ep.NPAR[m]), **P[m]))
        cstar = {m: ep.calibrated_threshold(sp, m, names[m], res[m], psi, B=B, rng=rng) for m in names}
        intervals = {}
        for m in names:
            for nm in names[m]:
                g, lam = ep.profile(sp, m, nm, res[m]['rss'], step=(0.005 if nm == 'alpha' else 0.05) * step)
                wmax = 0.2 if nm == 'alpha' else 1.0
                for lab, c in (('nominal', ep.CHI95), ('calibrated', cstar[m][nm])):
                    iv = ep.interval(g, lam, c, wmax); intervals[(m, nm, lab)] = iv
                    prof.append(dict(spectrum=r.spectrum, tissue=r.tissue, kk_consistent=kkok, model=m, parameter=nm, threshold=lab,
                                     critical_value=c, **iv))
        sigma = np.sqrt(res['M10']['rss'] / (sp.n - 5))
        cn = ep.criterion(sp, P['M11'], sigma); cc = ep.criterion(sp, P['M11'], sigma, z=np.sqrt(cstar['M10']['log10tau']))
        fc_hat = 1 / (2 * np.pi * 10**P['M10']['log10tau']); ivt = intervals[('M10', 'log10tau', 'nominal')]
        fc_upper = 1 / (2 * np.pi * 10**ivt['lo'])
        Rinf, Q, al, L = 10**P['M11']['log10Rinf'], 10**P['M11']['log10Q'], P['M11']['alpha'], 10**P['M11']['log10L']
        W8 = 2 * ep.Z95 * sigma / (al * ep.LN10 * cn['D']) * (f[0] / fc_hat)**al
        Wl = ep.W_lin(f, Rinf, Q, al, L, fc_hat, sigma, Wt=np.abs(Z))
        pred_c = fc_hat > cn['DL_closed']; pred_o = fc_hat > cn['f_DL']
        crit.append(dict(spectrum=r.spectrum, tissue=r.tissue, kk_consistent=kkok, sigma=sigma, alpha_m10=P['M10']['alpha'], alpha_m11=al,
                         D=cn['D'], sin_theta=cn['sin_theta'], Gamma=cn['Gamma'], DL_closed_nominal=cn['DL_closed'], DL_closed_calibrated=cc['DL_closed'],
                         f_DL_nominal=cn['f_DL'], f_DL_flag=cn['f_DL_flag'], f_DL_calibrated=cc['f_DL'], fc_hat=fc_hat, fc_upper=fc_upper,
                         W8_at_fc_hat=W8, Wlin_at_fc_hat=Wl, W_profile=ivt['width'], ident_closed=pred_c, ident_fDL=pred_o,
                         three_way='likely' if (pred_c and pred_o) else ('unlikely' if not (pred_c or pred_o) else 'uncertain'),
                         ident_observed=ivt['identifiable'], c_star_log10tau=cstar['M10']['log10tau']))
        # band truncation (nominal threshold)
        for sb, fs in SUBBANDS.items():
            idx = [FREQS.index(x) for x in fs]; s2 = ep.Spectrum(np.array(fs), Z[idx]); r10 = ep.fit(s2, 'M10'); r11 = ep.fit(s2, 'M11')
            p10, p11 = ep.params('M10', r10), ep.params('M11', r11); sg = np.sqrt(r10['rss'] / (s2.n - 5))
            c2 = ep.criterion(s2, p11, sg); g, lam = ep.profile(s2, 'M10', 'log10tau', r10['rss'], step=0.05 * step)
            it = ep.interval(g, lam, ep.CHI95, 1.0); fch = 1 / (2 * np.pi * 10**p10['log10tau'])
            idq = {}
            for nm, wm in (('log10Q', 1.0), ('alpha', 0.2)):
                g2, l2 = ep.profile(s2, 'M11', nm, r11['rss'], step=(0.005 if nm == 'alpha' else 0.05) * step); idq[nm] = ep.interval(g2, l2, ep.CHI95, wm)['identifiable']
            trunc.append(dict(spectrum=r.spectrum, tissue=r.tissue, kk_consistent=kkok, subband=sb, N=len(fs), Gamma=c2['Gamma'], DL_closed=c2['DL_closed'],
                              f_DL=c2['f_DL'], f_DL_flag=c2['f_DL_flag'], fc_hat=fch, fc_upper=1 / (2 * np.pi * 10**it['lo']),
                              tau_ident=it['identifiable'], tau_pred_closed=fch > c2['DL_closed'], tau_pred_fDL=fch > c2['f_DL'],
                              Q_ident=idq['log10Q'], alpha_ident=idq['alpha']))
    return fits, prof, crit, trunc

def _star(j):
    return process(*j)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('readings'); ap.add_argument('--manifest', default=os.path.join(os.path.dirname(__file__), 'spectra_manifest.csv'))
    ap.add_argument('--out', default='results'); ap.add_argument('--B', type=int, default=200)
    ap.add_argument('--psi', choices=['published', 'estimated'], default='estimated')
    ap.add_argument('--kk-threshold', type=float, default=3.0); ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--quick', action='store_true', help='coarser profile grids (testing only)')
    ap.add_argument('--jobs', type=int, default=1, help='parallel processes (one spectrum per task)')
    ap.add_argument('--spectrum-method', choices=['sweep_median', 'pooled'], default='sweep_median')
    ap.add_argument('--spectra', default='', help='comma-separated spectrum numbers to analyse (default: all)')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    step = 2 if a.quick else 1

    df = pd.read_csv(a.readings, dtype={'recording': str}); man = pd.read_csv(a.manifest, dtype={'recording': str})
    if a.spectra: man = man[man.spectrum.isin([int(x) for x in a.spectra.split(',')])]
    df = df[df['mode'].str.lower().eq('scan')] if 'mode' in df else df
    spectra = ep.spectra_from_readings(df, FREQS, method=a.spectrum_method); sweeps = ep.sweeps_from_readings(df, FREQS)
    keys = [(r.recording, r.tissue) for r in man.itertuples()]
    missing = [k for k in keys if k not in spectra]
    if missing: print('WARNING: spectra not found / incomplete:', missing)

    # --- quality control and M11 fits of sweeps (error model)
    qc = []; res_all = []; res_kk = []
    for r in man.itertuples():
        k = (r.recording, r.tissue)
        if k not in spectra: continue
        f, Z = spectra[k]; kk = ep.kk_test(f, Z); sp = ep.Spectrum(f, Z); f11 = ep.fit(sp, 'M11')
        ok = kk['rms'] < a.kk_threshold
        qc.append(dict(spectrum=r.spectrum, tissue=r.tissue, recording=r.recording, animal=r.animal, kk_rms_pct=kk['rms'],
                       kk_max_pct=kk['max'], kk_consistent=ok, m11_nrmse=np.sqrt(f11['rss'] / sp.N), n_sweeps=len(sweeps.get(k, []))))
        for Zs in sweeps.get(k, []):
            ss = ep.Spectrum(f, Zs); fs = ep.fit(ss, 'M11'); pr = ep.params('M11', fs)
            r_ = ep.stack((ep.model_Z('M11', f, pr) - Zs) / np.abs(Zs)); res_all.append(r_)
            if ok: res_kk.append(r_)
    qc = pd.DataFrame(qc); qc.to_csv(os.path.join(a.out, 'spectra_qc.csv'), index=False)
    em = {}
    if len(res_all) >= 5: em['all_sweeps'] = dict(psi=ep.estimate_error_model(res_all, FREQS), n_sweeps=len(res_all))
    if len(res_kk) >= 5: em['kk_consistent'] = dict(psi=ep.estimate_error_model(res_kk, FREQS), n_sweeps=len(res_kk))
    em['published_all_sweeps'] = dict(psi=ep.PSI_PUBLISHED)
    json.dump(em, open(os.path.join(a.out, 'error_model.json'), 'w'), indent=1)
    psi = em['all_sweeps']['psi'] if (a.psi == 'estimated' and 'all_sweeps' in em) else ep.PSI_PUBLISHED
    print('error model psi =', np.round(psi, 3), '(%s)' % ('estimated' if psi is not ep.PSI_PUBLISHED else 'published'))

    # --- fits, profiles, calibration, criterion (one task per spectrum; --jobs processes)
    from types import SimpleNamespace
    jobs = [(SimpleNamespace(**r._asdict()), spectra[(r.recording, r.tissue)], bool(qc.loc[qc.spectrum == r.spectrum, 'kk_consistent'].iloc[0]), psi, a.B, step, a.seed + int(r.spectrum))
            for r in man.itertuples() if (r.recording, r.tissue) in spectra]
    fits = []; prof = []; crit = []; trunc = []
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.jobs) as ex:
            results = list(ex.map(_star, jobs))
    else:
        results = []
        for j in jobs:
            results.append(_star(j)); print('spectrum %2d %-12s done (%.0f s)' % (j[0].spectrum, j[0].tissue, time.time() - t0), flush=True)
    for fi, pr, cr, tr in results:
        fits += fi; prof += pr; crit += cr; trunc += tr
    pd.DataFrame(fits).to_csv(os.path.join(a.out, 'fits.csv'), index=False)
    pd.DataFrame(prof).to_csv(os.path.join(a.out, 'profiles.csv'), index=False)
    pd.DataFrame(crit).to_csv(os.path.join(a.out, 'criterion.csv'), index=False)
    T = pd.DataFrame(trunc); T.to_csv(os.path.join(a.out, 'truncation.csv'), index=False)
    K = T[T.kk_consistent]
    if len(K):
        summ = K.groupby('subband').agg(N=('N', 'first'), Gamma=('Gamma', 'median'), DL_closed_khz=('DL_closed', lambda x: np.median(x) / 1e3),
                                        f_DL_khz=('f_DL', lambda x: np.median(x) / 1e3), fc_upper_khz=('fc_upper', lambda x: np.median(x) / 1e3),
                                        tau_observed=('tau_ident', 'sum'), tau_pred_closed=('tau_pred_closed', 'sum'), tau_pred_fDL=('tau_pred_fDL', 'sum'),
                                        Q_ident=('Q_ident', 'sum'), alpha_ident=('alpha_ident', 'sum'))
        summ.to_csv(os.path.join(a.out, 'truncation_summary.csv'))
    json.dump(dict(options=vars(a), psi=psi, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                   seconds=time.time() - t0), open(os.path.join(a.out, 'run_info.json'), 'w'), indent=1, default=str)
    print('done in %.0f s; results in %s' % (time.time() - t0, a.out))

if __name__ == '__main__':
    main()
