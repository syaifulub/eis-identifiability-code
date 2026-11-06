"""Test of the band criteria on measured wide-band spectra (e.g. tetrapolar bioimpedance spectroscopy).

Input: a CSV with columns  spectrum_id, f_Hz, re_ohm, im_ohm  (im negative for capacitive behaviour), one row per
frequency; optional columns sample_type, device, electrodes are carried into the output.

For each spectrum that passes the linear KK test on its full band:
  1. Reference: Cole+L (M10) fit and profile likelihood of log10 tau on the full band.
  2. Truncation (as for the battery column of Table 6): the band is truncated from below (f_max fixed) to every grid frequency that
     leaves at least one decade and 8 frequencies. On each truncated band, the identifiability of log10 tau is
     observed by profile likelihood (nominal threshold) and predicted from that band alone by the closed form
     (f_c_hat > f_min / Gamma), the operational rule (f_c_hat > f_DL) and the three-way rule; an identified tau is
     called accurate if within 0.5 decade of the full-band value.
  3. Prospective band design (Algorithm 2): a pilot spectrum is taken as the top decade of the band
     (f_max/10..f_max). From the pilot alone (R-CPE-L asymptote and effective error), f_DL is computed for every
     candidate band [f_min, f_max] on the measured frequency grid, and identifiability is predicted with the
     full-band f_c as an exact prior, and with the prior wrong by +-0.5 and +-1 decade. The prediction is compared
     with the profile likelihood observed on the candidate band.

Usage:  python -I wideband_test.py spectra.csv OUTDIR [--jobs 4] [--kk 3.0] [--quick] [--per-decade 10] [--fmin-step 0.125]
  --per-decade k : thin each spectrum to about k log-spaced frequencies per decade (0: keep all)
  --fmin-step d  : truncate at lower limits spaced d decades apart (0: every grid frequency)
Outputs: OUTDIR/truncation.csv, OUTDIR/design.csv, OUTDIR/summary.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eis_pipeline as ep

CRIT, WSTAR = ep.CHI95, 1.0
FMIN_STEP = 0.0


def tau_profile(spec, quick=False):
    full = ep.fit(spec, 'M10')
    g, lam = ep.profile(spec, 'M10', 'log10tau', rss_min=full['rss'], step=0.1 if quick else None, refine=not quick)
    it = ep.interval(g, lam, CRIT, WSTAR)
    ident = bool(it['closed'] and it['width'] < WSTAR)
    return full, it, ident


def asymptote(spec):
    r11 = ep.fit(spec, 'M11'); p = ep.params('M11', r11)
    return p


def sigma_eff(spec, full):
    return float(np.sqrt(full['rss'] / (spec.n - ep.NPAR['M10'])))


def balanced(obs, pred):
    obs = np.asarray(obs, bool); pred = np.asarray(pred, bool)
    tp = int(np.sum(obs & pred)); tn = int(np.sum(~obs & ~pred)); fp = int(np.sum(~obs & pred)); fn = int(np.sum(obs & ~pred))
    sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
    ba = np.nanmean([sens, spec]) if not (np.isnan(sens) and np.isnan(spec)) else np.nan
    return dict(n=int(obs.size), pos=int(obs.sum()), TP=tp, TN=tn, FP=fp, FN=fn, sens=sens, spec=spec, BA=ba)


def thin(f, Z, per_decade):
    if per_decade <= 0: return f, Z
    tg = 10**np.arange(np.log10(f.min()), np.log10(f.max()) + 1e-9, 1 / per_decade)
    idx = sorted({int(np.argmin(np.abs(np.log10(f) - np.log10(t)))) for t in tg} | {int(np.argmax(f))})
    return f[idx], Z[idx]


def fim_width(spec, p, sig, z=ep.Z95):
    w = spec.w
    def r(q): return ep.stack((10**q[0] + 10**q[1] / (1 + (1j * w * 10**q[2])**q[3]) + 1j * w * 10**q[4]) * spec.W)
    q0 = np.array([p['log10Rinf'], p['log10dR'], p['log10tau'], p['alpha'], p['log10L']])
    J = ep._jac(r, q0); C = np.linalg.inv(J.T @ J); return float(2 * z * sig * np.sqrt(max(C[2, 2], 0)))


def run_one(args):
    sid, f, Z, meta, quick, step, per_decade = args
    global FMIN_STEP; FMIN_STEP = step
    o = np.argsort(f); f, Z = f[o], Z[o]; f, Z = thin(f, Z, per_decade)
    o = np.argsort(f); f, Z = f[o], Z[o]
    out = dict(trunc=[], design=[], info=dict(spectrum_id=sid, **meta, f_min=float(f[0]), f_max=float(f[-1]), N=int(f.size)))
    kk = ep.kk_test(f, Z); out['info'].update(kk_rms=kk['rms'])
    spec = ep.Spectrum(f, Z)
    full, it, ident = tau_profile(spec, quick)
    lt_full = float(full['xi'][0]); fc_full = 1 / (2 * np.pi * 10**lt_full)
    out['info'].update(log10tau_full=lt_full, fc_full=fc_full, alpha_full=float(full['xi'][1]),
                       tau_ident_full=ident, tau_width_full=it['width'], sigma_full=sigma_eff(spec, full))
    # ---- truncation from below
    fmins = [f[k] for k in range(f.size)]
    if FMIN_STEP > 0:
        tg = 10**np.arange(np.log10(f[0]), np.log10(f[-1] / 10) + 1e-9, FMIN_STEP)
        fmins = sorted({float(f[np.argmin(np.abs(np.log10(f) - np.log10(t)))]) for t in tg})
    for fmin in fmins:
        sub = f >= fmin * (1 - 1e-9)
        if sub.sum() < 8 or f[-1] / fmin < 10 * (1 - 1e-6): break
        s = ep.Spectrum(f[sub], Z[sub])
        fu, iu, idu = tau_profile(s, quick)
        lt = float(fu['xi'][0]); fc_hat = 1 / (2 * np.pi * 10**lt); sig = sigma_eff(s, fu)
        p11 = asymptote(s)
        try:
            cr = ep.criterion(s, p11, sig, Wstar=WSTAR)
        except Exception:
            cr = dict(DL_closed=np.nan, f_DL=np.nan, f_DL_flag='failed', sin_theta=np.nan, Gamma=np.nan)
        pc = bool(fc_hat > cr['DL_closed']); po = bool(cr['f_DL_flag'] in ('ok', 'below') and fc_hat > cr['f_DL'])
        try:
            wl = ep.W_lin(s.f, 10**p11['log10Rinf'], 10**p11['log10Q'], p11['alpha'], 10**p11['log10L'], fc_hat, sig)
        except Exception:
            wl = np.inf
        try:   # FIM/CRLB baseline: Wald width of log10 tau at the full Cole+L fit
            pp = ep.params('M10', fu); wf = fim_width(s, pp, sig)
        except Exception:
            wf = np.inf
        three = 'likely' if (pc and po) else ('unlikely' if (not pc and not po) else 'uncertain')
        out['trunc'].append(dict(spectrum_id=sid, f_min=float(fmin), N=int(sub.sum()), log10tau=lt, fc_hat=fc_hat,
                                 dispersion=('within' if fc_full >= fmin else 'below'), sigma=sig,
                                 obs_ident=idu, width=iu['width'], closed=iu['closed'],
                                 accurate=bool(idu and abs(lt - lt_full) < 0.5), dev=float(lt - lt_full),
                                 pred_closed=pc, pred_fDL=po, three_way=three, DL_closed=cr['DL_closed'], f_DL=cr['f_DL'],
                                 f_DL_flag=cr['f_DL_flag'], sin_theta=cr['sin_theta'],
                                 W_lin_asym=wl, pred_Wlin=bool(wl < WSTAR), W_fim=wf, pred_fim=bool(wf < WSTAR),
                                 alpha_asym=p11['alpha']))
    # ---- prospective band design from a pilot (top decade)
    pil = f >= f[-1] / 10 * (1 - 1e-9)
    if pil.sum() >= 6:
        sp = ep.Spectrum(f[pil], Z[pil]); fp_ = ep.fit(sp, 'M10'); sigp = sigma_eff(sp, fp_); a = asymptote(sp)
        Rinf, Q, al, L = 10**a['log10Rinf'], 10**a['log10Q'], a['alpha'], 10**a['log10L']
        obs_by_fmin = {round(r['f_min'], 9): r['obs_ident'] for r in out['trunc']}
        for r in out['trunc']:
            fmin = r['f_min']; cand = f[f >= fmin * (1 - 1e-9)]
            dl, flag = ep.f_DL(cand, Rinf, Q, al, L, sigp, Wstar=WSTAR)
            row = dict(spectrum_id=sid, f_min=fmin, pilot_sigma=sigp, pilot_alpha=al, f_DL=dl, f_DL_flag=flag,
                       obs_ident=obs_by_fmin[round(fmin, 9)], fc_full=fc_full)
            for lab, shift in (('exact', 0.0), ('m05', -0.5), ('p05', 0.5), ('m1', -1.0), ('p1', 1.0)):
                fce = fc_full * 10**shift
                row['pred_' + lab] = bool(flag in ('ok', 'below') and fce > dl)
                try:
                    wl = ep.W_lin(cand, Rinf, Q, al, L, fce, sigp)
                except Exception:
                    wl = np.inf
                row['Wlin_' + lab] = wl; row['predW_' + lab] = bool(wl < WSTAR)
            out['design'].append(row)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('csv'); ap.add_argument('outdir')
    ap.add_argument('--jobs', type=int, default=4); ap.add_argument('--kk', type=float, default=3.0)
    ap.add_argument('--quick', action='store_true'); ap.add_argument('--fmin-step', type=float, default=0.0); ap.add_argument('--per-decade', type=float, default=0.0); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.csv)
    metacols = [c for c in ('sample_type', 'device', 'electrodes', 'source') if c in df.columns]
    tasks = []
    for sid, g in df.groupby('spectrum_id', sort=False):
        g = g.sort_values('f_Hz'); Z = g['re_ohm'].to_numpy() + 1j * g['im_ohm'].to_numpy()
        meta = {c: str(g[c].iloc[0]) for c in metacols}
        tasks.append((str(sid), g['f_Hz'].to_numpy(float), Z, meta, a.quick, a.fmin_step, a.per_decade))
    if a.limit: tasks = tasks[:a.limit]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(run_one, tasks))
    info = pd.DataFrame([r['info'] for r in res]); info['kk_ok'] = info['kk_rms'] < a.kk
    ok = set(info.loc[info.kk_ok, 'spectrum_id'])
    tr = pd.DataFrame([x for r in res for x in r['trunc']]); de = pd.DataFrame([x for r in res for x in r['design']])
    info.to_csv(os.path.join(a.outdir, 'spectra.csv'), index=False)
    tr.to_csv(os.path.join(a.outdir, 'truncation.csv'), index=False); de.to_csv(os.path.join(a.outdir, 'design.csv'), index=False)
    S = dict(n_spectra=int(len(info)), n_kk_ok=int(info.kk_ok.sum()),
             tau_ident_full=int(info.loc[info.kk_ok, 'tau_ident_full'].sum()))
    if len(tr):
        t = tr[tr.spectrum_id.isin(ok)]
        S['truncation'] = dict(n_cases=int(len(t)),
                               closed_form=balanced(t.obs_ident, t.pred_closed), f_DL=balanced(t.obs_ident, t.pred_fDL),
                               W_lin_asymptote=balanced(t.obs_ident, t.pred_Wlin), FIM_full_fit=balanced(t.obs_ident, t.pred_fim))
        d3 = t[t.three_way != 'uncertain']
        S['truncation']['three_way'] = dict(decided=int(len(d3)), referred=int((t.three_way == 'uncertain').sum()),
                                            **balanced(d3.obs_ident, d3.three_way == 'likely'))
        for disp in ('within', 'below'):
            u = t[(t.dispersion == disp) & t.obs_ident]
            S['truncation']['accuracy_' + disp] = dict(identified=int(len(u)), accurate=int(u.accurate.sum()),
                                                       median_abs_dev=float(np.median(np.abs(u.dev))) if len(u) else None)
    if len(de):
        d = de[de.spectrum_id.isin(ok)]
        S['design'] = {lab: balanced(d.obs_ident, d['pred_' + lab]) for lab in ('exact', 'm05', 'p05', 'm1', 'p1')}
        S['design_Wlin'] = {lab: balanced(d.obs_ident, d['predW_' + lab]) for lab in ('exact', 'm05', 'p05', 'm1', 'p1')}
        S['design']['n_cases'] = int(len(d))
    json.dump(S, open(os.path.join(a.outdir, 'summary.json'), 'w'), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float))


if __name__ == '__main__':
    main()
