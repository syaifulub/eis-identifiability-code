# eisvarpro

Identifiability-aware variable-projection fitting and frequency-band assessment of Cole-type equivalent circuits.

`eisvarpro` implements the methods of

> S. Anam, A. Zarkasi, Z. Md Rodzi, *Identifiability-Aware Frequency-Band Assessment for Impedance Measurement Systems: Closed-Form Criteria and Validation on Tissue, Battery and Tetrapolar Spectra* (submitted to the
> Iranian Journal of Science and Technology, Transactions of Electrical Engineering).

It fits the Cole+L (M10) and R–CPE–L (M11) circuits by bounded variable projection, checks whether the measured band
identifies each parameter (profile likelihood, optionally bootstrap-calibrated), applies the decision rule of the
article (Fig. 5) and screens candidate frequency bands from a pilot spectrum (Algorithm 2).

## Install

```
pip install .            # from this folder; requires Python >= 3.10, NumPy, SciPy
pip install .[test]      # with pytest
pytest -q tests          # 11 tests, about 1 min
```
In an offline environment: `pip install --no-build-isolation .`. If the build fails with an old system setuptools
(e.g. `AttributeError: install_layout`), use a virtual environment with a current setuptools, or run without installing:
`PYTHONPATH=src python -m pytest tests`, `PYTHONPATH=src python -m eisvarpro.cli analyze file.csv`.

## Quick start

```python
import numpy as np, eisvarpro as ev
a = np.genfromtxt('examples/synthetic_narrow_band.csv', delimiter=',', names=True)
f, Z = a['f_Hz'], a['re_ohm'] + 1j * a['im_ohm']

r = ev.analyze(f, Z)                                # Algorithm 1 + decision rule (nominal threshold)
r = ev.analyze(f, Z, psi=ev.PSI_PUBLISHED, B=200)   # thresholds calibrated with an error model (Eq. 5)
print(r['report'])          # what may be reported, and why
print(r['criteria'])        # D, sin(theta), Gamma, f_min/Gamma, f_DL

out = ev.screen_bands(f, Z, [(10e3, 100e3, 20), (300, 100e3, 20)], fc_prior=1e3)   # Algorithm 2
```

Command line (CSV columns `f_Hz, re_ohm, im_ohm`; imaginary part negative for capacitive behaviour):
```
eisvarpro analyze examples/synthetic_narrow_band.csv [--B 200] [--psi s0,gamma,rho_f,rho_RI,kappa]
eisvarpro screen  pilot.csv --bands 10e3:100e3:20 300:100e3:20 --fc-prior 1000 [--delta 0.5] [--kappa-s 1.34]
```
`python examples/run_examples.py` runs five documented examples (narrow and wide band, Algorithm 2, the domain check on
a measured tetrapolar pilot, the drift budget).

## Correspondence with the article

| Article | Function |
|---|---|
| Objective, Eq. (4); weights 1/max(\|Z\|, 1 Ω) | `_core.Spectrum`, `_core.weights` |
| M10 (Cole+L), M11 (R–CPE–L), bounds (Online Resource 1, Table S1) | `_core.basis`, `_core.BOUNDS`, `_core.params`, `_core.model_Z` |
| Bounded variable projection, Eq. (6): shifted NNLS inner problem, multistart TRF | `_core.fit`, `_core.bounded_lsq`; `estimation.fit_models` |
| Structured error model Σ(ψ), Eq. (5); published ψ | `_core.Sigma`, `_core.estimate_error_model`, `PSI_PUBLISHED` |
| Profile likelihood; nominal (3.84) and bootstrap-calibrated thresholds c* | `_core.profile`, `_core.interval`, `_core.calibrated_threshold`; `estimation.profiles` |
| Identifiability: closed interval narrower than 1 decade (0.2 for α) | `estimation.WSTAR`, `estimation.WSTAR_ALPHA` |
| Linear Kramers–Kronig test, 3 % RMS | `_core.kk_test`, `estimation.KK_LIMIT` |
| Band factor D and confounding angle, Eq. (8) | `criteria.band_factor` |
| Closed-form width, Eq. (8); Γ and f_min/Γ, Eq. (9) | `criteria.width_closed`, `criteria.gamma`, `criteria.below_band` |
| Descriptor bias, Eq. (10) | `criteria.descriptor_bias` |
| Tolerable error σ_max, Eq. (11) | `criteria.sigma_max` |
| Dispersion above the band, Eq. (12); window f_min/Γ < f_c < Γ′ f_max | `criteria.fit_lowfreq_asymptote`, `criteria.above_band` |
| Exact linearized width W_lin, Eq. (A6); operational detection limit f_DL | `criteria.W_lin`, `criteria.f_DL` |
| Three-way rule (likely / uncertain / unlikely) | `criteria.three_way`, `criteria.predict_below` |
| Fisher (Wald) baseline at the full Cole+L fit (Table 5) | `criteria.wald_width_full_fit` |
| Algorithm 1 and decision rule (Fig. 5) | `estimation.analyze`, `estimation.decision_rule` |
| Algorithm 2 with the domain check (pilot exponent at a bound → refer) | `design.screen_bands` |
| Per-sweep median spectrum, drift rate, drift budget T ≲ σ/ḋ (Sect. 4.3, Table 7) | `sweeps.sweep_median`, `sweeps.drift_rate`, `sweeps.drift_budget` |

`_core.py` is a verbatim copy of the numerical part of `code/pipeline/eis_pipeline.py`, which produced the measured-data
results of the article; `tests/test_eisvarpro.py::test_core_identical_to_pipeline` checks that both give identical
fits and detection limits.

## Scope and cautions (from the article)

* The closed form (Eqs. 8–12) is an analytical scaling law. As a numerical interval it is anti-conservative (coverage
  58–72 %); decisions use `W_lin` / `f_DL`.
* Domain: a Cole (or ZARC) element with the dispersion clearly outside the band (r = (f_min/f_c)^−α ≤ 0.5) and an
  estimable asymptote, without strong electrode polarization. Outside it, `screen_bands` refers bands as *uncertain*.
* Calibrated intervals are conditional on the fitted model and the error model; derive ψ from repeated sweeps of the
  instrument in use.
* Identifiable is not accurate: model discrepancy can place a spurious dispersion inside the band (article, Sect. 3.7).
  The requirement f_min ≤ f̂_c ≤ f_max of the decision rule catches a fitted dispersion outside the band, not this.
* Algorithm 2 is a screen, not a validated design optimization: confirm a chosen band by measurement and a
  profile-likelihood check, with the error level expected in the candidate band.

## Examples

`examples/synthetic_narrow_band.csv` and `synthetic_wide_band.csv` are simulated (Cole+L, f_c = 1 kHz, 1 % noise).
`examples/tetrapolar_SFB7_ID0001.csv` is spectrum ID 1 of the public compilation of Dell'Osa, Bravo and Villanueva
Jousset (Zenodo, doi:10.5281/zenodo.15795175, CC BY 4.0; original measurement: Ayllón, Gil-Pita and Seoane, PLoS ONE
11:e0156522, 2016), thinned to 10 frequencies per decade.

## License

MIT (see LICENSE).
