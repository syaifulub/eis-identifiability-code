> **Note:** historical README of an earlier archive version (manuscript then submitted to another journal). For the current article and numbering, see the root `README.md`.

# Online Resource 2 — program package

Code for S. Anam, A. Zarkasi, Z. Md Rodzi, *Identifiability-Aware Parameter Estimation and Frequency-Band Assessment
of Equivalent-Circuit Models: Application to In Vivo Tissue Bioimpedance* (Arabian Journal for Science and
Engineering, submitted). Corresponding author: Syaiful Anam (syaiful@ub.ac.id).

| Folder | Content | Start with |
|---|---|---|
| `eisvarpro/` | Installable Python package implementing the methods: Algorithm 1 (VarPro fitting, profile likelihood, bootstrap calibration), the decision rule (Fig. 5), the band criteria (Eqs. 8–12, A8, f_DL, three-way rule, Wald baseline) and Algorithm 2 with its domain check; tests and examples | `eisvarpro/README.md` |
| `code/pipeline/` | Pipeline for the measured tissue spectra (public data of Falagario et al.), reproduction from the public data, sweep-median spectra and drift, and the tetrapolar test | `code/pipeline/README.md` |
| `code/` | Scripts of the synthetic studies, the measured battery test and the domain check, with the JSON outputs used in the article | `code/README.md` |
| `figsrc/` | TikZ sources of Figs. 1 and 5 | `pdflatex Fig1_workflow.tex` |

## Install and test

```
cd eisvarpro
pip install .                    # Python >= 3.10, NumPy, SciPy  (offline: pip install --no-build-isolation .)
pytest -q tests                  # 11 tests, about 1 min
python examples/run_examples.py  # five documented examples
eisvarpro analyze examples/synthetic_narrow_band.csv
```
Without installing: `PYTHONPATH=src python -m pytest tests` and `PYTHONPATH=src python -m eisvarpro.cli ...` (from `eisvarpro/`).
The reproduction scripts additionally need pandas and Matplotlib.

## Where each result comes from

| Article | Folder / command |
|---|---|
| Tables 5, 6 (incl. f_DL column), S1, S13–S15, S26, S46, S50; Sects. 3.1–3.6 | `code/pipeline/` (`run_pipeline.py`), `code/subband_fdl.py` |
| Tables S67–S69 (reproduction from public data, sweep-median spectra, drift) | `code/pipeline/` (`compare_to_published.py`, `compare_data_versions.py`, `drift_analysis.py`) |
| Table 7, Tables S62–S66, domain check (Sect. 3.7) | `code/` (`classify_sim.py`, `alg2_test.py`, `threeway.py`, `alg2_sens.py`, `alg2_domain.py`) |
| Table 8, Tables S71, S73 (measured battery and tetrapolar tests, Sect. 3.8) | `code/battery_measured.py`; `code/pipeline/wideband_test.py` |
| Appendix A, Tables S55–S61, S70; Fig. 6, Fig. S17 | `code/` (`verify_criteria.py`, `mc_rigor.py`, `mc_plugin.py`, `boot_B.py`, `window_sim.py`, `ee_relevance.py`, `fig6_split.py`) |

Detailed commands and run times are in `code/README.md` and `code/pipeline/README.md`. The measured data are not
redistributed except for one example spectrum (CC BY 4.0, see `eisvarpro/README.md`); download links are given in
the READMEs.

## Data sources
* Falagario, De Momi, Mattos, *In vivo tissue bioimpedance dataset*, Scientific Data (2026); Zenodo record 20408128.
* Zhang et al., Nature Communications 11:1706 (2020); Zenodo 10.5281/zenodo.3633835 (CC BY 4.0).
* Dell'Osa, Bravo, Villanueva Jousset, *Single dispersion EIS measurements: a compilation*, Zenodo
  10.5281/zenodo.15795175 (CC BY 4.0); SFB7 subset from Ayllón, Gil-Pita, Seoane, PLoS ONE 11:e0156522 (2016).

## License
MIT for all code (see `eisvarpro/LICENSE`). Data files keep the licenses of their sources.
