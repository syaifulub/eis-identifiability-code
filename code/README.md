# Reproduction scripts

These scripts reproduce the synthetic studies and the tests on public battery data of the article. Numbers refer to
the submitted article (main text: Tables 1-7, Figs. 1-6, Appendix A; Online Resource 1: Notes S1-S5, Tables S1-S8).
*Report Table Sxx* refers to `../docs/extended_results.pdf`, an extended results report of an earlier manuscript
version whose S-numbers differ from Online Resource 1. The pipeline for the measured tissue spectra and the
tetrapolar test is in `pipeline/` (see `pipeline/README.md`). The reusable implementation of the methods is the
package `../eisvarpro/`.

## Requirements
Python >= 3.10, NumPy >= 1.24, SciPy >= 1.10, pandas >= 2.0 (pipeline, analyses), Matplotlib >= 3.7 (figures); tested
with Python 3.13, NumPy 2.5, SciPy 1.18. All random seeds are fixed. The JSON files in this folder are the outputs used
in the article; rerunning a script overwrites them.

## Reproduction (run from this folder)
| Output in the article | Command | Runtime (1 core) |
|---|---|---|
| Appendix A, Step 9; Note S2.2 (report Tables S55-S56) | `python verify_criteria.py verify_criteria_output.json` | ~1 min |
| kappa_S decomposition (Appendix A, Step 8; Note S2.1) | `python verify_kappa_decomposition.py` | seconds |
| Data of Fig. 6 (report Tables S57-S58, Fig. S17) | `python ee_relevance.py ee_relevance_output.json` | ~2 min |
| Fig. 6 | `python fig6_split.py [ee_relevance_output.json]` | seconds |
| Monte Carlo widths and coverage (Sect. 2.6; Note S2.2; report Table S59) | `python mc_rigor.py mc_out.json` | ~3 min |
| Plug-in detection limit (Note S2.2; report Table S60) | `python mc_plugin.py mc_plugin.json` | ~2 min |
| Bootstrap replicates B (report Table S61) | `python boot_B.py boot_out.json` | ~5 min |
| Table 5 (classification study, Fisher baseline; report Table S62) | `python classify_sim.py class_out.json`, then `python analyse_class.py` | ~7 min |
| Cluster-bootstrap intervals (Sect. 3.6) | `python ci_class.py` (needs `class_out.json`, `alg2_out.json`) | ~3 min |
| End-to-end timing (Table 5, Sect. 4.4) | `python timing_e2e.py timing_e2e.json` | ~1 min |
| Algorithm 2 with priors (Sect. 3.6; report Table S63) | `python alg2_test.py alg2_out.json`, then `python analyse_alg2.py` | ~1 min |
| Domain check of Algorithm 2 (Sect. 3.6; Note S4) | `python alg2_domain.py` (reads `alg2_out.json`, `class_out.json`) | seconds |
| Synthetic battery-type spectra (Sect. 3.7; report Table S64) | `python battery_grid.py battery_out.json` (uses `ee_relevance.py`) | ~1 min |
| Three-way rule (Sect. 3.6; report Table S65) | `python threeway.py` | seconds |
| Noise or electrode CPE in the candidate band (Sect. 3.6; report Table S66) | `python alg2_sens.py alg2_sens.json`, then `python analyse_sens.py` | ~8 min |
| Table 4, column f_DL | `python subband_fdl.py` | seconds |
| Dispersion above the band (Sect. 3.3; Note S2.3, Table S4) | `python window_sim.py window_out.json`, then `python analyse_window.py window_out.json`; bounds: `python window_constraint_check.py window_sim.py constraint_check.json` | ~5 min |
| Table 6, battery state V at 25 C (report Table S71) | `python battery_measured.py "EIS data" battery_measured.json 4`, then `python analyse_battery_measured.py` (folder `EIS data` from Zenodo 10.5281/zenodo.3633835) | ~20 min |
| Table 6, battery other states and temperatures; Note S3, Table S5 | `bash run_battery_ext.sh "EIS data"` (uses `battery_measured_ext.py`, `analyse_battery_ext.py`; outputs `battery_ext_all.json`, `battery_ext_metrics.json`) | ~10 min |
| Table 6, tetrapolar column; Note S4, Table S6 | see `pipeline/README.md`, Sect. 8.1 | ~10 min |
| Tables 3 and 4, Sects. 3.1-3.5; Note S5, Tables S7-S8 (measured tissue spectra) | see `pipeline/README.md` | ~10 min |

## Definitions used in the scripts
* `W8`: closed-form width, Eq. (8); `Wlin`: exact linearized (Fisher-information) width, Eq. (A6). The suffix `A8` in variable names
  (`DLA8`, `predA8`) refers to the same width, numbered Eq. (A8) in an earlier manuscript version.
* `DL8 = f_min/Gamma`, Eq. (9); `DLA8`: operational detection limit f_DL, the f_c at which
  `Wlin` = 1 decade along the fitted R-CPE-L asymptote (root finding over f_c).
* `predFIM`: Wald width of log10(tau) at the full Cole+L fit < 1 decade.
* `obs`: profile-likelihood identifiability (closed and narrower than one decade, nominal threshold).

## Domain check of Algorithm 2
`alg2_domain.py` re-evaluates the existing simulation with the domain check added after the tetrapolar test (refer
when the pilot R-CPE-L exponent lies at a bound): 36 of 240 choices referred (72% of them identifiable); balanced
accuracy of the decided choices 0.89 / 0.88 / 0.86 for priors exact / +-0.5 / +-1 decade (`alg2_domain.json`).
