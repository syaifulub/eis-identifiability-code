# Identifiability-aware EIS equivalent-circuit modelling

**Anonymous research repository | Source code, data and reproducibility results**

This repository accompanies the manuscript **“Identifiability-Aware Parameter Estimation and Frequency-Band Assessment of Equivalent-Circuit Models: Application to In Vivo Tissue Bioimpedance.”** Author information is omitted for double-blind peer review. The material is a reproducibility snapshot, not a claim of independent clinical validation.

The analysis studies **variable projection (VarPro)** for fitting separable electrical equivalent circuits, **profile-likelihood uncertainty calibration**, **practical parameter identifiability**, and **frequency-band assessment** using public in vivo porcine bioimpedance measurements and simulation studies.

> **Review status:** Manuscript table and figure numbering can change across revisions. This repository includes revision analyses and source material from more than one manuscript draft. Use the script/output mapping below (and `docs/README_SOURCE_SNAPSHOT.md`) rather than assuming that an older submitted PDF has the same numbering.

## At a glance

| Item | Details |
| --- | --- |
| Measured EIS dataset | 20 complete spectra from 12 tissue categories, 10 frequencies (10–100 kHz) |
| Repeated measurements | 51 reconstructed sweeps from the provided records |
| Quality check | Kramers–Kronig screening; 12 spectra meet the chosen consistency threshold |
| Circuit models | Cole + L (M10), R–CPE–L (M11), double-Cole + L (M12), and additional screening models |
| Numerical methodology | Original vs. VarPro fitting; matched evaluation budgets and stored run-level outputs |
| Uncertainty | Profile likelihood, parametric bootstrap calibration, and sensitivity analyses |
| Design analysis | Analytical band criterion plus simulated frequency-band experiments |

The above counts describe **this deposited snapshot** and do not imply population-level inference or validity outside the investigated circuits, measurement bands, and error models.

## Repository contents

```text
eis-identifiability-code/
├── README.md                    # This overview and usage instructions
├── GITHUB_UPLOAD_GUIDE_ID.md   # Indonesian GitHub upload instructions
├── LICENSE                      # MIT license for the original code
├── PreparedData.zip             # Original public data (CC BY 4.0; unchanged)
├── requirements.txt             # Broad Python dependency requirements
├── requirements-reported.txt    # Dependency versions reported by the source author
├── run_all.sh                   # Reproduction stages (verify / A / B / C / E / F / rerun)
├── EIS_Complete.ipynb           # Full Jupyter / Colab notebook
├── eisvarpro/                   # Installable package, examples, and nine tests
├── src/                         # Core computations, fit benchmarks, design scripts
│   ├── review/                  # Review analyses and supplementary-table generators
│   └── revision/                # Additional analyses and vector-figure generators
├── data/derived/                # Preprocessed spectra and quality-control summary
├── results/                     # Stored tables, figures, and benchmark outputs
├── manuscript_tables/           # Supplementary LaTeX tables
├── manuscript_figures/          # Figures from earlier/current analysis drafts
├── scripts/validate_repository.py  # Dependency-free integrity and structure checks
├── docs/README_SOURCE_SNAPSHOT.md  # Original README preserved without edits
└── MANIFEST.sha256              # SHA-256 list for all shipped files except itself
```

**Important:** `results/` is intentionally included for direct inspection and comparison; its CSV/JSON outputs are not deleted by `.gitignore`. Raw files are archived in `PreparedData.zip` and extracted into `raw/` when the main run script is used.

## Quick start

Recommended: Python **3.13** in a clean virtual environment. This source snapshot lists Python 3.13 with NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, matplotlib 3.11.2 and scikit-learn 1.9.1 as its reported testing environment. Reproduction may vary across numerical-library versions and operating systems.

```bash
# From the repository root
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ./eisvarpro

# Fast repository integrity check (Python standard library only)
python scripts/validate_repository.py

# Example fitting using the included spectrum
python -m eisvarpro eisvarpro/examples/porcine_bladder_10-100kHz.csv
```

To display JSON in the command-line example, add `--json`. The model package also supports direct Python access:

```python
import pandas as pd
from eisvarpro import assess, format_report

sample = pd.read_csv('eisvarpro/examples/porcine_bladder_10-100kHz.csv')
Z = sample['resistance_ohm'].to_numpy() + 1j * sample['reactance_ohm'].to_numpy()
output = assess(sample['frequency_hz'].to_numpy(), Z)
print(format_report(output))
```

If you need to match the author-reported environment more closely, use `python -m pip install -r requirements-reported.txt` in a fresh Python 3.13 environment. This reported environment has not been independently reproduced during packaging (see `docs/VALIDATION_STATUS.md`).

The scientific routines have multiple internal assumptions; read `eisvarpro/README.md` for parameter definitions and examples, especially for calibrated intervals and measurement-band limits.

### Test and verify

```bash
# Nine package tests (requires numerical packages)
cd eisvarpro
python -m pytest -q tests
cd ..

# Original verification stage (can recompute E13 / E14 tables)
bash run_all.sh verify
```

A local check with SciPy 1.17.0 encountered a native crash in `scipy.optimize.nnls`; numerical tests were **not** completed during packaging. See `docs/VALIDATION_STATUS.md` rather than treating the static verification pass as scientific validation.

The original scripts may take a long time and some stages overwrite files under `results/` or `manuscript_tables/`. **Make a working copy** if you want to preserve the deposited outputs for exact comparison. `verify` expands the data archive into the ignored `raw/` folder. Package tests require `pytest` (`python -m pip install pytest` if missing).

### Reproduce larger analyses

| Purpose | Command | Notes |
| --- | --- | --- |
| Original benchmark and ECM screening | `bash run_all.sh A` | Writes or overwrites original benchmark outputs |
| Variable-projection benchmark | `bash run_all.sh B` | Produces VarPro results and profile analyses |
| Frequency-band experiments | `bash run_all.sh C` | Runs band design, sensitivity and nested-model checks |
| Early review reanalyses | `bash run_all.sh E` | Uses review scripts |
| Further uncertainty and reproducibility analyses | `bash run_all.sh F` | Calculates supplementary analyses |
| Same-environment benchmark rerun | `bash run_all.sh rerun` | Computationally expensive; do not confuse reruns with stored results |

Run stages from the repository root. For detailed dependencies and outputs, see `run_all.sh` and `docs/README_SOURCE_SNAPSHOT.md`. Notebook users can open `EIS_Complete.ipynb` in Jupyter or Colab; the notebook contains earlier numbering and writes its own scripts.

### Additional analyses from `src/revision/`

These scripts reuse the included spectra or results and **do not require new experimental measurements**:

```bash
python src/revision/eA_band_truncation.py
python src/revision/eB_noise_sweep.py
python src/revision/eB2_boundary.py
python src/revision/eC_algorithm2_retro.py
python src/revision/make_fig1_workflow.py
python src/revision/make_vector_figs.py
python src/revision/make_fig5_decision.py
```

The outputs of these scripts are written to `src/revision/out_A/`, `out_B/`, `out_C/` and generated figure folders. The two figure scripts have an optional output-folder argument; see `src/revision/README.txt`.

## Where to find key outputs

| Subject | Source or stored output |
| --- | --- |
| VarPro vs. original fitting | `src/run_vp.py`, `src/analyze_vp.py`; `results/tables/Table_VP_performance.csv` |
| Optimization traces | `results/tables/convergence_traces.csv`, `results/figures_vp/` |
| Circuit-model screening | `results/tables/Table_ECM_screening.csv` |
| Profile-likelihood analysis | `src/run_vp_profiles.py`, `src/review/e1_calibrate.py` |
| Measurement error modelling | `src/review/e1_error_model.py`, `e14_kk_primary.py` |
| Analytical band criterion and simulations | `src/review/e10_asymptotic.py`, `e10b_refined.py`, `e5_band_factorial.py` |
| Repeated-sweep analyses | `src/review/e7_sweep_repeat.py`, `e11_drift.py` |
| Between-recording analysis | `src/review/e12_placement.py` |
| Additional manuscript revision analyses | `src/revision/eA_band_truncation.py`, `eB_noise_sweep.py`, `eC_algorithm2_retro.py` |
| Manuscript workflow and reporting diagrams | `src/revision/make_fig1_workflow.py`, `make_fig5_decision.py` |

For exact supplementary-table script mappings, consult `docs/README_SOURCE_SNAPSHOT.md`, `src/revision/README.txt`, and `manuscript_tables/`. Manuscript labels such as “Table 2”, “Table 3”, and “Table S51” **are version-specific** and should be checked against the paper version submitted to the journal.

## Data provenance, access, and licensing

**Source data:** Falagario, C., De Momi, E., and Mattos, L. S. (2026). *In vivo tissue bioimpedance dataset.* **Scientific Data**. https://doi.org/10.1038/s41597-026-07652-y

**Original data archive:** https://doi.org/10.5281/zenodo.18769577, file `PreparedData.zip`, original data licensed **CC BY 4.0**. The deposited raw ZIP is unchanged; full provenance and original SHA-256 are in `data/README_DATA.md` and `results/reproducibility_manifest.json`. Please cite both the data descriptor and Zenodo record when reusing the recordings.

**Code license:** MIT, in `LICENSE` (also supplied inside `eisvarpro/`). The dataset's Creative Commons license is separate from the code license. The generated tables and figures are included for methodological reproducibility; consult source/data permissions before reuse.

## Scope and limitations

- Data consist of only two animals and a restricted measured band (10–100 kHz). Statistical results on fitting problems should not be interpreted as independent samples from a tissue population.
- All coverage claims depend on the assumed circuit and error model; model misspecification can invalidate nominal or calibrated confidence intervals.
- The proposed band criterion and noise threshold are **model-dependent analytical tools**, not prospectively verified specifications for arbitrary hardware.
- The original revision code, parameter bounds, stored results, and original dataset are preserved; the GitHub packaging adds documentation and static checks only.

## Anonymous peer review

The repository intentionally omits the study authors' identities while the manuscript is under double-blind review. **A public GitHub account, commit metadata, issue profile, or DOI registration can still identify the author.** Before sharing a repository link with reviewers, use an institutionally appropriate blinded route and inspect public profile and Git history. The input data's original authors are properly credited and are **not** the anonymous manuscript authors.

Do not replace the placeholder repository URL in a submitted manuscript until the link works and the review policy permits it. For practical GitHub instructions, see [`GITHUB_UPLOAD_GUIDE_ID.md`](GITHUB_UPLOAD_GUIDE_ID.md).

## Citation

Until the manuscript is published, describe this as an **anonymous companion code and data repository** and cite the underlying public dataset above. After publication, update this README with the final article DOI, names, and repository archive DOI; do not invent a citation in advance.

## Acknowledgement of source snapshot

All originally supplied scientific files are retained byte-for-byte, except this top-level README (its original is preserved in `docs/README_SOURCE_SNAPSHOT.md`) and the regenerated checksum manifest. Packaging has not involved rerunning the full numerical study. See `docs/VALIDATION_STATUS.md` for checks completed and remaining.
