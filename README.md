# EIS Identifiability: Reproducible Research Code

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green)](eisvarpro/LICENSE)

**Research software and reproducibility materials** for:

> S. Anam, A. Zarkasi, and Z. Md Rodzi, *Identifiability-Aware Frequency-Band Assessment for Impedance Measurement Systems: Closed-Form Criteria and Validation on Tissue, Battery and Tetrapolar Spectra*. Iranian Journal of Science and Technology, Transactions of Electrical Engineering (manuscript submitted; publication details to be updated).

This repository contains the `eisvarpro` Python package, scripts and selected saved outputs for synthetic and measured-data experiments, and TikZ figure sources.

## Quick start

Requirements: **Python 3.10+**. From the repository root:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e "./eisvarpro[test]"
python -m pytest eisvarpro/tests -q
python eisvarpro/examples/run_examples.py
python -m eisvarpro.cli analyze eisvarpro/examples/synthetic_narrow_band.csv
```

For full reproduction scripts, install the additional analysis packages:

```bash
python -m pip install -r requirements-reproduction.txt
```

Reproduction scripts use saved, fixed random seeds, but exact numerical values may vary with versions and numerical libraries. Computationally intensive experiments can take minutes or longer.

## Repository organization

| Path | Purpose |
| --- | --- |
| [`eisvarpro/`](eisvarpro/) | Installable package, tests, examples, and detailed API/CLI instructions |
| [`code/`](code/) | Synthetic experiments, battery study and saved JSON outputs |
| [`code/pipeline/`](code/pipeline/) | Measured tissue spectra, dataset preparation, and comparison pipelines |
| [`figsrc/`](figsrc/) | LaTeX/TikZ sources for selected figures |
| [`docs/`](docs/) | Extended results report of an earlier manuscript version and the original archive README |
| [`CITATION.cff`](CITATION.cff) | Machine-readable software citation |

## How to reproduce manuscript outputs

Numbering follows the submitted article: main text (Tables 1-7, Figs. 1-6, Appendix A) and its Online Resource 1
(Supplementary Notes S1-S5, Tables S1-S8). `docs/extended_results.pdf` is a longer results report from an earlier
version of the manuscript; its tables are cited below as *report Table Sxx*, and their numbers differ from those of
Online Resource 1.

The analysis has multiple independent pipelines. Start with the detailed instructions and command-to-table maps in [`code/README.md`](code/README.md) and [`code/pipeline/README.md`](code/pipeline/README.md); these specify dataset prerequisites and long-running commands.

Examples of self-contained commands (from `code/`):

```bash
cd code
python verify_kappa_decomposition.py
python verify_criteria.py verify_criteria_output.json
```

**Caution:** Some scripts overwrite output JSON/CSV files included for reference. Run them in a clean worktree or commit a baseline before regeneration. Not every pipeline runs without externally downloaded research data.

## Data provenance and licensing

Third-party research datasets are **not bundled in full**. Follow the linked dataset references and acquisition instructions in the package and pipeline READMEs:

- Falagario, De Momi, Mattos, *In vivo tissue bioimpedance dataset* (Scientific Data, 2026), Zenodo record **20408128**.
- Zhang et al., *Nature Communications* **11**, 1706 (2020), Zenodo DOI **10.5281/zenodo.3633835**.
- Dell'Osa, Bravo, Villanueva Jousset, *Single dispersion EIS measurements: a compilation*, Zenodo DOI **10.5281/zenodo.15795175**.

Example and derived data may be subject to separate third-party licenses; consult the source records before reuse. Included `synthetic_*` data are for demonstration, not patient records.

## Citation

Use GitHub's **Cite this repository** button (powered by [`CITATION.cff`](CITATION.cff)). Replace the provisional manuscript status with a DOI and final journal citation when available.

## License

The Python code is provided under the **MIT License** in [`eisvarpro/LICENSE`](eisvarpro/LICENSE). Licensing of externally sourced data remains with its respective owners.

## Contact

Syaiful Anam — **syaiful@ub.ac.id**

## Additional documentation

The original archive README is preserved at [`docs/ORIGINAL_README.md`](docs/ORIGINAL_README.md).
