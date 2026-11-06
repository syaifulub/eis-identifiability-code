# Measured-data pipeline

This folder reproduces the analysis of the 20 measured spectra, the inputs to Tables 3 and 4 of the article and to report Tables S1, S13–S15, S26 and S50 of the extended results report
(docs/extended_results.pdf; an earlier manuscript version, whose S-numbers differ from Online Resource 1), from the public dataset. It needs nothing outside this package except the data.

## 1. Data

Falagario et al., *In vivo tissue bioimpedance dataset*, Scientific Data (2026), https://doi.org/10.1038/s41597-026-07652-y.
The data are on Zenodo, record 20408128 (https://zenodo.org/records/20408128), under CC BY 4.0. Download `PreparedData.zip`.
The analysis uses the bipolar frequency-scan recordings of the scan probe: 10 frequencies, 10–100 kHz in 10 kHz steps.

`spectra_manifest.csv` maps each of the 20 analysed spectra to its recording (time stamp) and tissue. The spectra are
keyed by recording and tissue, because two tissues share recording 202309231534. Recordings from 2023 belong to the
first animal and those from 2024 to the second. The manifest also stores the published values of report Tables S1 and S26,
which `compare_to_published.py` uses as a regression check.

## 2. Input format

`run_pipeline.py` reads one tidy CSV with one row per individual reading. The required columns are:

| column | meaning |
|---|---|
| `recording` | recording time stamp as in the manifest (string) |
| `tissue` | tissue name in lower case as in the manifest |
| `frequency_hz` | excitation frequency (Hz) |
| `z_real`, `z_imag` | impedance (Ohm) |
| `order` | acquisition order within the recording (needed to split complete sweeps) |
| `mode` | `scan` for frequency-scan readings (optional; other modes are ignored) |

`prepare_falagario.py` converts the public data directly:

    python prepare_falagario.py <path>/PreparedData readings.csv

It reads the yellow bipolar frequency-scan files of subsets A (first animal) and D (second animal). The acquisition
order is taken from the line order, because the scaled time stamp is not monotone in subset A.

For other datasets, `prepare_readings.py` converts any table of readings into this format. You supply the source column names as options,
so no particular layout of `PreparedData.zip` is assumed. Example:

    python prepare_readings.py PreparedData/scan_readings.csv readings.csv \
        --recording <time-stamp column> --tissue <tissue column> --frequency <frequency column> \
        --magnitude <|Z| column> --phase-deg <phase column> --order <index column> --mode <mode column>

## 3. Run

    pip install numpy scipy pandas
    python run_pipeline.py readings.csv --out results --B 1000 --jobs 4
    python compare_to_published.py results

On one core, each spectrum takes about 1 minute with B = 200 and about 4 minutes with B = 1,000.
The main options are:

* `--psi estimated` (default) estimates the error model from the complete sweeps; `--psi published` uses report Table S46.
* `--kk-threshold 3.0` sets the KK-consistency threshold in percent.
* `--spectra 1,4,7` analyses only the listed spectra.
* `--quick` uses coarse grids and is for testing only.

## 4. Outputs

| file | content | corresponds to |
|---|---|---|
| `spectra_qc.csv` | KK RMS and maximum residual, KK consistency, M11 NRMSE, number of complete sweeps | report Table S1 |
| `error_model.json` | ψ = (σ0, γ, ρf, ρRI, κ) from all sweeps and from the KK-consistent sweeps; published values | Sect. 3.1; report Table S46 |
| `fits.csv` | M10 and M11 parameters, RSS, NRMSE, AICc, BIC | Sect. 3.1; report Table S44 |
| `profiles.csv` | nominal and calibrated profile intervals, critical values, closure, identifiability | Table 3; report Table S15 |
| `criterion.csv` | σ, D, sin ϑ, Γ, closed-form detection limit, **operational f_DL per spectrum** (nominal and calibrated), f̂c, upper bound of fc, widths, predictions (closed form, f_DL, three-way), observed identifiability | Sects. 3.3 and 3.6; report Table S26 |
| `truncation.csv`, `truncation_summary.csv` | per spectrum and sub-band: Γ, closed-form limit, **f_DL**, upper bound of fc, identifiability of τ, Q, α, predictions | Table 4; report Table S50 |
| `run_info.json` | options, versions, run time | |

## 5. Relation to the published analysis

`eis_pipeline.py` is a self-contained implementation of the Methods as written in the manuscript (Sects. 2.1–2.7,
Algorithm 1, Eq. (A6)). Where the text leaves a detail open, the docstring of the function concerned states the
choice made; search for `CHOICE`. These choices are:

* the median of the real and imaginary parts separately;
* the RMS definition of the KK residual;
* AICc with the variance counted as a parameter;
* the degrees-of-freedom rescaling of σ0.

Results can therefore differ slightly from the published tables, and `compare_to_published.py` reports the size of
the differences. Any difference that changes a classification (KK consistency, identifiability, prediction) should be
investigated before the numbers are cited.

## 6. Reproduction on the public data (version 2 of the Zenodo record)

The results of the following commands are in `results_public_v2/`:

    python prepare_falagario.py PreparedData readings.csv
    python run_pipeline.py readings.csv --out results --B 200 --jobs 2 --spectrum-method pooled   # about 11 min
    python compare_to_published.py results

* The pipeline finds the same 20 complete spectra and 51 complete sweeps.
* It reproduces the error model of report Table S46.
* For 11 spectra (1, 4, 7, 8, 13–15, 17–20) the median spectrum is identical, and the published report Tables S1 and S26 are
  reproduced within rounding. The one exception is the upper bound of fc, which differs by at most 0.02 decade
  (the profile-grid resolution).
* The other nine spectra are multi-sweep recordings whose median spectrum is sensitive to a few readings, because the
  per-frequency medians of the real and imaginary parts can switch between drifting sweeps. They differ from the
  published values, and they are not reproduced from the raw files either (see below).
* On the public data:
  * identifiability of log10 τ agrees in 20/20 spectra;
  * the closed-form predictions agree in 20/20;
  * KK consistency agrees in 17/20 (spectra 3, 6 and 11 become KK-consistent).

### Raw files

`raw_to_impedance.py` converts the raw SmartProbe files of `Supplementary.zip` (`OriginalData/SmartProbe`) to
impedance. It uses the gain factors and system phase of the dataset's `Data_Freq_Processing.m`, with the
moving-median outlier removal of `SmartProbe_EBIData_Preparation.m` as an option (`--rmoutliers`):

    python raw_to_impedance.py <path>/OriginalData/SmartProbe readings_raw.csv --rmoutliers

With `--rmoutliers`, it reproduces the same 11 spectra as version 2. Neither variant reproduces the other nine.

### Spectra from per-sweep medians (default)

By default, `run_pipeline.py` builds each spectrum as the per-frequency median of its per-sweep median spectra
(`--spectrum-method sweep_median`). This removes the sensitivity of multi-sweep recordings to individual readings.
`--spectrum-method pooled` gives the construction of the paper. Results on the public data are in
`results_v2_sweepmedian/`, and they are compared with the paper in Online Resource 1, Note S5.1 (Table S7).

### Drift between sweeps

`drift_analysis.py` reads the raw files. In those files the time column is the duration of each reading, so elapsed
time is its cumulative sum. For every recording the script reports:

* sweep durations;
* the drift of |Z| and phase between consecutive sweeps, per second;
* the KK residual of the pooled spectrum, of each sweep and of the per-sweep median.

    python drift_analysis.py <path>/OriginalData/SmartProbe drift.csv

Results: Online Resource 1, Note S5.2 (Table S8).

## 7. Testing without the data

    python make_synthetic_readings.py synthetic_readings.csv
    python run_pipeline.py synthetic_readings.csv --out synthetic_results --B 200 --jobs 2
    python compare_to_published.py synthetic_results

The synthetic readings use the published α, fc and σ of each spectrum, so the comparison only checks that the pipeline
runs and recovers sensible values. They are **not** the measured data. `synthetic_readings.csv` and `example_synthetic_results/` (B = 200, about 10 min with two processes) are included for reference.

## 8. Test on measured wide-band spectra (`wideband_test.py`)

For a prospective test of the band criteria on wide-band (for example tetrapolar, 1 kHz-1 MHz) spectra.
Input CSV: `spectrum_id, f_Hz, re_ohm, im_ohm` (one row per frequency; optional `sample_type, device, electrodes, source`).

```
python -I wideband_test.py spectra.csv results_wideband --jobs 8          # full profiles
python -I wideband_test.py spectra.csv results_wideband --jobs 8 --quick  # coarse profiles (step 0.1 decade)
```

Outputs `spectra.csv` (full-band reference, KK), `truncation.csv` (truncation from below: observed identifiability
of log10 tau, closed form, f_DL, three-way rule, accuracy against the full band), `design.csv` (Algorithm 2 from a
top-decade pilot, priors exact / +-0.5 / +-1 decade) and `summary.json`. Only spectra with a full-band KK RMS
residual below `--kk` (default 3%) enter the summary. `inspect_mat.py file.mat` prints the structure of a MATLAB
file so that a converter to this CSV format can be written. Tested on synthetic spectra (`/tmp` example in the
script header); no measured wide-band data are included.

### 8.1 Measured tetrapolar spectra (Table 6; Online Resource 1, Note S4, Table S6)

```
python -I prepare_dellosa.py registros_EIS.mat wideband_all.csv      # file from Zenodo 10.5281/zenodo.15795175
# split: sample_type == error_free (SFB7, 50 spectra) and source == GonzalezCorrea2018 (Xitron, 36 spectra)
python -I wideband_test.py sfb7.csv   results_sfb7   --jobs 2 --per-decade 10 --fmin-step 0.125
python -I wideband_test.py xitron.csv results_xitron --jobs 2 --per-decade 10 --fmin-step 0.125
```
Run time about 5 min each on two cores. Outputs of the published run: `results_wideband/`.
