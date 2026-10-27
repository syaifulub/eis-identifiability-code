#!/usr/bin/env bash
# Reproduce the analyses of the manuscript from the command line (alternative to the notebook).
# Usage (from this folder):   bash run_all.sh [stage ...]
#   verify  ~3 min  unit tests of eisvarpro + recomputation of E13/E14 from the stored run-level results
#   A       ~75 min original benchmark, ECM screening, identifiability (Part A; results/tables already contain it)
#   B       ~90 min variable-projection benchmark, profiles, timing (Part B)
#   C       ~40 min band design, sensitivity, nested model, feature stability (Part C)
#   E       ~10 min first-round reanalyses (Part E)
#   F       ~90 min second- and third-round analyses E1-E14, supplementary tables and figures (Part F)
#   rerun   ~3 h    complete Table 2 benchmark rerun in the current environment and same-machine timing
# Default: verify.  Stages B-F overwrite files in results/; keep a copy if you want to compare.
set -euo pipefail
PY=${PYTHON:-python3}
cd "$(dirname "$0")"
ROOT=$(pwd); RAW=$ROOT/raw/PreparedData; RUNS=25
export EIS_RAW_ZIP=$ROOT/PreparedData.zip PYTHONWARNINGS=ignore
[ -d "$RAW" ] || $PY -c "import zipfile; zipfile.ZipFile('PreparedData.zip').extractall('raw')"
mkdir -p results/tables results/figures results/figures_vp manuscript_tables
stage() { echo; echo "=== stage $1 ($(date +%H:%M:%S)) ==="; }

run_verify() {
  stage verify
  (cd eisvarpro && $PY -c "
import sys, time; sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import test_eisvarpro as t
for n in sorted(x for x in dir(t) if x.startswith('test_')): getattr(t, n)(); print('ok', n)")
  (cd src/review && $PY e13_stats.py > ../../results/log_e13.txt && $PY make_supp_e13.py ../../manuscript_tables/supp_e13.tex)
  (cd src/review && $PY e14_kk_primary.py "$RAW" > ../../results/log_e14.txt && $PY make_supp_e14.py ../../manuscript_tables/supp_e14.tex)
  echo "verify finished: compare manuscript_tables/supp_e13.tex and supp_e14.tex with Online Resource 1 (Tables S38-S49)"
}
run_A() {
  stage A
  $PY src/run_experiment.py
  (cd src && $PY run_m12.py && $PY run_rev3.py)
  $PY src/make_figures.py || true
}
run_B() {
  stage B
  (cd src && $PY run_vp.py M11_RCPEL $RUNS && $PY run_vp.py M10_ColeL $RUNS && $PY run_vp.py M12_DoubleColeL $RUNS)
  (cd src && $PY analyze_vp.py > ../results/log_analyze_vp.txt && $PY run_vp_profiles.py && $PY time_fair.py 25)
}
run_C() {
  stage C
  (cd src && $PY band_design.py 30 > ../results/log_band.txt && $PY make_fig_band.py && $PY sensitivity.py)
  (cd src && $PY nested_vp.py 25 > ../results/log_nested.txt && $PY feature_stability.py && $PY make_fig_stability.py)
}
run_E() {
  stage E
  (cd src/review && $PY review_fast.py && $PY review_slow.py R13 && $PY review_slow.py R5 && $PY review_slow.py R12 && $PY review_slow.py R2)
}
run_F() {
  stage F
  cd src/review
  $PY e1_error_model.py "$RAW" && $PY e1b_bias.py && $PY e2_hf.py
  $PY e1_calibrate.py M11 && $PY e1_calibrate.py M10 && $PY e1_calibrate.py DB
  $PY e3_model_selection.py && $PY e4_thresholds.py && $PY e7_sweep_repeat.py "$RAW"
  $PY e5_band_factorial.py 20 && $PY e6_mechanism.py && $PY r2_figs.py
  $PY e8_errcheck.py "$RAW" && $PY e9_summary_ci.py
  $PY e10_asymptotic.py && $PY e10b_refined.py && $PY e10_fig.py && $PY e11_drift.py "$RAW" && $PY e11b_drift_ci.py
  $PY e12_placement.py "$RAW" && $PY e13_stats.py && $PY e14_kk_primary.py "$RAW"
  $PY make_supp_round2.py ../../manuscript_tables/supp_round2.tex && $PY make_supp_refined.py ../../manuscript_tables/supp_refined.tex
  $PY make_supp_e12.py ../../manuscript_tables/supp_e12.tex && $PY make_supp_e13.py ../../manuscript_tables/supp_e13.tex
  $PY make_supp_e14.py ../../manuscript_tables/supp_e14.tex
  cd ../..
  (cd src && $PY make_fig1_workflow_v3.py && $PY make_fig_decision_v3.py)
}
run_rerun() {
  stage rerun
  (cd src && $PY rerun_same_env.py M10_ColeL && $PY rerun_same_env.py M11_RCPEL && $PY rerun_same_env.py M12_DoubleColeL && $PY time_fair.py 25)
  (cd src/review && $PY make_supp_rerun.py ../../manuscript_tables/supp_moved.tex ../../manuscript_tables/supp_rerun.tex)
}
STAGES=${*:-verify}
for s in $STAGES; do run_$s; done
echo; echo "done ($(date +%H:%M:%S))"
