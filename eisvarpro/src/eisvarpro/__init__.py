"""eisvarpro: identifiability-aware estimation and frequency-band assessment of Cole-type equivalent circuits.

Implements the methods of Anam, Zarkasi and Rodzi, "Identifiability-Aware Parameter Estimation and Frequency-Band
Assessment of Equivalent-Circuit Models: Application to In Vivo Tissue Bioimpedance":
  estimation.analyze      Algorithm 1 + decision rule (Fig. 5)
  design.screen_bands     Algorithm 2 (band screening from a pilot spectrum, with domain check)
  criteria                Eqs. (8)-(12), W_lin (Eq. A6), f_DL, three-way rule, Wald baseline
  sweeps                  per-sweep median spectrum, drift rate, drift budget
  kk_test, Sigma, estimate_error_model, calibrated_threshold   (Sects. 2.1, 2.3, 2.6)
"""
from ._core import (Spectrum, fit, params, model_Z, profile, interval, kk_test, Sigma, estimate_error_model,
                    calibrated_threshold, PSI_PUBLISHED, BOUNDS, CHI95, Z95)
from . import criteria, design, sweeps, estimation
from .estimation import analyze, fit_models, decision_rule
from .design import screen_bands
__version__ = "1.0.0"
