"""Dose-response analysis for phospho data.

Wraps `CurveCurator <https://github.com/kusterlab/curve_curator>`_ (Kuster
lab, TUM) which fits 4-parameter log-logistic curves to dose-response MS
experiments and reports per-curve quality (pEC50, fold change, R²,
target-decoy FDR-adjusted q-value).

CurveCurator is a CLI tool — :func:`fit_dose_response` orchestrates the
TSV + TOML input building, subprocess invocation, and output parsing,
returning a tidy per-curve DataFrame.

For multi-timepoint experiments (one dose-response per timepoint),
CurveCurator is run independently per timepoint and the curves are
concatenated; CurveCurator does NOT fit dose × time jointly. For
single-timepoint experiments, pass ``timepoint_col=None``.
"""

from alphaphos.dose_response.curve_curator import (
    DEFAULT_CURVE_COLS,
    fit_dose_response,
)

__all__ = [
    "fit_dose_response",
    "DEFAULT_CURVE_COLS",
]
