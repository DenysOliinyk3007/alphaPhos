"""Phospho-specific QC dashboard (HTML + bokeh).

Composable per-metric functions plus a one-call
:func:`generate_dashboard` that builds a single HTML file with all
panels. Phospho-aware: S/T/Y stratification, localization-probability
distribution, multiplicity composition, Class I per-cell vs
condition-aware comparison, contaminant breakdown, hybrid-imputation
MAR/MNAR split, pipeline waterfall, replicate correlation matrix.
"""

from alphaphos.qc.dashboard import generate_dashboard
from alphaphos.qc.metrics import (
    compute_classI_comparison,
    compute_contaminant_breakdown,
    compute_imputation_summary,
    compute_localization_distribution,
    compute_missingness,
    compute_multiplicity_distribution,
    compute_n_classI_per_sample,
    compute_per_condition_cv_by_aa,
    compute_pipeline_waterfall,
    compute_replicate_correlation,
    compute_sample_dendrogram_order,
    compute_sty_ratio,
)

__all__ = [
    "generate_dashboard",
    # metrics (importable for programmatic access without rendering)
    "compute_pipeline_waterfall",
    "compute_sty_ratio",
    "compute_localization_distribution",
    "compute_multiplicity_distribution",
    "compute_missingness",
    "compute_n_classI_per_sample",
    "compute_per_condition_cv_by_aa",
    "compute_replicate_correlation",
    "compute_sample_dendrogram_order",
    "compute_classI_comparison",
    "compute_contaminant_breakdown",
    "compute_imputation_summary",
]
