"""alphaPhos -- phosphoproteomics analysis toolkit.

Top-level re-exports for the most common workflow.

Typical usage::

    import alphaphos as ap
    import pandas as pd

    # 1) Read a Spectronaut PSM report
    psm_df = ap.read_spectronaut("report.parquet")

    # 2) Sample metadata (must contain 'sample' and 'condition' columns)
    condition_df = pd.read_csv("conditions.csv")

    # 3) Collapse to a site-level AnnData
    adata = ap.collapse_sites(psm_df, condition_df=condition_df)

    # 4) (Optional) attach kinase windows -- required only for kinase enrichment
    adata = ap.add_kinase_windows(adata, fasta_path="human.fasta")

    # 5) QC dashboard
    ap.generate_dashboard(adata, "qc.html")

See :func:`collapse_sites` for the settings dict; see
:mod:`alphaphos.preprocess`, :mod:`alphaphos.kinase`, :mod:`alphaphos.qc`,
:mod:`alphaphos.ksea`, :mod:`alphaphos.dose_response`, and :mod:`alphaphos.io`
for domain-specific tooling.
"""

__version__ = "0.6.1"

# Public entry points, re-exported for convenience.
from alphaphos.io.diann import (
    DEFAULT_DIANN_IO_SETTINGS,
    resolve_diann_io_settings,
)
from alphaphos.io.diann import (
    read_psm as read_diann,
)
from alphaphos.io.fragpipe import (
    DEFAULT_FRAGPIPE_IO_SETTINGS,
    read_fragpipe_sites,
    resolve_fragpipe_io_settings,
)
from alphaphos.io.spectronaut import (
    DEFAULT_IO_SETTINGS,
    resolve_io_settings,
)
from alphaphos.io.spectronaut import (
    read_psm as read_spectronaut,
)
from alphaphos.kinase.annotation import add_kinase_windows, load_fasta
from alphaphos.preprocess.anndata import to_anndata
from alphaphos.preprocess.batch_correct import (
    DEFAULT_COMBAT_SETTINGS,
    batch_correct_combat,
)
from alphaphos.preprocess.collapse import (
    DEFAULT_COLLAPSE_SETTINGS,
    collapse_sites,
    resolve_settings,
)
from alphaphos.preprocess.filter import filter_by_completeness
from alphaphos.preprocess.impute import impute_hybrid, impute_knn_site_based
from alphaphos.stats.diff_exp import DEFAULT_STATS_SETTINGS, diff_exp_limma


# QC dashboard: only imported lazily to avoid pulling bokeh at package import.
def generate_dashboard(*args, **kwargs):  # pragma: no cover
    """See :func:`alphaphos.qc.generate_dashboard`."""
    from alphaphos.qc import generate_dashboard as _gd

    return _gd(*args, **kwargs)


__all__ = [
    "__version__",
    "read_spectronaut",
    "read_diann",
    "read_fragpipe_sites",
    "DEFAULT_IO_SETTINGS",
    "DEFAULT_DIANN_IO_SETTINGS",
    "DEFAULT_FRAGPIPE_IO_SETTINGS",
    "resolve_io_settings",
    "resolve_diann_io_settings",
    "resolve_fragpipe_io_settings",
    "collapse_sites",
    "DEFAULT_COLLAPSE_SETTINGS",
    "resolve_settings",
    "filter_by_completeness",
    "impute_hybrid",
    "impute_knn_site_based",
    "batch_correct_combat",
    "DEFAULT_COMBAT_SETTINGS",
    "diff_exp_limma",
    "DEFAULT_STATS_SETTINGS",
    "add_kinase_windows",
    "load_fasta",
    "to_anndata",
    "generate_dashboard",
]
