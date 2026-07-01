"""Peptide collapse, normalization, filtering, imputation, class I/II/III handling."""

from alphaphos.preprocess.anndata import to_anndata
from alphaphos.preprocess.attribution import (
    filter_to_top_n_positions,
    parse_loc_dict,
    parse_precid_phospho_positions,
    top_n_positions,
)
from alphaphos.preprocess.classify import (
    META_COLS,
    apply_condition_aware_classI_mask,
)
from alphaphos.preprocess.collapse import (
    DEFAULT_COLLAPSE_SETTINGS,
    collapse_sites,
    resolve_settings,
)
from alphaphos.preprocess.contaminants import (
    filter_contaminants,
    get_default_contaminants_fasta,
    parse_fasta_accessions,
)
from alphaphos.preprocess.filter import filter_by_completeness
from alphaphos.preprocess.impute import (
    impute_hybrid,
    impute_knn_site_based,
)

__all__ = [
    # Top-N attribution (peptide -> site dedup, Spectronaut over-export fix)
    "filter_to_top_n_positions",
    "parse_loc_dict",
    "parse_precid_phospho_positions",
    "top_n_positions",
    # Contaminant filter (drop trypsin/BSA/keratin PSMs)
    "filter_contaminants",
    "get_default_contaminants_fasta",
    "parse_fasta_accessions",
    # Site-level collapse -- returns a ready-to-use AnnData in one call.
    "collapse_sites",
    "DEFAULT_COLLAPSE_SETTINGS",
    "resolve_settings",
    # Condition-aware Class I masking (legacy shape; used internally by collapse
    # for now, exposed as a public helper for callers with externally-collapsed data).
    "apply_condition_aware_classI_mask",
    "META_COLS",
    # Site-completeness filter (drop sites failing valid-fraction threshold)
    "filter_by_completeness",
    # Imputation (phospho-aware)
    "impute_knn_site_based",  # Dublin-equivalent; legacy-parity path
    "impute_hybrid",  # MAR (site-KNN) + MNAR (downshifted Gaussian) per cell
    # AnnData construction escape hatch -- use collapse_sites in normal workflows.
    "to_anndata",
]
