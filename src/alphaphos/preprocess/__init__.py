"""Peptide collapse, normalization, filtering, imputation, class I/II/III handling."""

from alphaphos.io.contaminants import (
    filter_contaminants,
    get_default_contaminants_fasta,
    parse_fasta_accessions,
)
from alphaphos.preprocess.anndata import to_anndata
from alphaphos.preprocess.attribution import (
    filter_to_top_n_positions,
    parse_loc_dict,
    parse_precid_phospho_positions,
    top_n_positions,
)
from alphaphos.preprocess.classify import apply_condition_aware_classI_mask
from alphaphos.preprocess.collapse import (
    DEFAULT_COLLAPSE_SETTINGS,
    collapse_sites,
    resolve_settings,
)
from alphaphos.preprocess.collapse_precursors import (
    DEFAULT_PRECURSOR_COLLAPSE_SETTINGS,
    aggregate_to_site_level,
    collapse_precursors,
    precursor_to_site_view,
    resolve_precursor_settings,
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
    # Precursor-level collapse -- sibling of collapse_sites, no residue
    # attribution / no localization masking.  Used when localization is
    # unreliable on low-abundance features and detection matters more than
    # residue resolution.
    "collapse_precursors",
    "DEFAULT_PRECURSOR_COLLAPSE_SETTINGS",
    "resolve_precursor_settings",
    "precursor_to_site_view",
    "aggregate_to_site_level",
    # Condition-aware Class I masking -- DEPRECATED shim over
    # _collapse.masking.mask_condition_aware; use collapse_sites instead.
    "apply_condition_aware_classI_mask",
    # Site-completeness filter (drop sites failing valid-fraction threshold)
    "filter_by_completeness",
    # Imputation (phospho-aware)
    "impute_knn_site_based",  # legacy-parity KNN; prefer impute_hybrid for new work
    "impute_hybrid",  # MAR (site-KNN) + MNAR (downshifted Gaussian) per cell
    # AnnData construction escape hatch for externally collapsed (sites x samples)
    # matrices with alphaPhos `Protein|Gene|Site|Mult` keys -- use collapse_sites
    # in normal workflows.
    "to_anndata",
]
