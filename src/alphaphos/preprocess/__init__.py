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
from alphaphos.preprocess.collapse import PeptideCollapse, collapse_sites
from alphaphos.preprocess.contaminants import (
    filter_contaminants,
    get_default_contaminants_fasta,
    parse_fasta_accessions,
)
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
    # Site-level collapse (Hogrebe consolidate ported to Python)
    "collapse_sites",  # one-call function — returns (sites, loc_per_run)
    "PeptideCollapse",  # class form — for fine-grained access to stats etc.
    # Condition-aware Class I masking (per-condition majority rule)
    "apply_condition_aware_classI_mask",
    "META_COLS",
    # Imputation (phospho-aware; complements alphapepttools.pp.impute_*)
    "impute_knn_site_based",  # Dublin-equivalent; legacy-parity path
    "impute_hybrid",  # MAR (site-KNN) + MNAR (downshifted Gaussian) per cell
    # AnnData export (for scverse / alphapepttools downstream)
    "to_anndata",
]
