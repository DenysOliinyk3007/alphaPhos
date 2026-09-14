"""Kinase activity inference — alphaPhos's phospho-specific layer above the
generic alphapepttools downstream stack.

Submodules:

- ``alphaphos.kinase.annotation`` — FASTA-based ±N residue window
  annotation of a collapsed AnnData. Attaches ``adata.var["kinase_sequence"]``,
  which is the input format expected by the Yaffe library scoring and by
  the KSEA workflows in this subpackage. Decoupled from
  :func:`alphaphos.collapse_sites` so non-human data (and any workflow not
  needing kinase-level analysis) doesn't have to touch a FASTA.

- ``alphaphos.kinase.library`` — per-site PWM-based kinase prediction using
  the Yaffe Kinase Library (Johnson et al. *Nature* 2023, Yaromenko et al.
  2024; 311 Ser/Thr + 78 tyrosine PWMs in kinase-library 1.8). Sequence-
  based, predicts upstream kinases for ANY site with a ±7 flanking window
  — including novel sites with no database evidence.  Ranked by percentile
  (default) or raw score.  Requires the optional ``kinase-library`` package
  (see the module docstring for the ``--no-deps`` install recipe).

- ``alphaphos.kinase.enrichment`` — kinase enrichment / KSEA. Three
  statistical frameworks:

  * :func:`kinase_enrichment_from_diffexp` — Fisher's exact per direction;
    classical KSEA from a diff_exp table.
  * :func:`kinase_mea` — GSEA-style weighted K-S on the full ranked list
    (no hard threshold). Most rigorous.
  * :func:`kinase_enrichment_binary` — Fisher's exact, custom
    foreground/background.
"""

from alphaphos.kinase.annotation import (
    add_kinase_windows,
    extract_window,
    load_fasta,
    resolve_fasta_accession,
)
from alphaphos.kinase.enrichment import (
    kinase_enrichment_binary,
    kinase_enrichment_from_diffexp,
    kinase_mea,
)
from alphaphos.kinase.library import (
    predict_kinases,
    score_kinases,
)

__all__ = [
    # FASTA-based kinase-window annotation (adds adata.var["kinase_sequence"])
    "add_kinase_windows",
    "extract_window",
    "load_fasta",
    "resolve_fasta_accession",
    # Per-site PWM prediction
    "predict_kinases",
    "score_kinases",
    # Kinase enrichment (KSEA)
    "kinase_enrichment_from_diffexp",  # Fisher per direction from diff_exp
    "kinase_mea",  # GSEA-style continuous ranking
    "kinase_enrichment_binary",  # Fisher foreground/background
]
