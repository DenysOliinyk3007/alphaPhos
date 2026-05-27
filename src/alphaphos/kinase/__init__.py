"""Kinase activity inference — alphaPhos's phospho-specific layer above the
generic alphapepttools downstream stack.

Submodules:

- ``alphaphos.kinase.library`` — per-site PWM-based kinase prediction using
  the Yaffe Kinase Library (Johnson et al. *Nature* 2023; 311 Ser/Thr + 78
  tyrosine kinases). Sequence-based, predicts upstream kinase for ANY
  site that has a ±7 flanking window — including novel sites with no
  database evidence.

- ``alphaphos.kinase.enrichment`` — kinase enrichment / KSEA. Three
  statistical frameworks:

  * :func:`kinase_enrichment_from_diffexp` — Fisher's exact per direction;
    classical KSEA from a diff_exp table.
  * :func:`kinase_mea` — GSEA-style weighted K-S on the full ranked list
    (no hard threshold). Most rigorous.
  * :func:`kinase_enrichment_binary` — Fisher's exact, custom
    foreground/background.
"""

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
    # Per-site PWM prediction
    "predict_kinases",
    "score_kinases",
    # Kinase enrichment (KSEA)
    "kinase_enrichment_from_diffexp",  # Fisher per direction from diff_exp
    "kinase_mea",  # GSEA-style continuous ranking
    "kinase_enrichment_binary",  # Fisher foreground/background
]
