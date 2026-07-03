"""Gene-level preranked GSEA -- third submodule of alphaphos.enrichment.

Answers the rank-based counterpart of the ORA question:
"reading the full log2FC-ranked list of my genes (no threshold, no
significance call), which biological pathways / GO terms / gene sets
are coherently up- or down-regulated?"

Wraps ``gseapy.prerank`` (Fang et al. 2023) which implements the
Subramanian 2005 running-sum ES with permutation-based FDR.  Uses the
same Enrichr gene-set libraries as :mod:`alphaphos.enrichment.pathway`.
"""

from alphaphos.enrichment.pathway_gsea.gsea import (
    DEFAULT_LIBRARIES_HUMAN,
    DEFAULT_LIBRARIES_MOUSE,
    pathway_gsea,
)

__all__ = [
    "pathway_gsea",
    "DEFAULT_LIBRARIES_HUMAN",
    "DEFAULT_LIBRARIES_MOUSE",
]
