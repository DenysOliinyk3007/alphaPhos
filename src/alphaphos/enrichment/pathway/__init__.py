"""Gene-level pathway ORA -- second submodule of alphaphos.enrichment.

Answers the classical question: "given my significant phosphosites,
which biological pathways / GO terms / gene sets are over-represented
in the underlying gene set, relative to my study's own background?"

Wraps ``gseapy.enrichr`` (Fang et al. 2023) to query the Enrichr
gene-set libraries.  The scientifically-critical piece is the
**background**: for a phospho experiment, using genome-wide background
overstates enrichment because your detection universe is much smaller
than the genome.  Defaults enforce a phosphoproteome (all sites tested)
background; passing a proteome gene list is preferred when available.
"""

from alphaphos.enrichment.pathway.enrichment import (
    DEFAULT_LIBRARIES_HUMAN,
    DEFAULT_LIBRARIES_MOUSE,
    pathway_enrichment,
)

__all__ = [
    "pathway_enrichment",
    "DEFAULT_LIBRARIES_HUMAN",
    "DEFAULT_LIBRARIES_MOUSE",
]
