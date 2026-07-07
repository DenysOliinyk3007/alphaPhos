"""Cross-species phosphosite ortholog mapping.

Maps a source-species (e.g., Chinese hamster, mouse, rat) phospho ``AnnData``
to human orthologs by searching the ``±window_size`` sequence context around
each site against a pre-built index of the human phospho-acceptor windows.

The window IS the identity check: a source ``±7`` window that matches a human
protein at position X (with the central residue an S/T/Y) provides
*simultaneous* evidence of protein-level orthology AND site-level position
transfer, without requiring a separate ortholog resolution step.  A shuffled-
per-protein decoy index (same-composition scramble that preserves S/T/Y
positions) is built alongside and used to estimate false-discovery rate in the
proteomics style (Elias & Gygi 2007 adapted to sequence-window matching).

Precedent
---------
This is the same principle used by PhosphoSitePlus for its cross-species site
groups (Hornbeck et al. 2012 *Nucleic Acids Res* 40:D261-D270) and by iPTMnet
(Huang et al. 2018).  We adapt it to a FOSS-reproducible pipeline that ships
with alphaPhos.

Public API
----------
:func:`map_to_human`         -- entry point.  Requires ``adata.var["kinase_sequence"]``
                                (populated by :func:`alphaphos.add_kinase_windows`).
:data:`DEFAULT_ORTHOLOGY_SETTINGS`
:func:`resolve_orthology_settings`
"""

from alphaphos.orthology.mapping import (
    DEFAULT_ORTHOLOGY_SETTINGS,
    map_to_human,
    resolve_orthology_settings,
)

__all__ = [
    "map_to_human",
    "DEFAULT_ORTHOLOGY_SETTINGS",
    "resolve_orthology_settings",
]
