"""alphaPhos signalome: module detection + kinase network on phospho data.

Signalome analysis extracts a network / module-level view of the phospho
signaling landscape, complementing site-level (limma / ANOVA), kinase-level
(KSEA), and pathway-level (Enrichr) tools that alphaPhos already ships:

- **Modules**: data-driven groups of coordinately-regulated sites, discovered
  from the per-site kinase-prediction profile via Ward hierarchical
  clustering.
- **Site -> module attribution**: each site (and each protein) is assigned
  to a module along with its top predicted kinase.
- **Module x kinase table**: percent shares showing which kinases explain
  which modules.
- **Kinase-kinase network**: edges between kinases whose per-site
  prediction profiles are highly correlated (they co-regulate similar
  sites).
- **Expanded view**: denormalized per-focal-kinase table for graph /
  visualisation exports.

Clean-room re-implementation of the signalome analysis in PhosPy
(github.com/falconsmilie/phospy), which itself is a Python port of PhosR
(Kim et al. 2021, *Cell Reports Methods*).  PhosPy is GPL-3.0 and is used
here only as a numerical oracle in parity tests, never as a source.

Single-call entry point
-----------------------

::

    import alphaphos as ap

    # Given an AnnData that has run through ap.score_kinases().  Build the
    # [0, 1] input (Yaffe percentile / 100, kinases with confident substrates
    # among the sites of interest) and run:
    matrix = ap.signalome.prediction_matrix_from_adata(adata, sites=significant_sites)
    result = ap.signalome.build_signalome(
        matrix,
        substrate_support_cutoff=0.9,             # top-10 % percentile = substrate
        kinase_substrates=my_reference_network,   # optional {kinase: [substrate_site_ids]}
    )
    result.site_assignments   # DataFrame -- per-site module_id + top_kinase + ...
    result.module_table       # DataFrame -- module x kinase % shares
    result.network.edges      # DataFrame -- kinase-kinase edges
    result.network.nodes      # DataFrame -- degree + n_substrates per kinase
    result.expanded           # DataFrame -- denormalised per-focal-kinase view
    result.provenance         # dict     -- settings + counts + versions
"""

from __future__ import annotations

from alphaphos.signalome._orchestrator import (
    SignalomeResult,
    build_signalome,
    prediction_matrix_from_adata,
)
from alphaphos.signalome.assignments import (
    build_module_assignments,
    select_kinase_substrates,
)
from alphaphos.signalome.clustering import (
    ClusterCandidateScore,
    ModuleCountSelectionResult,
    SignalomeClusteringResult,
    build_ward_tree,
    cluster_sites,
    cut_labels,
    precondition_scores,
    resolve_scoring_mode,
    select_module_count,
    summarize_profile_degeneracy,
)
from alphaphos.signalome.expanded import build_expanded_table
from alphaphos.signalome.modules import build_module_table
from alphaphos.signalome.network import KinaseNetwork, build_kinase_network
from alphaphos.signalome.protein_resolution import (
    derive_protein_modules,
    extract_site_metadata,
    extract_site_to_protein,
)

__all__ = [
    # High-level entry point
    "build_signalome",
    "prediction_matrix_from_adata",
    "SignalomeResult",
    # Per-stage functions
    "cluster_sites",
    "derive_protein_modules",
    "build_module_assignments",
    "build_module_table",
    "build_kinase_network",
    "build_expanded_table",
    # Helpers
    "extract_site_metadata",
    "extract_site_to_protein",
    "select_kinase_substrates",
    # Lower-level algorithms (for advanced use / testing)
    "build_ward_tree",
    "cut_labels",
    "precondition_scores",
    "resolve_scoring_mode",
    "select_module_count",
    "summarize_profile_degeneracy",
    # Result / diagnostic dataclasses
    "ClusterCandidateScore",
    "KinaseNetwork",
    "ModuleCountSelectionResult",
    "SignalomeClusteringResult",
]
