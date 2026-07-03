"""Site-set enrichment for phosphoproteomics.

Answers site-level *functional* enrichment questions:

* Is my hit set enriched for **activating** vs **inhibitory**
  phosphosites?
* Is it enriched for sites that **disrupt / induce** protein-protein
  interactions?
* Is it enriched for phosphosites **mutated in cancer** (ClinVar /
  TCGA)?
* Is it enriched for sites with **high intrinsic functional scores**
  (Ochoa 2020)?

Backed by an integrated PTM functional database (8 sources merged,
504k phospho relations, 235k unique phosphosites).

Kinase-substrate inference lives in
:func:`alphaphos.enrichment.kinase_activity` (decoupler ULM on the
OmniPath or PTM-DB network); gene-level pathway ORA lives in
:func:`alphaphos.enrichment.pathway_enrichment` (gseapy Enrichr against
GO / KEGG / Reactome / Hallmark libraries); gene-level preranked GSEA
lives in :func:`alphaphos.enrichment.pathway_gsea` (rank-based
counterpart of ``pathway_enrichment``); sequence-based per-site kinase
prediction lives in :mod:`alphaphos.kinase.library` (Yaffe PWM). The
five cover different scientific questions and are intentionally kept
separate.

Public API (v1, Phase 1):

* :func:`load_ptm_db` -- load the DB parquet.
* :func:`emit_libraries` -- generate GMT-format site-set files.
* :func:`load_gmt` -- read a GMT file back into a Python dict.

The matching engine, ORA / GSEA scoring, and validation harness land
in later phases.
"""

from alphaphos.enrichment.db import (
    DEFAULT_DB_PATH,
    TEST_FIXTURE_PATH,
    load_ptm_db,
    site_id,
)
from alphaphos.enrichment.enrich import gsea, ora
from alphaphos.enrichment.ksea import (
    fetch_omnipath_ks_network,
    kinase_activity,
    load_ptm_ks_network,
)
from alphaphos.enrichment.libraries import emit_libraries, load_gmt, load_libraries
from alphaphos.enrichment.matching import (
    MatchResult,
    attach_site_ids,
    match_sites,
    parse_alphaphos_key,
)
from alphaphos.enrichment.pathway import (
    DEFAULT_LIBRARIES_HUMAN,
    DEFAULT_LIBRARIES_MOUSE,
    pathway_enrichment,
)
from alphaphos.enrichment.pathway_gsea import pathway_gsea
from alphaphos.enrichment.validation import (
    build_kinase_substrate_library,
    ev3_expectations_for_condition,
    load_ev2,
    load_ev3,
    score_against_ev3,
)

__all__ = [
    "load_ptm_db",
    "site_id",
    "emit_libraries",
    "load_gmt",
    "load_libraries",
    "match_sites",
    "attach_site_ids",
    "parse_alphaphos_key",
    "MatchResult",
    "ora",
    "gsea",
    "kinase_activity",
    "fetch_omnipath_ks_network",
    "load_ptm_ks_network",
    "pathway_enrichment",
    "pathway_gsea",
    "DEFAULT_LIBRARIES_HUMAN",
    "DEFAULT_LIBRARIES_MOUSE",
    "build_kinase_substrate_library",
    "load_ev3",
    "load_ev2",
    "ev3_expectations_for_condition",
    "score_against_ev3",
    "DEFAULT_DB_PATH",
    "TEST_FIXTURE_PATH",
]
