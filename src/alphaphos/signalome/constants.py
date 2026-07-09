"""Column names + policy tokens used across the signalome subpackage.

Kept as a separate module so downstream code (assignments, modules, network,
expanded) references the same string constants and no rename can silently
break a join or a schema check.  Names align with the PhosPy conventions
where practical to make the numerical parity tests straightforward.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Column names -- kept identical to PhosPy where the schema is user-visible,
# to make parity comparisons straightforward.
# ---------------------------------------------------------------------------

SITE_KEY_COLUMN = "site_key"
DISPLAY_ID_COLUMN = "display_id"
GENE_SYMBOL_COLUMN = "gene_symbol"
SITE_COLUMN = "site"
PROTEIN_COLUMN = "protein"
PROTEIN_ACCESSION_COLUMN = "protein_accession"
ISOFORM_ID_COLUMN = "isoform_id"
MODULE_ID_COLUMN = "module_id"

KINASE_COLUMN = "kinase"
SOURCE_KINASE_COLUMN = "source_kinase"
TARGET_KINASE_COLUMN = "target_kinase"
CORRELATION_COLUMN = "correlation"
DEGREE_COLUMN = "degree"
N_SUBSTRATES_COLUMN = "n_substrates"

TOP_KINASE_COLUMN = "top_kinase"
TOP_SCORE_COLUMN = "top_score"
TOP_KINASE_CANDIDATES_COLUMN = "top_kinase_candidates"
TOP_KINASE_WEIGHTS_COLUMN = "top_kinase_weights"
TOP_KINASE_TIE_COUNT_COLUMN = "top_kinase_tie_count"
TOP_KINASE_IS_AMBIGUOUS_COLUMN = "top_kinase_is_ambiguous"
TOP_KINASE_SELECTION_POLICY_COLUMN = "top_kinase_selection_policy"

MODULE_TOP_KINASE_COLUMN = "module_top_kinase"
MODULE_TOP_KINASE_CANDIDATES_COLUMN = "module_top_kinase_candidates"
MODULE_TOP_KINASE_TIE_COUNT_COLUMN = "module_top_kinase_tie_count"
MODULE_TOP_KINASE_IS_AMBIGUOUS_COLUMN = "module_top_kinase_is_ambiguous"
MODULE_TOP_KINASE_SELECTION_POLICY_COLUMN = "module_top_kinase_selection_policy"

SITE_CLUSTER_COLUMN = "site_cluster"

# ---------------------------------------------------------------------------
# Policy tokens
# ---------------------------------------------------------------------------

UNSUPPORTED_KINASE = "unsupported"
LEXICOGRAPHIC_TIE_BREAK_POLICY = "lexicographic"
NO_SUPPORT_SELECTION_POLICY = "no_support"

NETWORK_POLICY_SIGNED = "signed"
NETWORK_POLICY_POSITIVE_ONLY = "positive_only"
NETWORK_POLICY_ABSOLUTE = "absolute"
NETWORK_POLICIES = (
    NETWORK_POLICY_SIGNED,
    NETWORK_POLICY_POSITIVE_ONLY,
    NETWORK_POLICY_ABSOLUTE,
)

SCORING_MODE_EXACT = "exact"
SCORING_MODE_SAMPLED = "sampled"
SCORING_MODE_AUTO = "auto"
SCORING_MODES = (SCORING_MODE_EXACT, SCORING_MODE_SAMPLED, SCORING_MODE_AUTO)

# ---------------------------------------------------------------------------
# Defaults -- match PhosPy where the intent is bit-exact numerical parity.
# ---------------------------------------------------------------------------

DEFAULT_MAX_MODULES = 10
DEFAULT_PRIMARY_THRESHOLD = 0.5
DEFAULT_FALLBACK_THRESHOLD = 0.1
DEFAULT_MAX_EXACT_SITES = 5000
DEFAULT_MAX_APPROX_SAMPLES_PER_CLUSTER = 200
DEFAULT_NETWORK_CORRELATION_THRESHOLD = 0.5
DEFAULT_SUBSTRATE_SUPPORT_CUTOFF = 0.5
NEAR_CONSTANT_VARIANCE_TOLERANCE = 1e-12
