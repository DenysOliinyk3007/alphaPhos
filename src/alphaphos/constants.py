"""Canonical column, layer, obs, uns, and internal working names.

Single source of truth for every string literal that would otherwise be
repeated across io, preprocess, kinase, and qc modules. Any code that
touches the pipeline data model imports from here rather than hard-coding
the string.

Naming convention:

* ``COL_*``   -- input column names (Spectronaut / DIA-NN etc. schema)
* ``PTM_*``   -- internal working columns used inside the collapse pipeline
* ``VAR_*``   -- AnnData ``var`` columns (public output schema)
* ``OBS_*``   -- AnnData ``obs`` columns (public output schema)
* ``LAYER_*`` -- AnnData ``layers`` keys
* ``UNS_*``   -- AnnData ``uns`` keys
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Spectronaut input columns
# ---------------------------------------------------------------------------

COL_R_FILENAME = "R.FileName"
COL_EG_PRECURSOR_ID = "EG.PrecursorId"
COL_PEP_PEPTIDE_POSITION = "PEP.PeptidePosition"
COL_EG_PTM_ASSAY_PROB = "EG.PTMAssayProbability"
COL_EG_PTM_LOC_PROBS = "EG.PTMLocalizationProbabilities"
COL_EG_IS_DECOY = "EG.IsDecoy"
COL_EG_QVALUE = "EG.Qvalue"
COL_PG_QVALUE = "PG.Qvalue"
COL_PG_GENES = "PG.Genes"
COL_PG_PROTEIN_GROUPS = "PG.ProteinGroups"

# Canonical quant slot the collapse pipeline consumes. Also happens to be
# Spectronaut's "auto" (Settings) quant column name.
COL_CANONICAL_QUANT = "EG.TotalQuantity (Settings)"


# ---------------------------------------------------------------------------
# Internal working columns (pipeline-local, not user-facing)
# ---------------------------------------------------------------------------

PTM_GROUP = "PTM_group"
PTM_BASE_SEQ = "PTM_base_seq"
PTM_POS_VAL = "PTM_0_pos_val"
PTM_NUM = "PTM_0_num"
PTM_AA = "PTM_0_aa"
PTM_LOCALIZATION = "PTM_localization"
PTM_UPD_SEQ = "UPD_seq"
PTM_LOC_DICT = "_loc_dict"


# ---------------------------------------------------------------------------
# AnnData .var columns (public output schema)
# ---------------------------------------------------------------------------

VAR_FULL_KEY = "full_key"
VAR_SHORT_KEY = "short_key"
VAR_PG_KEY = "pg_key"
VAR_PROTEIN_GROUP_ID = "protein_group_id"
VAR_GENE = "gene"
VAR_SITE_AA = "site_aa"
VAR_SITE_POSITION = "site_position"
VAR_ABSOLUTE_POSITION = "absolute_position"
VAR_MULTIPLICITY = "multiplicity"
VAR_UPD_SEQ = "UPD_seq"
VAR_N_SAMPLES_DETECTED = "n_samples_detected"
VAR_MEAN_LOC_PROB = "mean_loc_prob"
VAR_MAX_LOC_PROB = "max_loc_prob"
VAR_MIN_LOC_PROB = "min_loc_prob"
VAR_N_CLASSI_SAMPLES = "n_classI_samples"
VAR_FRACTION_CLASSI = "fraction_classI"


# ---------------------------------------------------------------------------
# AnnData .obs columns (public output schema)
# ---------------------------------------------------------------------------

OBS_SAMPLE = "sample"
OBS_CONDITION = "condition"
OBS_PHOSPHO_SELECTIVITY_PCT = "phospho_selectivity_pct"


# ---------------------------------------------------------------------------
# AnnData .layers keys
# ---------------------------------------------------------------------------

LAYER_INTENSITY_LOG2 = "intensity_log2"
LAYER_LOCALIZATION = "localization"


# ---------------------------------------------------------------------------
# AnnData .uns keys
# ---------------------------------------------------------------------------

UNS_ALPHAPHOS = "alphaphos"
UNS_SOURCE_ATTRS = "source_attrs"
