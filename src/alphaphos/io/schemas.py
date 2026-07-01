"""Column-name schemas per search engine.

This module is the single source of truth for "which columns does search
engine X export, and what do they mean?". Both the IO readers (which need
to know what to load) and the collapse pipeline (which needs to know
what to consume) reference these dictionaries.

Currently only Spectronaut (``"SN"``) is filled in. When adding DIA-NN /
FragPipe / PEAKS, follow the same shape: three sub-dicts per engine
(REQUIRED / OPTIONAL / QUANT_CANDIDATES), one FALLBACK_CHAINS entry per
requested quantification level.

Column names use the dotted Spectronaut convention throughout (``R.FileName``,
not ``R_FileName``). The reader normalizes underscore-form to dotted-form
so downstream code sees only dotted names.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# What the collapse pipeline MUST have to run (hard error if missing).
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "SN": (
        "R.FileName",
        "EG.PrecursorId",
        "PEP.PeptidePosition",
        "EG.PTMAssayProbability",
        "PG.Genes",
        "PG.ProteinGroups",
    ),
}


# ---------------------------------------------------------------------------
# Columns the collapse pipeline CAN use when present (better output).
# ---------------------------------------------------------------------------

OPTIONAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "SN": (
        "EG.PTMLocalizationProbabilities",  # per-position loc probs (much better than joint)
        "EG.IsDecoy",  # for drop_decoys filter
        "EG.Qvalue",  # for eg_qvalue_max filter
        "PG.Qvalue",  # for pg_qvalue_max filter
    ),
}


# ---------------------------------------------------------------------------
# Quantification column candidates.
#
# Structure: {engine: {level: (col1, col2, ...)}}.
# Columns are ordered by preference within a level; the reader picks the
# first one present.
#
# "auto" means "whatever Spectronaut was configured for" -- this is the
# ``(Settings)``-suffixed column that Spectronaut chose during search
# and is safe as a broadly-applicable fallback.
# ---------------------------------------------------------------------------

QUANT_COLUMN_CANDIDATES: dict[str, dict[str, tuple[str, ...]]] = {
    "SN": {
        "auto": (
            "EG.TotalQuantity (Settings)",
            "FG.Quantity",
        ),
        "MS1": (
            "FG.MS1Quantity",
            "FG.MS1RawQuantity",
            "EG.MS1Quantity",
            "EG.RawIntensityMS1",
        ),
        "MS2": (
            "FG.MS2Quantity",
            "FG.MS2RawQuantity",
            "EG.MS2Quantity",
            "EG.RawIntensityMS2",
        ),
    },
}


# ---------------------------------------------------------------------------
# Fallback chain per requested quantification level.
#
# If the user's chosen level has no columns in the report, we walk the
# chain and pick the first available. A warning is issued so the fallback
# is visible in the log and in ``adata.uns["alphaphos"]["stats"]``.
#
# The MS2 default has the deepest chain because MS2 is the most common
# real-world request but the least universally exported.
# ---------------------------------------------------------------------------

FALLBACK_CHAINS: dict[str, tuple[str, ...]] = {
    "MS2": ("MS2", "MS1", "auto"),
    "MS1": ("MS1", "auto"),
    "auto": ("auto",),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def all_needed_columns(engine: str) -> frozenset[str]:
    """Return the full set of columns the reader should try to load.

    Union of REQUIRED, OPTIONAL, and every quant candidate across every
    level. The reader intersects this with what the file actually
    contains -- extras are ignored.

    Raises ``KeyError`` if the engine isn't recognized.
    """
    if engine not in REQUIRED_COLUMNS:
        raise KeyError(f"Unknown engine {engine!r}. Available: {sorted(REQUIRED_COLUMNS)}")
    needed: set[str] = set(REQUIRED_COLUMNS[engine])
    needed.update(OPTIONAL_COLUMNS.get(engine, ()))
    for level_cols in QUANT_COLUMN_CANDIDATES.get(engine, {}).values():
        needed.update(level_cols)
    return frozenset(needed)


def resolve_quant_column(
    available_columns: set[str],
    *,
    engine: str,
    requested_level: str,
) -> tuple[str, str]:
    """Pick the best quant column, walking the fallback chain if needed.

    Parameters
    ----------
    available_columns : set[str]
        Columns actually present in the DataFrame (post-normalization to
        dotted form).
    engine : str
        Search engine id (e.g. ``"SN"``).
    requested_level : str
        Level the caller asked for. Must be a key of :data:`FALLBACK_CHAINS`.

    Returns
    -------
    (column_name, level_used)
        The chosen dotted column name and the level it belongs to. If a
        fallback happened, ``level_used != requested_level``.

    Raises
    ------
    KeyError
        If ``engine`` isn't recognized, or ``requested_level`` isn't in
        :data:`FALLBACK_CHAINS`.
    ValueError
        If no candidate column from any level in the chain is present.
    """
    if engine not in QUANT_COLUMN_CANDIDATES:
        raise KeyError(f"Unknown engine {engine!r}. Available: {sorted(QUANT_COLUMN_CANDIDATES)}")
    if requested_level not in FALLBACK_CHAINS:
        raise KeyError(
            f"Unknown quantification_level {requested_level!r}. "
            f"Available: {sorted(FALLBACK_CHAINS)}"
        )

    chain = FALLBACK_CHAINS[requested_level]
    for level in chain:
        for col in QUANT_COLUMN_CANDIDATES[engine][level]:
            if col in available_columns:
                return col, level

    # Nothing in the chain was found -- collect the failure evidence.
    tried_cols: list[str] = []
    for level in chain:
        tried_cols.extend(QUANT_COLUMN_CANDIDATES[engine][level])
    raise ValueError(
        f"No quantification column found for engine={engine!r}, "
        f"requested_level={requested_level!r} (fallback chain: {list(chain)}). "
        f"Tried columns: {tried_cols}. "
        f"Available in data: {sorted(available_columns)[:20]}..."
    )
