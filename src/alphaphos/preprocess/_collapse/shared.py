"""Helpers shared by the two public collapse entry points.

:func:`alphaphos.collapse_sites` and :func:`alphaphos.collapse_precursors`
both need to (1) pick the quantification column through the schema fallback
chain and (2) optionally attach a stderr handler for ``verbose=True``.  One
implementation here keeps the two entry points from drifting apart.
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd

from alphaphos.constants import COL_CANONICAL_QUANT
from alphaphos.io.schemas import DEFAULT_QUANT_LEVEL, resolve_quant_column


def select_quantification_column(
    df: pd.DataFrame,
    *,
    engine: str,
    requested_level: str | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str, str]:
    """Pick the quant column and copy its values into the canonical slot.

    ``requested_level=None`` means the engine's default
    (:data:`alphaphos.io.schemas.DEFAULT_QUANT_LEVEL`: MS2 for Spectronaut,
    MS1 for DIA-NN).  Walks the ``MS2 -> MS1 -> auto`` fallback chain via
    :func:`alphaphos.io.schemas.resolve_quant_column`. If a fallback was
    needed, emits a ``UserWarning`` (attributed to the caller of the public
    entry point) AND logs a warning.

    Returns
    -------
    (df, chosen_col, level_used)
        DataFrame (a copy, only if the source column was NOT already the
        canonical slot), the actual source column name, and the level the
        source column belongs to.
    """
    if requested_level is None:
        requested_level = DEFAULT_QUANT_LEVEL[engine]
        logger.info("quantification_level=None -> engine default %r", requested_level)
    chosen_col, level_used = resolve_quant_column(
        set(df.columns), engine=engine, requested_level=requested_level
    )
    if level_used != requested_level:
        msg = (
            f"quantification_level={requested_level!r} unavailable in the input; "
            f"falling back to level={level_used!r} via column {chosen_col!r}."
        )
        # stacklevel: 1 = here, 2 = the public entry point, 3 = the user's call.
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)

    if chosen_col != COL_CANONICAL_QUANT:
        df = df.copy()
        df[COL_CANONICAL_QUANT] = df[chosen_col]

    logger.info("Using quantification column: %r (level=%s)", chosen_col, level_used)
    return df, chosen_col, level_used


def enable_verbose_logging(logger: logging.Logger) -> None:
    """Make ``logger`` emit INFO to stderr for the ``verbose=True`` convenience path.

    Attaches a ``StreamHandler`` once and lowers the level to INFO only when
    the logger is unconfigured or *less* permissive -- a user who already set
    DEBUG keeps DEBUG.  ``verbose=False`` callers must NOT touch the logger at
    all (library code should never raise a level the user configured).
    """
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
