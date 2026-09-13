"""Typed experimental-design matrix construction for limma-style analyses.

Companion to :func:`alphaphos.diff_exp_limma` (2-group) and the multi-contrast
functions :func:`alphaphos.diff_exp_anova` + :func:`alphaphos.diff_exp_limma_contrasts`.
The design matrix is the ``(n_samples, n_coefficients)`` numeric encoding of
the experimental design; contrasts (the questions asked on top of the fit) live
elsewhere.

Uses the ``0 + condition + covariate + ...`` no-intercept parameterisation --
one column per condition level, no reference-level absorption -- so contrasts
are simple linear combinations of coefficients (``ICM - Healthy`` etc.) rather
than requiring intercept adjustments.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# Reserved obs column names that would clash with sanitized encoding.  Fails
# loudly if the caller tries to use them as covariates.
_RESERVED_COVARIATE_NAMES = frozenset({"__condition__", "__block__", "__intercept__"})


@dataclass(frozen=True)
class DesignMatrix:
    """Typed design matrix + provenance.

    Attributes
    ----------
    frame : pd.DataFrame
        Shape ``(n_samples, n_coefficients)``; sample names on the index,
        coefficient names on the columns.  Numeric dtype throughout.
    condition_column : str
        Name of the primary condition ``.obs`` column that was encoded.
    condition_levels : tuple[str, ...]
        Original (unsanitised) condition levels in the encoding order.
        The first entry is ``reference_level``.
    reference_level : str
        The condition level used as the reference (first coefficient).
    condition_coefficients : tuple[str, ...]
        The coefficient labels (in ``frame.columns``) corresponding to the
        condition levels, in the same order as ``condition_levels``.  Use
        these when building contrasts programmatically.
    covariates : tuple[str, ...]
        Names of extra ``.obs`` columns that were folded in.
    block_column : str | None
        Optional paired-block factor (e.g. ``"patient"``, ``"donor"``).  When
        set, dummy columns for the block levels are appended, minus the
        reference level.
    block_levels : tuple[str, ...]
        Block levels in encoding order.  Empty when ``block_column`` is None.
    sample_names : tuple[str, ...]
        ``adata.obs_names`` in encoding order.  Handy for downstream
        alignment.
    level_sanitization : dict[str, str]
        Maps original condition-level strings to the sanitized identifiers
        that appear in ``frame.columns`` (patsy / inmoose require valid
        Python identifiers because ``makeContrasts`` eval()s the string).
    """

    frame: pd.DataFrame
    condition_column: str
    condition_levels: tuple[str, ...]
    reference_level: str
    condition_coefficients: tuple[str, ...]
    covariates: tuple[str, ...] = ()
    block_column: str | None = None
    block_levels: tuple[str, ...] = ()
    sample_names: tuple[str, ...] = ()
    level_sanitization: dict[str, str] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.frame.shape[0])

    @property
    def n_coefficients(self) -> int:
        return int(self.frame.shape[1])

    @property
    def coefficient_labels(self) -> tuple[str, ...]:
        return tuple(self.frame.columns)


def design_matrix(
    adata: ad.AnnData,
    *,
    condition_column: str,
    reference_level: str | None = None,
    covariates: list[str] | None = None,
    block_column: str | None = None,
    sample_mask: np.ndarray | None = None,
) -> DesignMatrix:
    """Build a typed design matrix from an :class:`AnnData` object.

    Parameters
    ----------
    adata
        ``AnnData``; the ``obs`` frame is inspected.  ``adata.X`` /
        ``layers`` are not touched.
    condition_column
        Name of the primary factor column in ``adata.obs``.  Must be
        categorical / string.  Missing values are not allowed.
    reference_level
        Level to place first in the encoding (interpreted as the
        "reference" for contrasts).  Default: the alphabetically-first
        level present in ``adata.obs[condition_column]``.
    covariates
        Extra ``adata.obs`` columns to include as **additive fixed
        effects** (batch, technical covariates).  Categorical covariates
        are one-hot encoded (reference-level omitted).  Numeric covariates
        pass through as a single column.
    block_column
        Optional paired-block factor (patient / donor / plate).  When set,
        dummy columns for the block levels are appended (reference-level
        omitted).  This is limma's "fixed block" convention -- the block
        levels enter as fixed effects.
    sample_mask
        Optional boolean array over ``adata.obs`` selecting the subset of
        samples to encode.  ``None`` (default) uses all samples.

    Returns
    -------
    DesignMatrix
        The encoded matrix plus provenance.  ``.frame.columns`` are the
        coefficient names to reference when constructing contrasts.

    Raises
    ------
    KeyError
        If any of ``condition_column`` / ``covariates`` / ``block_column``
        is absent from ``adata.obs``.
    ValueError
        If ``reference_level`` is not present in the condition column, or
        if the condition column has NaN, or if the design would be
        rank-deficient.
    """
    obs = adata.obs
    if sample_mask is not None:
        if len(sample_mask) != adata.n_obs:
            raise ValueError(f"sample_mask length {len(sample_mask)} != adata.n_obs {adata.n_obs}")
        obs = obs.loc[sample_mask].copy()
    else:
        obs = obs.copy()

    if condition_column not in obs.columns:
        raise KeyError(f"condition_column {condition_column!r} not in adata.obs")
    cond_raw = obs[condition_column]
    if cond_raw.isna().any():
        raise ValueError(
            f"adata.obs[{condition_column!r}] contains NaN in "
            f"{int(cond_raw.isna().sum())} sample(s); drop or impute first."
        )
    cond_str = cond_raw.astype(str)

    # Level order: reference first, then remaining alphabetical.
    unique_levels = sorted(cond_str.unique())
    if reference_level is not None:
        reference_level = str(reference_level)
        if reference_level not in unique_levels:
            raise ValueError(
                f"reference_level {reference_level!r} not in "
                f"adata.obs[{condition_column!r}] levels {unique_levels}"
            )
        rest = [lv for lv in unique_levels if lv != reference_level]
        level_order = [reference_level, *rest]
    else:
        level_order = unique_levels
        reference_level = level_order[0]
    if len(level_order) < 2:
        raise ValueError(
            f"condition_column {condition_column!r} has only 1 level "
            f"({level_order[0]!r}); need at least 2 for a differential design."
        )

    # Sanitize levels for identifier-safety (inmoose.makeContrasts eval()s
    # the contrast strings, so column names must be valid Python idents).
    level_sanitization = sanitize_and_map_levels(level_order)

    # Validate covariates + block against the reserved-names set + presence.
    covariates_tuple = tuple(covariates) if covariates else ()
    for cov in covariates_tuple:
        if cov in _RESERVED_COVARIATE_NAMES:
            raise ValueError(f"covariate name {cov!r} is reserved")
        if cov not in obs.columns:
            raise KeyError(f"covariate {cov!r} not in adata.obs")
        if obs[cov].isna().any():
            raise ValueError(
                f"adata.obs[{cov!r}] contains NaN; drop or impute before adding it as a covariate."
            )
    if block_column is not None:
        if block_column in _RESERVED_COVARIATE_NAMES:
            raise ValueError(f"block_column {block_column!r} is reserved")
        if block_column not in obs.columns:
            raise KeyError(f"block_column {block_column!r} not in adata.obs")
        if obs[block_column].isna().any():
            raise ValueError(
                f"adata.obs[{block_column!r}] contains NaN; block factor must be complete."
            )

    # ---- Build the frame column by column.  Order matters:
    # condition levels first, then each covariate's encoded columns, then
    # the block dummies (reference-level omitted for both).
    columns: dict[str, pd.Series] = {}

    # Condition: one column per level (no reference absorption -- explicit
    # ``0 +`` parameterisation).
    for lv in level_order:
        safe = level_sanitization[lv]
        col_name = f"{condition_column}[{safe}]"
        columns[col_name] = (cond_str == lv).astype(np.float64)
    condition_coefs = tuple(f"{condition_column}[{level_sanitization[lv]}]" for lv in level_order)

    # Covariates
    for cov in covariates_tuple:
        col = obs[cov]
        if covariate_is_continuous(col, name=cov):
            columns[cov] = col.astype(np.float64).to_numpy()
        else:
            # Categorical -> dummy-code, drop the first level as reference.
            cov_str = col.astype(str)
            cov_levels = sorted(cov_str.unique())
            cov_sanit = sanitize_and_map_levels(cov_levels)
            for lv in cov_levels[1:]:  # drop reference (first level)
                safe = cov_sanit[lv]
                columns[f"{cov}[{safe}]"] = (cov_str == lv).astype(np.float64)

    # Block factor (fixed-effect encoding, reference-level dropped)
    block_levels: tuple[str, ...] = ()
    if block_column is not None:
        blk_str = obs[block_column].astype(str)
        blk_levels = sorted(blk_str.unique())
        block_levels = tuple(blk_levels)
        if len(blk_levels) < 2:
            logger.info(
                "design_matrix: block_column %r has only 1 level; skipping block encoding.",
                block_column,
            )
        else:
            blk_sanit = sanitize_and_map_levels(blk_levels)
            for lv in blk_levels[1:]:
                safe = blk_sanit[lv]
                columns[f"{block_column}[{safe}]"] = (blk_str == lv).astype(np.float64)

    frame = pd.DataFrame(columns, index=obs.index)

    # Rank check: catch rank-deficient designs (e.g. block level perfectly
    # confounded with condition level).
    rank = int(np.linalg.matrix_rank(frame.to_numpy()))
    if rank < frame.shape[1]:
        raise ValueError(
            f"design matrix is rank-deficient (rank {rank} < "
            f"{frame.shape[1]} coefficients).  Common cause: a block or "
            "covariate is perfectly confounded with the condition.  "
            f"Coefficients: {list(frame.columns)}"
        )

    logger.info(
        "design_matrix: %s (ref=%s) -> %d samples x %d coefficients (covariates=%s, block=%s)",
        condition_column,
        reference_level,
        frame.shape[0],
        frame.shape[1],
        list(covariates_tuple),
        block_column,
    )
    return DesignMatrix(
        frame=frame,
        condition_column=condition_column,
        condition_levels=tuple(level_order),
        reference_level=reference_level,
        condition_coefficients=condition_coefs,
        covariates=covariates_tuple,
        block_column=block_column,
        block_levels=block_levels,
        sample_names=tuple(obs.index.astype(str)),
        level_sanitization=level_sanitization,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_]")


def covariate_is_continuous(col: pd.Series, *, name: str) -> bool:
    """Decide how an ``obs`` covariate column enters the design matrix.

    * float dtype  -> continuous (one column, linear effect)
    * str / object / categorical -> categorical (dummy-coded)
    * **integer or bool dtype -> ``ValueError``**.  A batch/plate/run column
      coded ``1, 2, 3`` would otherwise be fit as a *linear trend* -- a silent
      model misspecification (the pre-0.24 behaviour).  The caller must say
      what they mean: ``.astype(str)`` for a factor, ``.astype(float)`` for a
      genuinely continuous covariate.
    """
    kind = col.dtype.kind
    if kind == "f":
        return True
    if kind in ("i", "u", "b"):
        raise ValueError(
            f"covariate {name!r} has {col.dtype} dtype, which is ambiguous: a batch / "
            "plate / run id coded as integers would be fit as a LINEAR TREND, not as "
            "groups. Cast it explicitly -- adata.obs[col].astype(str) for a categorical "
            "factor (dummy-coded), or .astype(float) for a continuous covariate."
        )
    return False


def sanitize_and_map_levels(labels: list[str]) -> dict[str, str]:
    """Map arbitrary level strings to valid Python identifiers (bijective).

    Levels like ``"EGF+"``, ``"D1120_Healthy_Scar"``, ``"KO/WT"`` all become
    identifier-safe.  Collisions are resolved by suffixing ``_2``, ``_3``, ...
    Shared by :mod:`alphaphos.stats.design` and the inmoose path in
    :mod:`alphaphos.stats.diff_exp` so both encode levels identically.
    """
    seen: dict[str, str] = {}
    used: set[str] = set()
    for raw in labels:
        s = str(raw)
        # Replace anything non-identifier with '_'
        safe = _ID_SAFE_RE.sub("_", s)
        # If it starts with a digit, prefix with 'x_' to be identifier-safe
        if safe and safe[0].isdigit():
            safe = f"x_{safe}"
        if not safe:
            safe = "empty"
        # Handle collisions
        base = safe
        counter = 2
        while safe in used:
            safe = f"{base}_{counter}"
            counter += 1
        used.add(safe)
        seen[s] = safe
    return seen
