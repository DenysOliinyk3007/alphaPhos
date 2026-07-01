"""Site-completeness filtering for phosphosite AnnData.

Drops sites (``.var`` rows) whose valid-value fraction is below a
threshold. Three group-handling strategies:

- ``"all"`` (default when no group column): compute the valid fraction
  across all samples together and keep sites that meet the threshold
  globally.
- ``"any"`` (needs ``group_column``): compute the valid fraction WITHIN
  each group and keep sites that meet the threshold in AT LEAST ONE
  group. Preserves condition-specific sites (e.g. induced upon
  stimulation, absent in control).
- ``"each"`` (needs ``group_column``): keep sites that meet the
  threshold in EVERY group. Ensures every condition contributes
  reliable data — the safest choice before differential testing.

Dropping var rows dispatches automatically to ``.X``, every layer, and
every ``varm`` matrix (AnnData contract), so this module doesn't touch
any of them explicitly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


VALID_STRATEGIES = ("all", "any", "each")


def filter_by_completeness(
    adata: ad.AnnData,
    *,
    min_valid_frac: float,
    group_column: str | None = None,
    keep_strategy: Literal["all", "any", "each"] = "all",
    layer: str | None = None,
) -> ad.AnnData:
    """Return an ``AnnData`` with sites failing the completeness threshold dropped.

    Parameters
    ----------
    adata : AnnData
        Shape ``(n_samples, n_sites)``.
    min_valid_frac : float in [0, 1]
        Minimum fraction of non-NaN values REQUIRED to keep a site. Applied
        per group (when ``group_column`` is set) or globally (``keep_strategy="all"``).
        E.g. ``0.7`` = require observations in >=70% of samples.
    group_column : str, optional
        Column in ``adata.obs`` that partitions samples into groups
        (e.g. ``"condition"``). Required for ``keep_strategy in ("any", "each")``;
        MUST be ``None`` for ``keep_strategy="all"``.
    keep_strategy : {"all", "any", "each"}
        - ``"all"``: no groups. A site is kept if its overall valid
          fraction >= ``min_valid_frac``. Default.
        - ``"any"``: needs ``group_column``. Keeps a site if it passes the
          threshold in at least one group.
        - ``"each"``: needs ``group_column``. Keeps a site only if it passes
          the threshold in EVERY group.
    layer : str, optional
        Which layer to compute missingness on. ``None`` (default) = ``adata.X``.
        Only affects the completeness computation; the filter DROPS var rows
        from every layer / obsm / varm automatically via AnnData subsetting.

    Returns
    -------
    AnnData
        A new AnnData with the failing sites removed. Always returns; does
        not mutate the input.

    Raises
    ------
    ValueError
        If ``min_valid_frac`` is out of range, ``keep_strategy`` is unknown,
        or the strategy vs ``group_column`` combination is invalid.
    KeyError
        If ``group_column`` is set but not in ``adata.obs``, or ``layer`` is
        set but not in ``adata.layers``.
    """
    if not 0.0 <= float(min_valid_frac) <= 1.0:
        raise ValueError(f"min_valid_frac must be in [0, 1], got {min_valid_frac!r}")
    if keep_strategy not in VALID_STRATEGIES:
        raise ValueError(f"keep_strategy must be one of {VALID_STRATEGIES}, got {keep_strategy!r}")
    if keep_strategy == "all" and group_column is not None:
        raise ValueError(
            "keep_strategy='all' computes a global (ungrouped) filter; "
            "drop group_column, or set keep_strategy to 'any' or 'each'."
        )
    if keep_strategy in ("any", "each") and group_column is None:
        raise ValueError(f"keep_strategy={keep_strategy!r} requires group_column to be set.")
    if group_column is not None and group_column not in adata.obs.columns:
        raise KeyError(
            f"group_column={group_column!r} not in adata.obs. Available: {list(adata.obs.columns)}"
        )
    if layer is not None and layer not in adata.layers:
        raise KeyError(
            f"layer={layer!r} not in adata.layers. Available: {list(adata.layers.keys())}"
        )

    X = np.asarray(adata.X if layer is None else adata.layers[layer], dtype=float)
    not_nan = ~np.isnan(X)  # (n_samples, n_sites) True where observed

    if keep_strategy == "all":
        n_samples = X.shape[0]
        valid_frac = not_nan.sum(axis=0) / max(n_samples, 1)
        keep = valid_frac >= min_valid_frac
    else:
        groups = adata.obs[group_column].astype(str).values
        unique_groups = np.unique(groups)
        # per_group_pass[i, j] = does site j pass in group i?
        per_group_pass = np.zeros((len(unique_groups), X.shape[1]), dtype=bool)
        for gi, g in enumerate(unique_groups):
            mask = groups == g
            n_g = int(mask.sum())
            if n_g == 0:
                continue
            valid_frac = not_nan[mask].sum(axis=0) / n_g
            per_group_pass[gi] = valid_frac >= min_valid_frac
        if keep_strategy == "any":
            keep = per_group_pass.any(axis=0)
        else:  # "each"
            keep = per_group_pass.all(axis=0)

    n_before = adata.n_vars
    result = adata[:, keep].copy()

    logger.info(
        "filter_by_completeness: %d -> %d sites (strategy=%s, min_valid_frac=%.2f%s)",
        n_before,
        result.n_vars,
        keep_strategy,
        min_valid_frac,
        f", group={group_column}" if group_column else "",
    )
    return result
