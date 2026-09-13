"""Deprecated shim for the condition-aware Class-I mask.

The rule itself lives in :func:`alphaphos.preprocess._collapse.masking.mask_condition_aware`
and is applied by :func:`alphaphos.collapse_sites` when
``localization_strategy="condition"`` (the default).  This module used to carry
an independent copy that operated on the pre-0.10 "PeptideCollapse v4" wide
table (``PTM_Collapse_key`` + metadata columns).  That layout no longer exists
anywhere in alphaPhos, so :func:`apply_condition_aware_classI_mask` now takes a
(sites × samples) quant matrix plus the matching localization matrix and
delegates.  It emits a ``DeprecationWarning`` and will be removed in a future
minor release.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import pandas as pd

from alphaphos.constants import OBS_CONDITION, OBS_SAMPLE
from alphaphos.preprocess._collapse.masking import drop_all_nan_sites, mask_condition_aware


def _coerce_condition_df(
    sample_to_condition: Mapping[str, str] | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Accept the legacy ``{sample: condition}`` / Series inputs as a condition_df."""
    if isinstance(sample_to_condition, pd.DataFrame):
        return sample_to_condition
    series = (
        sample_to_condition
        if isinstance(sample_to_condition, pd.Series)
        else pd.Series(dict(sample_to_condition))
    )
    return pd.DataFrame(
        {OBS_SAMPLE: series.index.astype(str), OBS_CONDITION: series.astype(str).to_numpy()}
    )


def apply_condition_aware_classI_mask(
    site_quant: pd.DataFrame,
    loc_per_run: pd.DataFrame,
    sample_to_condition: Mapping[str, str] | pd.Series | pd.DataFrame,
    classI_cutoff: float = 0.75,
    condition_threshold: float = 0.50,
    *,
    drop_all_nan: bool = True,
    return_decision_table: bool = False,
):
    """Deprecated -- use ``collapse_sites(advanced={"localization_strategy": "condition"})``.

    Parameters
    ----------
    site_quant
        ``(sites × samples)`` LINEAR intensity matrix indexed by site key.
    loc_per_run
        ``(sites × samples)`` localization probabilities, same index / columns.
    sample_to_condition
        ``{sample: condition}`` mapping, Series, or a ``condition_df`` with
        ``sample`` + ``condition`` columns.
    classI_cutoff, condition_threshold
        See :func:`alphaphos.preprocess._collapse.masking.mask_condition_aware`.
    drop_all_nan
        Drop sites whose whole row becomes NaN after masking.
    return_decision_table
        Also return the per-(site, condition) Class-I fraction table.
    """
    warnings.warn(
        "apply_condition_aware_classI_mask is deprecated and will be removed in a future "
        "release. Use alphaphos.collapse_sites(advanced={'localization_strategy': "
        "'condition'}) or alphaphos.preprocess._collapse.masking.mask_condition_aware.",
        DeprecationWarning,
        stacklevel=2,
    )
    masked, decision = mask_condition_aware(
        site_quant,
        loc_per_run,
        condition_df=_coerce_condition_df(sample_to_condition),
        classI_cutoff=classI_cutoff,
        condition_threshold=condition_threshold,
    )
    if drop_all_nan:
        masked, decision = drop_all_nan_sites(masked, decision)
    return (masked, decision) if return_decision_table else masked


__all__ = ["apply_condition_aware_classI_mask"]
