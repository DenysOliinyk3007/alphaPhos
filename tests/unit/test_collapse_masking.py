"""Unit tests for alphaphos.preprocess._collapse.masking.

The per_run / global_max / condition strategies are exercised end-to-end
in tests/integration/test_collapse_spiked.py (TestLocalizationStrategies).
This file keeps only the pure-code paths those integration tests don't
reach: the below-cutoff branch of mask_condition_aware, condition_df
validation, the drop-when-max-below-cutoff branch of filter_by_global_max,
and the drop_all_nan_sites alignment invariant.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from alphaphos.preprocess._collapse.masking import (
    drop_all_nan_sites,
    filter_by_global_max,
    mask_condition_aware,
)


def test_filter_by_global_max_drops_low_max_sites():
    idx = pd.Index(["site1", "site2", "site3"], name="full_key")
    quant = pd.DataFrame({"s1": [10.0, 20.0, 30.0], "s2": [15.0, 25.0, 35.0]}, index=idx)
    loc = pd.DataFrame({"s1": [0.99, 0.99, 0.10], "s2": [0.99, 0.99, 0.20]}, index=idx)
    out_q, out_loc = filter_by_global_max(quant, loc, cutoff=0.75)
    # site3 max = 0.20 < 0.75 -> dropped from both matrices.
    assert list(out_q.index) == ["site1", "site2"]
    assert list(out_loc.index) == ["site1", "site2"]


def test_mask_condition_aware_masks_low_loc_when_minority_class_i():
    idx = pd.Index(["site1"], name="full_key")
    cols = ["c1", "c2", "c3"]
    quant = pd.DataFrame([[10.0, 20.0, 30.0]], index=idx, columns=cols)
    # 1 of 3 class-I -> below 0.5 threshold -> masks non-class-I cells.
    loc = pd.DataFrame([[0.90, 0.20, 0.10]], index=idx, columns=cols)
    cdf = pd.DataFrame({"sample": cols, "condition": ["ctrl"] * 3})
    masked, _ = mask_condition_aware(
        quant, loc, condition_df=cdf, classI_cutoff=0.75, condition_threshold=0.5
    )
    assert masked.loc["site1", "c1"] == 10.0
    assert np.isnan(masked.loc["site1", "c2"])
    assert np.isnan(masked.loc["site1", "c3"])


def test_mask_condition_aware_rejects_missing_condition_column():
    idx = pd.Index(["site1"], name="full_key")
    quant = pd.DataFrame([[1.0]], index=idx, columns=["s1"])
    loc = pd.DataFrame([[1.0]], index=idx, columns=["s1"])
    cdf = pd.DataFrame({"sample": ["s1"]})  # missing 'condition'
    with pytest.raises(ValueError, match="must contain both"):
        mask_condition_aware(quant, loc, condition_df=cdf)


def test_drop_all_nan_sites_aligns_all_three_matrices():
    idx = pd.Index(["site1", "site2", "site3"], name="full_key")
    quant = pd.DataFrame({"s1": [1.0, np.nan, 3.0], "s2": [1.0, np.nan, 3.0]}, index=idx)
    loc = pd.DataFrame({"s1": [0.9, 0.9, 0.9], "s2": [0.9, 0.9, 0.9]}, index=idx)
    meta = pd.DataFrame({"gene": ["G1", "G2", "G3"]}, index=idx)
    q_new, loc_new, meta_new = drop_all_nan_sites(quant, loc, meta)
    assert list(q_new.index) == ["site1", "site3"]
    assert list(loc_new.index) == ["site1", "site3"]
    assert list(meta_new.index) == ["site1", "site3"]


def test_mask_condition_aware_unlabelled_sample_gets_per_run_rule(caplog):
    # Sample u1 has no condition label. It must NOT bypass masking: its
    # low-loc cell is masked by the strict per-run rule, with a warning.
    idx = pd.Index(["site1"], name="full_key")
    cols = ["c1", "c2", "u1"]
    quant = pd.DataFrame([[10.0, 20.0, 30.0]], index=idx, columns=cols)
    loc = pd.DataFrame([[0.90, 0.85, 0.10]], index=idx, columns=cols)
    cdf = pd.DataFrame({"sample": ["c1", "c2"], "condition": ["ctrl", "ctrl"]})
    with caplog.at_level(logging.WARNING):
        masked, decision = mask_condition_aware(quant, loc, condition_df=cdf)
    assert masked.loc["site1", "c1"] == 10.0
    assert masked.loc["site1", "c2"] == 20.0
    assert np.isnan(masked.loc["site1", "u1"])
    assert list(decision.columns) == ["ctrl"]
    assert any("no entry in condition_df" in r.message for r in caplog.records)


def test_mask_condition_aware_duplicated_condition_rows_do_not_inflate_n():
    idx = pd.Index(["site1"], name="full_key")
    cols = ["c1", "c2"]
    quant = pd.DataFrame([[10.0, 20.0]], index=idx, columns=cols)
    loc = pd.DataFrame([[0.90, 0.10]], index=idx, columns=cols)
    cdf = pd.DataFrame({"sample": ["c1", "c1", "c2"], "condition": ["ctrl"] * 3})
    _, decision = mask_condition_aware(quant, loc, condition_df=cdf)
    # 1 Class-I of 2 distinct samples -- not 2 of 3 from the duplicated row.
    assert decision.loc["site1", "ctrl"] == 0.5
