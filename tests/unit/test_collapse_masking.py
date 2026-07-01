"""Unit tests for alphaphos.preprocess._collapse.masking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaphos.preprocess._collapse.masking import (
    drop_all_nan_sites,
    filter_by_global_max,
    mask_condition_aware,
    mask_per_run,
)


@pytest.fixture
def small_site_matrices():
    idx = pd.Index(["site1", "site2", "site3"], name="full_key")
    cols = ["s1", "s2", "s3", "s4"]
    # site1: all samples well-localized
    # site2: some samples below cutoff (loc < 0.75)
    # site3: only one sample above cutoff
    quant = pd.DataFrame(
        {
            "s1": [100.0, 200.0, 300.0],
            "s2": [150.0, 250.0, 350.0],
            "s3": [120.0, 220.0, 320.0],
            "s4": [180.0, 280.0, 380.0],
        },
        index=idx,
    )
    loc = pd.DataFrame(
        {
            "s1": [0.99, 0.99, 0.20],
            "s2": [0.99, 0.30, 0.10],
            "s3": [0.99, 0.99, 0.90],
            "s4": [0.99, 0.30, 0.10],
        },
        index=idx,
    )
    return quant, loc, cols


# ---------------------------------------------------------------------------
# mask_per_run
# ---------------------------------------------------------------------------


class TestMaskPerRun:
    def test_below_cutoff_becomes_nan(self, small_site_matrices):
        quant, loc, _ = small_site_matrices
        out = mask_per_run(quant, loc, cutoff=0.75)
        # site1: all >=0.99 -> unchanged
        assert not out.loc["site1"].isna().any()
        # site2: s2 (0.30) and s4 (0.30) -> masked
        assert np.isnan(out.loc["site2", "s2"])
        assert np.isnan(out.loc["site2", "s4"])
        assert out.loc["site2", "s1"] == 200.0  # kept
        # site3: only s3 kept
        row3 = out.loc["site3"]
        assert not np.isnan(row3["s3"])
        for s in ("s1", "s2", "s4"):
            assert np.isnan(row3[s])

    def test_shape_preserved(self, small_site_matrices):
        quant, loc, _ = small_site_matrices
        out = mask_per_run(quant, loc, cutoff=0.75)
        assert out.shape == quant.shape


# ---------------------------------------------------------------------------
# filter_by_global_max
# ---------------------------------------------------------------------------


class TestFilterByGlobalMax:
    def test_drops_low_max_sites(self, small_site_matrices):
        quant, loc, _ = small_site_matrices
        # site1 max=0.99, site2 max=0.99, site3 max=0.99 -> nothing dropped
        out_q, out_loc = filter_by_global_max(quant, loc, cutoff=0.75)
        assert list(out_q.index) == ["site1", "site2", "site3"]

    def test_drops_when_max_below_cutoff(self, small_site_matrices):
        quant, loc, _ = small_site_matrices
        # Zero out site3's row -> max=0
        loc.loc["site3"] = 0.10
        out_q, out_loc = filter_by_global_max(quant, loc, cutoff=0.75)
        assert list(out_q.index) == ["site1", "site2"]
        # aligned
        assert list(out_loc.index) == ["site1", "site2"]


# ---------------------------------------------------------------------------
# mask_condition_aware
# ---------------------------------------------------------------------------


class TestMaskConditionAware:
    def test_majority_classI_keeps_all(self):
        # 3 reps of ctrl, 3 reps of treated
        idx = pd.Index(["site1"], name="full_key")
        cols = ["c1", "c2", "c3", "t1", "t2", "t3"]
        quant = pd.DataFrame([[10, 20, 30, 40, 50, 60]], index=idx, columns=cols).astype(float)
        # site1 is classI in ctrl (>=2/3) and in treated (>=2/3), even though
        # one rep per group has low loc.
        loc = pd.DataFrame([[0.90, 0.90, 0.20, 0.85, 0.85, 0.30]], index=idx, columns=cols)
        cdf = pd.DataFrame(
            {"sample": cols, "condition": ["ctrl"] * 3 + ["treated"] * 3}
        )
        masked, decision = mask_condition_aware(
            quant, loc, condition_df=cdf, classI_cutoff=0.75, condition_threshold=0.5
        )
        # All cells kept because both conditions pass the threshold
        assert masked.loc["site1"].notna().all()
        assert decision.loc["site1", "ctrl"] == 2 / 3
        assert decision.loc["site1", "treated"] == 2 / 3

    def test_minority_classI_masks_low_loc(self):
        idx = pd.Index(["site1"], name="full_key")
        cols = ["c1", "c2", "c3"]
        quant = pd.DataFrame([[10.0, 20.0, 30.0]], index=idx, columns=cols)
        # 1 of 3 classI -> below 0.5 threshold -> mask non-classI cells
        loc = pd.DataFrame([[0.90, 0.20, 0.10]], index=idx, columns=cols)
        cdf = pd.DataFrame({"sample": cols, "condition": ["ctrl"] * 3})
        masked, _ = mask_condition_aware(
            quant, loc, condition_df=cdf, classI_cutoff=0.75, condition_threshold=0.5
        )
        # Only c1 kept (0.90 >= cutoff)
        assert masked.loc["site1", "c1"] == 10.0
        assert np.isnan(masked.loc["site1", "c2"])
        assert np.isnan(masked.loc["site1", "c3"])

    def test_missing_columns_in_condition_df_raises(self):
        idx = pd.Index(["site1"], name="full_key")
        quant = pd.DataFrame([[1.0]], index=idx, columns=["s1"])
        loc = pd.DataFrame([[1.0]], index=idx, columns=["s1"])
        cdf = pd.DataFrame({"sample": ["s1"]})  # missing 'condition'
        with pytest.raises(ValueError, match="must contain both"):
            mask_condition_aware(quant, loc, condition_df=cdf)


# ---------------------------------------------------------------------------
# drop_all_nan_sites
# ---------------------------------------------------------------------------


class TestDropAllNanSites:
    def test_drops_rows(self):
        idx = pd.Index(["site1", "site2", "site3"], name="full_key")
        quant = pd.DataFrame(
            {"s1": [1.0, np.nan, 3.0], "s2": [1.0, np.nan, 3.0]}, index=idx
        )
        loc = pd.DataFrame(
            {"s1": [0.9, 0.9, 0.9], "s2": [0.9, 0.9, 0.9]}, index=idx
        )
        meta = pd.DataFrame({"gene": ["G1", "G2", "G3"]}, index=idx)
        q_new, loc_new, meta_new = drop_all_nan_sites(quant, loc, meta)
        assert list(q_new.index) == ["site1", "site3"]
        assert list(loc_new.index) == ["site1", "site3"]
        assert list(meta_new.index) == ["site1", "site3"]
