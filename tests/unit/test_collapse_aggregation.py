"""Unit tests for alphaphos.preprocess._collapse.aggregation.

Tests both the strategy dispatch (``aggregate_by_key``) and the Hogrebe
``consolidate`` algorithm in isolation. Focus is on correctness at the
edges (empty matrices, single row, all-NaN samples) since these are the
places past bugs have hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaphos.preprocess._collapse.aggregation import (
    aggregate_by_key,
    consolidate,
)

# ---------------------------------------------------------------------------
# consolidate (Hogrebe ratio imputation + sum)
# ---------------------------------------------------------------------------


class TestConsolidate:
    def test_empty_matrix(self):
        # No rows, 3 samples -> all-NaN vector
        out = consolidate(np.empty((0, 3)))
        assert out.shape == (3,)
        assert np.all(np.isnan(out))

    def test_single_row_fewer_than_2_values_dropped(self):
        # Row with <=1 non-NaN values gets pre-filtered -> all-NaN result
        m = np.array([[1.0, np.nan, np.nan]])
        out = consolidate(m)
        assert np.all(np.isnan(out))

    def test_single_row_with_2plus_values_kept(self):
        # Single surviving row -> returned verbatim
        m = np.array([[10.0, 20.0, 30.0]])
        out = consolidate(m)
        assert np.allclose(out, [10.0, 20.0, 30.0])

    def test_two_rows_no_missing(self):
        m = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
        out = consolidate(m)
        # Both rows complete -> just sum
        assert np.allclose(out, [11.0, 22.0, 33.0])

    def test_impute_from_partner_ratio(self):
        # Row 1 = 2 * Row 2 across shared positions.
        # Row 2 has a gap at position 2 that should be filled from Row 1 * 0.5 = 20 * 0.5 = 10
        m = np.array(
            [
                [10.0, 20.0, 30.0],
                [5.0, np.nan, 15.0],
            ]
        )
        out = consolidate(m)
        # After imputation: [10, 20, 30] + [5, 10, 15] = [15, 30, 45]
        assert np.allclose(out, [15.0, 30.0, 45.0])

    def test_all_nan_column_unrecoverable(self):
        # When a column is NaN across ALL rows, imputation can't fill it.
        # Because both rows also have this shared gap and can't help each
        # other, the algorithm drops rows until nothing survives and
        # returns all-NaN. This is the documented behavior of the R script.
        m = np.array(
            [
                [10.0, np.nan, 30.0],
                [5.0, np.nan, 15.0],
            ]
        )
        out = consolidate(m)
        # All NaN because both rows have unrecoverable gaps and the
        # algorithm can't produce a consistent sum.
        assert np.all(np.isnan(out))

    def test_partial_impute_when_other_row_has_signal(self):
        # Row 0 has a gap that row 1 CAN fill (row 1 has value at that position).
        # Row 1 has no gaps.
        m = np.array(
            [
                [10.0, np.nan, 30.0],  # gap at pos 1
                [5.0, 12.0, 15.0],  # complete
            ]
        )
        out = consolidate(m)
        # Row 0 should be imputed: ratio(row0/row1) via shared positions
        # (0 and 2) is median(10/5, 30/15) = median(2, 2) = 2, so row0[1] ~ 2*12 = 24.
        # Then sum: [15, 36, 45].
        assert np.allclose(out, [15.0, 36.0, 45.0])

    def test_returns_correct_shape(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert consolidate(m).shape == (2,)


# ---------------------------------------------------------------------------
# aggregate_by_key -- dispatch tests
# ---------------------------------------------------------------------------


@pytest.fixture
def small_wide_df():
    # 2 sites (A, B), 3 samples (s1, s2, s3), 2 precursor rows per site
    df = pd.DataFrame(
        {"s1": [10.0, 20.0, 5.0, 15.0], "s2": [30.0, 40.0, 25.0, 35.0], "s3": [1.0, 2.0, 3.0, 4.0]},
        index=pd.Index(["A", "A", "B", "B"], name="key"),
    )
    return df


class TestAggregateByKey:
    def test_sum(self, small_wide_df):
        out = aggregate_by_key(small_wide_df, "sum", ["s1", "s2", "s3"])
        assert list(out.index) == ["A", "B"]
        assert out.loc["A", "s1"] == 30.0  # 10 + 20
        assert out.loc["B", "s2"] == 60.0  # 25 + 35

    def test_median(self, small_wide_df):
        out = aggregate_by_key(small_wide_df, "median", ["s1", "s2", "s3"])
        assert out.loc["A", "s1"] == 15.0  # median(10, 20)
        assert out.loc["B", "s2"] == 30.0  # median(25, 35)

    def test_mean(self, small_wide_df):
        out = aggregate_by_key(small_wide_df, "mean", ["s1", "s2", "s3"])
        assert out.loc["A", "s1"] == 15.0
        assert out.loc["B", "s3"] == 3.5

    def test_consolidate(self, small_wide_df):
        # With no missing values, consolidate degenerates to per-sample sum.
        out = aggregate_by_key(small_wide_df, "consolidate", ["s1", "s2", "s3"])
        # Row order in output depends on sort-by-median; check by index label.
        assert out.loc["A", "s1"] == 30.0
        assert out.loc["B", "s1"] == 20.0

    def test_unknown_method_raises(self, small_wide_df):
        with pytest.raises(ValueError, match="aggregation method"):
            aggregate_by_key(small_wide_df, "sqrt-of-count", ["s1"])
