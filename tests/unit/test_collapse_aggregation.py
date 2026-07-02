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
        m = np.array(
            [
                [10.0, np.nan, 30.0],
                [5.0, 12.0, 15.0],
            ]
        )
        out = consolidate(m)
        # Row 0 imputed via row 1: ratio via shared positions is 2, so
        # row0[1] ~ 24; sum with row 1 -> [15, 36, 45].
        assert np.allclose(out, [15.0, 36.0, 45.0])


# ---------------------------------------------------------------------------
# aggregate_by_key -- dispatch tests
# Numeric correctness of each aggregation method (sum / median / mean /
# consolidate) is verified end-to-end in tests/integration/test_collapse_spiked.py
# (TestAggregationMethods). Only the "unknown method raises" pure-code path
# needs a unit test here.
# ---------------------------------------------------------------------------


def test_unknown_method_raises():
    df = pd.DataFrame(
        {"s1": [1.0, 2.0]}, index=pd.Index(["A", "A"], name="key")
    )
    with pytest.raises(ValueError, match="aggregation method"):
        aggregate_by_key(df, "sqrt-of-count", ["s1"])
