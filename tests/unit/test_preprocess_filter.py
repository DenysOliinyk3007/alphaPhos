"""Unit tests for :mod:`alphaphos.preprocess.filter`.

Fixture layout used across tests: 6 samples in 2 groups (3 ctrl, 3 trt),
4 sites with different missingness patterns::

    site      ctrl_1  ctrl_2  ctrl_3   trt_1  trt_2  trt_3   overall
    A          v       v       v       v      v      v       6/6 valid
    B          v       v       nan     v      v      v       5/6 valid
    C          v       v       nan     nan    nan    nan     2/6 valid
    D          nan     nan     nan     v      v      v       3/6 valid

Where 'v' is a real value and 'nan' is missing. With min_valid_frac=0.7:

- **"all" (global)**: A (100%), B (83%) pass; C (33%) and D (50%) fail.
- **"any" (per-group)**: A passes in both. B passes in trt (100%). C passes
  nowhere (66% in ctrl, 0% in trt). D passes in trt (100%). Keep = A, B, D.
- **"each" (per-group)**: A passes in both. B passes in trt but only 66%
  in ctrl -> fails. C fails everywhere. D passes trt but fails ctrl.
  Keep = A only.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphaphos.preprocess.filter import filter_by_completeness


@pytest.fixture
def sample_adata():
    v = 1.0
    X = np.array(
        [
            # ctrl_1  ctrl_2  ctrl_3   trt_1   trt_2   trt_3
            [v, v, v, v, v, v],  # site A: all observed (6/6)
            [v, v, np.nan, v, v, v],  # site B: 5/6 (ctrl 2/3, trt 3/3)
            [v, v, np.nan, np.nan, np.nan, np.nan],  # site C: 2/6 (ctrl 2/3, trt 0/3)
            [np.nan, np.nan, np.nan, v, v, v],  # site D: 3/6 (ctrl 0/3, trt 3/3)
        ],
        dtype=float,
    ).T  # AnnData wants (n_samples, n_sites)
    obs = pd.DataFrame(
        {"condition": ["ctrl", "ctrl", "ctrl", "trt", "trt", "trt"]},
        index=["ctrl_1", "ctrl_2", "ctrl_3", "trt_1", "trt_2", "trt_3"],
    )
    var = pd.DataFrame(index=["A", "B", "C", "D"])
    return ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)


# ---------------------------------------------------------------------------
# keep_strategy="all" (global filter)
# ---------------------------------------------------------------------------


class TestGlobalFilter:
    @pytest.mark.parametrize(
        ("threshold", "expected"),
        [
            (0.0, ["A", "B", "C", "D"]),  # permissive: all pass
            (0.7, ["A", "B"]),  # moderate: A (100%) + B (83%)
            (1.0, ["A"]),  # strict: only A has every sample observed
        ],
    )
    def test_threshold_gates_correctly(self, sample_adata, threshold, expected):
        out = filter_by_completeness(sample_adata, min_valid_frac=threshold)
        assert list(out.var_names) == expected


# ---------------------------------------------------------------------------
# keep_strategy="any" (per-group, keep if passing anywhere)
# ---------------------------------------------------------------------------


class TestAnyStrategy:
    def test_keeps_condition_specific_sites(self, sample_adata):
        # 0.7 in EACH group: A passes both, B passes trt (100%), D passes trt (100%).
        # C fails both (2/3=0.67 ctrl, 0/3=0 trt).
        out = filter_by_completeness(
            sample_adata,
            min_valid_frac=0.7,
            group_column="condition",
            keep_strategy="any",
        )
        assert sorted(out.var_names) == ["A", "B", "D"]

    def test_permissive_keeps_all(self, sample_adata):
        out = filter_by_completeness(
            sample_adata,
            min_valid_frac=0.0,
            group_column="condition",
            keep_strategy="any",
        )
        assert list(out.var_names) == ["A", "B", "C", "D"]

    def test_strict_needs_full_group(self, sample_adata):
        # 1.0 in ANY group: A (both), B (trt), D (trt). C fails.
        out = filter_by_completeness(
            sample_adata,
            min_valid_frac=1.0,
            group_column="condition",
            keep_strategy="any",
        )
        assert sorted(out.var_names) == ["A", "B", "D"]


# ---------------------------------------------------------------------------
# keep_strategy="each" (per-group, must pass in every group)
# ---------------------------------------------------------------------------


class TestEachStrategy:
    def test_strict_keeps_only_universal_sites(self, sample_adata):
        # 0.7 in EACH group: A passes both (100%/100%). B fails in ctrl (2/3=0.67).
        # C fails ctrl (0.67) and trt (0). D fails ctrl (0).
        out = filter_by_completeness(
            sample_adata,
            min_valid_frac=0.7,
            group_column="condition",
            keep_strategy="each",
        )
        assert list(out.var_names) == ["A"]

    def test_relaxed_threshold(self, sample_adata):
        # 0.5 in EACH group: A both (1.0/1.0), B ctrl (0.67), trt (1.0). C ctrl (0.67), trt (0) -> C fails.
        # D ctrl (0.0) -> D fails.
        out = filter_by_completeness(
            sample_adata,
            min_valid_frac=0.5,
            group_column="condition",
            keep_strategy="each",
        )
        assert sorted(out.var_names) == ["A", "B"]


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class TestLayers:
    def test_filter_on_layer(self, sample_adata):
        # Add a layer where all values are NaN so any layer-based filter drops everything.
        sample_adata.layers["empty"] = np.full_like(sample_adata.X, np.nan)
        out = filter_by_completeness(sample_adata, min_valid_frac=0.1, layer="empty")
        assert out.n_vars == 0

    def test_layers_propagate_after_drop(self, sample_adata):
        # A second layer is preserved for kept rows and dropped for filtered ones.
        sample_adata.layers["copy"] = sample_adata.X.copy()
        out = filter_by_completeness(sample_adata, min_valid_frac=0.7)
        assert out.n_vars == 2
        assert out.layers["copy"].shape == (6, 2)

    def test_missing_layer_raises(self, sample_adata):
        with pytest.raises(KeyError, match="not in adata.layers"):
            filter_by_completeness(sample_adata, min_valid_frac=0.5, layer="doesnt_exist")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_valid_frac": 1.5}, "min_valid_frac"),
            ({"min_valid_frac": 0.5, "keep_strategy": "most"}, "keep_strategy"),
            (
                {
                    "min_valid_frac": 0.5,
                    "group_column": "condition",
                    "keep_strategy": "all",
                },
                "global",
            ),
            (
                {"min_valid_frac": 0.5, "keep_strategy": "any"},
                "requires group_column",
            ),
            (
                {"min_valid_frac": 0.5, "keep_strategy": "each"},
                "requires group_column",
            ),
        ],
    )
    def test_validation_raises(self, sample_adata, kwargs, match):
        with pytest.raises(ValueError, match=match):
            filter_by_completeness(sample_adata, **kwargs)

    def test_missing_group_column_raises(self, sample_adata):
        with pytest.raises(KeyError, match="not in adata.obs"):
            filter_by_completeness(
                sample_adata,
                min_valid_frac=0.5,
                group_column="nope",
                keep_strategy="each",
            )


# ---------------------------------------------------------------------------
# Non-mutation contract
# ---------------------------------------------------------------------------


class TestNonMutation:
    def test_input_unchanged(self, sample_adata):
        original_var = list(sample_adata.var_names)
        out = filter_by_completeness(sample_adata, min_valid_frac=1.0)
        # Input is untouched.
        assert list(sample_adata.var_names) == original_var
        # Result is a new (filtered) object.
        assert list(out.var_names) == ["A"]
        assert out is not sample_adata


# ---------------------------------------------------------------------------
# Integration: filter feeds impute cleanly
# ---------------------------------------------------------------------------


class TestFilterThenImpute:
    def test_filter_then_impute_hybrid(self, sample_adata):
        # After filtering to sites present in >= 70% of each group, impute_hybrid
        # should complete without the "all-nan column" error.
        from alphaphos.preprocess.impute import impute_hybrid

        # Use "each" strict per-group to drop D (all-nan in ctrl).
        # "each" 0.5 -> A, B pass; C, D fail.
        filtered = filter_by_completeness(
            sample_adata,
            min_valid_frac=0.5,
            group_column="condition",
            keep_strategy="each",
        )
        assert filtered.n_vars == 2
        # impute_hybrid defaults to layer='intensity_log2'; this fixture only
        # has .X, so target that explicitly.
        imputed, _audit = impute_hybrid(filtered, layer=None, return_audit=True, copy=True)
        # The single NaN in site B should have been imputed.
        assert not np.isnan(imputed.X).any()
