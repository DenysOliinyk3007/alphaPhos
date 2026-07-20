"""Tests for the advisory decision tree in ``alphaphos.recommend``.

The advisor is print-only, so tests capture stdout with ``capsys`` and assert
on the reasoning lines and code snippet that come out.  Also unit-tests the
pure decision helpers directly so branch coverage isn't captive to
stdout format.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphaphos import recommend_pipeline
from alphaphos.recommend import (
    CLASSI_DEFAULT,
    N_CAP,
    _audit_and_drop_cells,
    _decide_classI,
    _decide_de,
    _decide_filter,
    _decide_imputer,
    _n_from_smallest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk_adata(
    n_samples: int,
    n_features: int,
    obs: pd.DataFrame,
    *,
    with_loc_prob: bool = True,
    nan_fraction: float = 0.3,
    seed: int = 0,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.normal(20, 2, size=(n_samples, n_features)).astype(np.float32)
    nan_mask = rng.random(X.shape) < nan_fraction
    X[nan_mask] = np.nan
    var = pd.DataFrame(index=[f"site_{i}" for i in range(n_features)])
    if with_loc_prob:
        var["mean_loc_prob"] = rng.uniform(0.4, 1.0, size=n_features)
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def small_two_cond() -> ad.AnnData:
    """n=6, 2 conditions × 3 reps.  Mimics an EGF-style pilot."""
    obs = pd.DataFrame(
        {
            "condition": ["EGF+", "EGF+", "EGF+", "EGF-", "EGF-", "EGF-"],
        },
        index=[f"s{i}" for i in range(6)],
    )
    return _mk_adata(6, 200, obs, nan_fraction=0.2, seed=1)


@pytest.fixture
def factorial_medium() -> ad.AnnData:
    """n=60, 4 fiber_type × 3 time_point (5 per cell)."""
    fiber = ["I", "IIa", "IIx", "IIb"]
    time_pts = ["T1", "T2", "T3"]
    obs_rows = [(f, t, r) for f in fiber for t in time_pts for r in range(5)]
    obs = pd.DataFrame(
        obs_rows,
        columns=["fiber_type", "time_point", "replicate"],
        index=[f"s{i}" for i in range(len(obs_rows))],
    )
    return _mk_adata(len(obs), 500, obs, nan_fraction=0.35, seed=2)


@pytest.fixture
def factorial_with_singleton() -> ad.AnnData:
    """4x3 balanced except 'mixed|T3' has only 1 sample -- must be auto-dropped."""
    rows = []
    for f in ["I", "IIa", "IIx", "IIb"]:
        for t in ["T1", "T2", "T3"]:
            for r in range(5):
                rows.append((f, t, r))
    rows.append(("mixed", "T3", 0))  # the offender
    obs = pd.DataFrame(
        rows,
        columns=["fiber_type", "time_point", "replicate"],
        index=[f"s{i}" for i in range(len(rows))],
    )
    return _mk_adata(len(obs), 400, obs, nan_fraction=0.4, seed=3)


@pytest.fixture
def large_cohort() -> ad.AnnData:
    """n=320, 4x2 factorial -- triggers the LARGE_COHORT PIMMS-DAE branch."""
    rows = [(f, t) for f in ["A", "B", "C", "D"] for t in ["T1", "T2"] for _ in range(40)]
    obs = pd.DataFrame(rows, columns=["group", "time"], index=[f"s{i}" for i in range(len(rows))])
    return _mk_adata(len(obs), 300, obs, nan_fraction=0.55, seed=4)


# ---------------------------------------------------------------------------
# Pure decision helpers
# ---------------------------------------------------------------------------


class TestDecideClassI:
    def test_proteome_skips(self, small_two_cond):
        cutoff, why = _decide_classI("proteome", small_two_cond)
        assert cutoff is None
        assert "proteome" in why.lower()

    def test_phospho_default_075(self, small_two_cond):
        cutoff, why = _decide_classI("phospho", small_two_cond)
        assert cutoff == CLASSI_DEFAULT == 0.75
        assert "mean_loc_prob" in why

    def test_missing_column_skips(self, small_two_cond):
        adata = _mk_adata(6, 100, small_two_cond.obs, with_loc_prob=False)
        cutoff, why = _decide_classI("phospho", adata)
        assert cutoff is None
        assert "mean_loc_prob" in why


class TestAuditAndDropCells:
    def test_no_drop_when_balanced(self, factorial_medium):
        dropped, why = _audit_and_drop_cells(factorial_medium, "fiber_type", "time_point")
        assert dropped == []
        assert "≥" in why or "all" in why.lower()

    def test_drops_singleton_secondary_cell(self, factorial_with_singleton):
        dropped, why = _audit_and_drop_cells(factorial_with_singleton, "fiber_type", "time_point")
        assert len(dropped) == 1
        assert "mixed" in why

    def test_no_secondary_checks_primary(self):
        obs = pd.DataFrame(
            {"cond": ["A"] * 4 + ["B"] * 2 + ["C"] * 1},
            index=[f"s{i}" for i in range(7)],
        )
        adata = _mk_adata(7, 50, obs, with_loc_prob=False)
        dropped, why = _audit_and_drop_cells(adata, "cond", None)
        # 'C' has only 1 sample → dropped; 'B' has 2, also < MIN_CELL_N=3
        assert set(dropped) == {"s4", "s5", "s6"}
        assert "cond" in why

    def test_missing_factor_returns_empty(self, small_two_cond):
        dropped, why = _audit_and_drop_cells(small_two_cond, "does_not_exist", None)
        assert dropped == []
        assert "not in obs" in why


class TestNFromSmallest:
    @pytest.mark.parametrize(
        "smallest, expected",
        [(3, 3), (5, 3), (7, 4), (10, 6), (17, 10), (100, 10)],
    )
    def test_cap_at_10(self, smallest, expected):
        assert _n_from_smallest(smallest) == expected

    def test_never_below_3(self):
        assert _n_from_smallest(1) == 3
        assert _n_from_smallest(0) == 3


class TestDecideFilter:
    def test_primary_de_uses_interaction_when_cells_large_enough(self, factorial_medium):
        cfg, _why = _decide_filter("primary_de", factorial_medium, "fiber_type", "time_point")
        assert cfg["keep_strategy"] == "each"
        assert cfg["group_column"] == "fiber_type_x_time_point"
        assert "_synth_from" in cfg
        assert cfg["min_valid_n"] == _n_from_smallest(5)  # smallest cell = 5

    def test_primary_de_falls_back_when_cells_too_small(self):
        obs = pd.DataFrame(
            {
                "cond": ["A"] * 6 + ["B"] * 6,
                "batch": ["b1", "b2"] * 6,  # 3 in each cell → smallest=3 <5 → fallback
            },
            index=[f"s{i}" for i in range(12)],
        )
        adata = _mk_adata(12, 50, obs, with_loc_prob=False)
        cfg, _why = _decide_filter("primary_de", adata, "cond", "batch")
        assert cfg["group_column"] == "cond"
        assert cfg["keep_strategy"] == "each"

    def test_interaction_de_forces_synth_group(self, factorial_medium):
        cfg, _why = _decide_filter("interaction_de", factorial_medium, "fiber_type", "time_point")
        assert "_synth_from" in cfg
        assert cfg["keep_strategy"] == "each"

    def test_interaction_de_no_secondary_falls_back(self, small_two_cond):
        cfg, why = _decide_filter("interaction_de", small_two_cond, "condition", None)
        assert cfg["group_column"] == "condition"
        assert cfg["keep_strategy"] == "each"
        assert "cannot form interaction cells" in why

    def test_onoff_discovery_uses_any(self, factorial_medium):
        cfg, _why = _decide_filter("onoff_discovery", factorial_medium, "fiber_type", None)
        assert cfg["keep_strategy"] == "any"
        assert cfg["group_column"] == "fiber_type"

    def test_profiling_uses_frac(self, small_two_cond):
        cfg, _why = _decide_filter("profiling", small_two_cond, "condition", None)
        assert cfg.get("min_valid_frac") == 0.7
        assert cfg["keep_strategy"] == "all"

    def test_viz_only_permissive(self, factorial_medium):
        cfg, _why = _decide_filter("viz_only", factorial_medium, "fiber_type", None)
        assert cfg["keep_strategy"] == "any"
        # loose n (0.3 × smallest, floor 3); smallest fiber group = 15 → 4-5
        assert cfg["min_valid_n"] <= N_CAP


class TestDecideDE:
    def test_onoff_returns_dedicated_detector(self):
        name, _why, snippet = _decide_de("onoff_discovery")
        assert "on_off_detection" in name
        assert "on_off_detection" in snippet

    def test_viz_only_no_de(self):
        name, _why, snippet = _decide_de("viz_only")
        assert "none" in name.lower()
        assert snippet == ""

    def test_primary_de_has_both_hooks(self):
        _name, _why, snippet = _decide_de("primary_de")
        assert "limma_observed_only" in snippet
        assert "limma_contrasts" in snippet


class TestDecideImputer:
    def test_no_imputer_when_de_handles_nan(self):
        method, why, _ = _decide_imputer("primary_de", n_samples=100, pct_nan_post_filter=0.3)
        assert method is None
        assert "skip imputation" in why

    def test_knn_for_tiny_cohort(self):
        method, _why, snippet = _decide_imputer("viz_only", n_samples=20, pct_nan_post_filter=0.3)
        assert method == "knn"
        assert "impute_knn" in snippet

    def test_pimms_dae_for_large_cohort(self):
        method, _why, snippet = _decide_imputer("viz_only", n_samples=350, pct_nan_post_filter=0.2)
        assert method == "pimms_dae"
        assert "impute_pimms" in snippet

    def test_pimms_dae_for_heavy_nan(self):
        method, _why, _ = _decide_imputer("viz_only", n_samples=100, pct_nan_post_filter=0.5)
        assert method == "pimms_dae"

    def test_knn_default_middle_cohort_low_nan(self):
        method, _why, _ = _decide_imputer("viz_only", n_samples=100, pct_nan_post_filter=0.2)
        assert method == "knn"


# ---------------------------------------------------------------------------
# End-to-end print (integration)
# ---------------------------------------------------------------------------


class TestRecommendPipelinePrint:
    def _get_output(self, capsys) -> str:
        return capsys.readouterr().out

    def test_prints_all_sections(self, factorial_medium, capsys):
        recommend_pipeline(
            factorial_medium,
            goal="primary_de",
            data_type="phospho",
            primary_factor="fiber_type",
            secondary_factor="time_point",
        )
        out = self._get_output(capsys)
        assert "ALPHAPHOS PIPELINE RECOMMENDATION" in out
        assert "Decision trace" in out
        assert "Recommended code" in out
        assert "Expected outcome" in out

    def test_shows_class_I_step_for_phospho(self, factorial_medium, capsys):
        recommend_pipeline(
            factorial_medium,
            goal="marginal_de",
            data_type="phospho",
            primary_factor="fiber_type",
        )
        out = self._get_output(capsys)
        assert 'mean_loc_prob"] >= 0.75' in out

    def test_skips_class_I_step_for_proteome(self, factorial_medium, capsys):
        recommend_pipeline(
            factorial_medium,
            goal="marginal_de",
            data_type="proteome",
            primary_factor="fiber_type",
        )
        out = self._get_output(capsys)
        assert "proteome" in out.lower()
        assert 'mean_loc_prob"] >= 0.75' not in out

    def test_shows_drop_step_when_singleton_present(self, factorial_with_singleton, capsys):
        recommend_pipeline(
            factorial_with_singleton,
            goal="primary_de",
            data_type="phospho",
            primary_factor="fiber_type",
            secondary_factor="time_point",
        )
        out = self._get_output(capsys)
        assert "Drop 1 sample" in out
        assert "mixed" in out

    def test_pimms_dae_recommended_for_large_cohort_viz(self, large_cohort, capsys):
        recommend_pipeline(
            large_cohort,
            goal="viz_only",
            data_type="phospho",
            primary_factor="group",
            secondary_factor="time",
        )
        out = self._get_output(capsys)
        assert 'impute_pimms(adata, model="DAE"' in out

    def test_knn_recommended_for_small_cohort_viz(self, small_two_cond, capsys):
        recommend_pipeline(
            small_two_cond,
            goal="viz_only",
            data_type="phospho",
            primary_factor="condition",
        )
        out = self._get_output(capsys)
        assert "impute_knn_site_based" in out

    def test_expected_outcome_reports_shape_and_nan(self, factorial_medium, capsys):
        recommend_pipeline(
            factorial_medium,
            goal="primary_de",
            data_type="phospho",
            primary_factor="fiber_type",
            secondary_factor="time_point",
        )
        out = self._get_output(capsys)
        assert "Final matrix" in out
        assert "Missingness" in out
        assert "Interaction OK" in out
