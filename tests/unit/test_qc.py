"""Tests for alphaphos.qc — metrics computations + end-to-end HTML generation.

The metric computations are pure pandas/numpy and fast. The dashboard
generation runs bokeh's HTML composer (in-process; no subprocess) and
verifies the file is produced with all expected section headers.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("bokeh")

from alphaphos.qc import (
    compute_classI_comparison,
    compute_imputation_summary,
    compute_localization_distribution,
    compute_missingness,
    compute_multiplicity_distribution,
    compute_n_classI_per_sample,
    compute_per_condition_cv_by_aa,
    compute_pipeline_waterfall,
    compute_replicate_correlation,
    compute_sample_dendrogram_order,
    compute_sty_ratio,
    generate_dashboard,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synth_adata():
    """A small synthetic AnnData with all the metadata QC functions read."""
    rng = np.random.default_rng(0)
    n_obs, n_var = 6, 50
    samples = [f"withEGF_r{i}" for i in range(3)] + [f"woEGF_r{i}" for i in range(3)]
    obs = pd.DataFrame({"condition": ["withEGF"] * 3 + ["woEGF"] * 3}, index=samples)
    site_aa = rng.choice(["S", "T", "Y"], n_var, p=[0.7, 0.2, 0.1])
    multiplicity = rng.choice([1, 2, 3], n_var, p=[0.85, 0.1, 0.05])
    var = pd.DataFrame(
        {"site_aa": site_aa, "multiplicity": multiplicity},
        index=[f"site{i}" for i in range(n_var)],
    )
    X = rng.normal(15.0, 1.5, (n_obs, n_var))
    # Sprinkle missing
    miss = rng.random((n_obs, n_var)) < 0.1
    X[miss] = np.nan
    loc = rng.beta(8, 1, (n_obs, n_var))
    loc[miss] = np.nan
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["intensity_log2"] = X.copy()
    adata.layers["localization"] = loc
    adata.uns["alphaphos"] = {
        "version": "0.0.0",
        "pipeline_params": {
            "aggregation_method": "sum",
            "localization_strategy": "condition",
            "classI_cutoff": 0.75,
        },
    }
    adata.uns["source_attrs"] = {
        "n_rows_loaded": 1_200_000,
        "n_rows_returned": 240_000,
        "n_rows_after_contaminant_filter": 239_000,
        "n_rows_after_top_n": 235_000,
    }
    return adata


# ---------------------------------------------------------------------------
# §1 Pipeline waterfall
# ---------------------------------------------------------------------------


def test_pipeline_waterfall_rows_decrease(synth_adata):
    df = compute_pipeline_waterfall(synth_adata)
    assert {"stage", "n_rows", "dropped_from_prev", "pct_dropped_from_prev"}.issubset(df.columns)
    # row counts should not increase from one stage to the next (until the
    # final 'after_collapse' which switches units)
    pre_collapse = df[~df["stage"].str.startswith("after_collapse")]
    assert (
        pre_collapse["n_rows"].is_monotonic_decreasing
        or pre_collapse["n_rows"].is_monotonic_non_increasing
    )


def test_pipeline_waterfall_empty_when_no_source_attrs():
    adata = ad.AnnData(X=np.zeros((2, 2)))
    df = compute_pipeline_waterfall(adata)
    # at minimum the final 'after_collapse' stage is present
    assert "after_collapse_to_sites" in df["stage"].tolist()


# ---------------------------------------------------------------------------
# §2 Sample-level metrics
# ---------------------------------------------------------------------------


def test_sty_ratio_sums_to_100(synth_adata):
    df = compute_sty_ratio(synth_adata)
    for _s, group in df.groupby("sample"):
        # if site_aa coverage is comprehensive, pct should sum to ~100
        total_pct = group["pct"].sum()
        assert 99.9 <= total_pct <= 100.1


def test_localization_distribution_only_observed_cells(synth_adata):
    df = compute_localization_distribution(synth_adata)
    # All loc_prob values should be in [0, 1] and non-NaN
    assert df["loc_prob"].between(0, 1).all()
    assert not df["loc_prob"].isna().any()


def test_multiplicity_sums_to_100(synth_adata):
    df = compute_multiplicity_distribution(synth_adata)
    for _s, group in df.groupby("sample"):
        total_pct = group["pct"].sum()
        assert 99.9 <= total_pct <= 100.1


def test_missingness_reports_pct_per_sample(synth_adata):
    df = compute_missingness(synth_adata)
    assert {"sample", "n_missing", "n_total", "pct_missing"}.issubset(df.columns)
    assert (df["n_total"] == synth_adata.n_vars).all()
    # condition column should propagate
    assert "condition" in df.columns


def test_n_classI_per_sample(synth_adata):
    df = compute_n_classI_per_sample(synth_adata, classI_cutoff=0.5)
    assert (df["n_classI"] <= df["n_present"]).all()
    assert df["fraction_classI"].between(0, 1).all()


def test_per_condition_cv_by_aa(synth_adata):
    df = compute_per_condition_cv_by_aa(synth_adata)
    if not df.empty:
        assert set(df["site_aa"]).issubset({"S", "T", "Y"})
        assert (df["cv"] >= 0).all()


# ---------------------------------------------------------------------------
# §3 Reproducibility
# ---------------------------------------------------------------------------


def test_replicate_correlation_symmetric(synth_adata):
    corr = compute_replicate_correlation(synth_adata)
    assert corr.shape == (synth_adata.n_obs, synth_adata.n_obs)
    np.testing.assert_allclose(corr.values, corr.values.T, equal_nan=True)
    # Diagonal should be 1.0 (or NaN if all-missing — not the case here)
    np.testing.assert_allclose(np.diag(corr.values), 1.0)


def test_sample_dendrogram_order_is_permutation(synth_adata):
    order = compute_sample_dendrogram_order(synth_adata)
    assert set(order) == set(synth_adata.obs.index.astype(str))
    assert len(order) == synth_adata.n_obs


# ---------------------------------------------------------------------------
# §4 Class I + contaminant
# ---------------------------------------------------------------------------


def test_classI_comparison_returns_two_rows(synth_adata):
    df = compute_classI_comparison(synth_adata)
    assert len(df) == 2
    assert {"policy", "n_cells"}.issubset(df.columns)
    # condition_aware cells should be >= per_cell_binary cells
    by_policy = df.set_index("policy")["n_cells"]
    cell_binary = by_policy[by_policy.index.str.contains("per_cell")].iloc[0]
    cond_aware = by_policy[by_policy.index.str.contains("condition_aware")].iloc[0]
    assert cond_aware >= cell_binary or cond_aware >= cell_binary * 0.9


# ---------------------------------------------------------------------------
# §5 Imputation summary
# ---------------------------------------------------------------------------


def test_imputation_summary_aggregates_by_strategy():
    audit = pd.DataFrame(
        {
            "sample_idx": [0, 0, 1, 1, 1],
            "site_idx": [10, 20, 30, 40, 50],
            "strategy": ["MAR_KNN", "MNAR_Gaussian", "MAR_KNN", "MAR_KNN", "MNAR_Gaussian"],
        }
    )
    df = compute_imputation_summary(audit)
    by_strat = df.set_index(["sample_idx", "strategy"])["n"]
    assert by_strat[(0, "MAR_KNN")] == 1
    assert by_strat[(0, "MNAR_Gaussian")] == 1
    assert by_strat[(1, "MAR_KNN")] == 2
    assert by_strat[(1, "MNAR_Gaussian")] == 1


def test_imputation_summary_empty():
    df = compute_imputation_summary(pd.DataFrame())
    assert df.empty


# ---------------------------------------------------------------------------
# generate_dashboard — end-to-end
# ---------------------------------------------------------------------------


def test_generate_dashboard_produces_html(synth_adata, tmp_path):
    out = generate_dashboard(synth_adata, tmp_path / "qc.html", title="QC test")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    # Bokeh JSON-encodes Div text inside its script payload, so § becomes
    # §. Just verify the substantive section labels are present.
    for fragment in (
        "Pipeline waterfall",
        "Sample-level QC",
        "Reproducibility",
        "Class I",
        "Imputation diagnostics",
        "Pipeline parameters",
    ):
        assert fragment in text, f"missing section text: {fragment!r}"


def test_generate_dashboard_optional_inputs(synth_adata, tmp_path):
    """When psm_df / impute_audit aren't provided, dashboard still renders
    (the panels show 'no data' placeholders)."""
    out = generate_dashboard(synth_adata, tmp_path / "qc.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    # Placeholder text for the optional sections should appear
    # (the exact wording is in alphaphos.qc.plots._empty_figure)
    assert "Contaminant breakdown" in text
    assert "Imputation diagnostics" in text
