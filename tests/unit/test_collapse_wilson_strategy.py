"""End-to-end tests for the ``localization_strategy="wilson"`` collapse path.

Verifies:
- ``classI_wilson_lb`` is always populated in ``var`` after collapse,
  regardless of which strategy was used (feature is zero-cost hygiene).
- ``strategy="wilson"`` delegates to ``global_max`` at the precursor level
  and then applies the site-level Wilson filter — the two-step recipe
  matches doing the steps manually.
- Validation of the new ``wilson_threshold`` setting (float, "auto",
  or unknown → ValueError).
- The Wilson filter's provenance stamp lands in ``adata.uns["wilson_filter"]``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import alphaphos as ap
from alphaphos.constants import VAR_CLASSI_WILSON_LB
from alphaphos.preprocess.classI_wilson import wilson_lower_bound

# Reuse the synthetic PSM factory from the existing collapse tests
from tests.unit.test_collapse_end_to_end import (
    _make_synthetic_conditions,
    _make_synthetic_psm,
)

# ---------------------------------------------------------------------------
# Always-populated column
# ---------------------------------------------------------------------------


class TestWilsonLbAlwaysPopulated:
    @pytest.mark.parametrize("strategy", ["per_run", "global_max", "condition"])
    def test_column_present_regardless_of_strategy(self, strategy):
        psm = _make_synthetic_psm()
        cond = _make_synthetic_conditions() if strategy == "condition" else None
        adata = ap.collapse_sites(
            psm,
            condition_df=cond,
            advanced={"localization_strategy": strategy},
        )
        assert VAR_CLASSI_WILSON_LB in adata.var.columns
        lb = adata.var[VAR_CLASSI_WILSON_LB].to_numpy()
        assert np.all((lb >= 0.0) & (lb <= 1.0))

    def test_lb_matches_manual_wilson_from_counts(self):
        psm = _make_synthetic_psm()
        adata = ap.collapse_sites(
            psm,
            advanced={"localization_strategy": "global_max"},
        )
        k = adata.var["n_classI_samples"].to_numpy()
        n = adata.var["n_samples_detected"].to_numpy()
        expected = wilson_lower_bound(k, n)
        got = adata.var[VAR_CLASSI_WILSON_LB].to_numpy()
        assert np.allclose(expected, got, atol=1e-9)


# ---------------------------------------------------------------------------
# Wilson strategy end-to-end
# ---------------------------------------------------------------------------


class TestWilsonStrategy:
    def test_wilson_strategy_valid_in_resolve_settings(self):
        settings = ap.resolve_settings({"localization_strategy": "wilson", "wilson_threshold": 0.5})
        assert settings["localization_strategy"] == "wilson"
        assert settings["wilson_threshold"] == 0.5

    def test_wilson_strategy_end_to_end_smaller_than_global_max(self):
        psm = _make_synthetic_psm()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # small-cohort warning on n=4 fixture
            global_max = ap.collapse_sites(
                psm,
                advanced={"localization_strategy": "global_max"},
            )
            wilson = ap.collapse_sites(
                psm,
                advanced={"localization_strategy": "wilson", "wilson_threshold": 0.1},
            )
        # Wilson must always be a strict subset of global_max sites (same
        # precursor mask, then Wilson filter removes some sites).
        assert wilson.n_vars <= global_max.n_vars
        assert set(wilson.var_names).issubset(set(global_max.var_names))

    def test_wilson_strategy_stamps_uns_provenance(self):
        psm = _make_synthetic_psm()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = ap.collapse_sites(
                psm,
                advanced={"localization_strategy": "wilson", "wilson_threshold": 0.2},
            )
        assert "wilson_filter" in adata.uns
        prov = adata.uns["wilson_filter"]
        assert prov["threshold"] == pytest.approx(0.2)
        assert "reason" in prov
        assert prov["n_sites_pre_filter"] >= prov["n_sites_kept"]

    def test_wilson_matches_manual_two_step(self):
        """strategy='wilson' at fixed threshold == global_max + apply_wilson_filter."""
        psm = _make_synthetic_psm()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct = ap.collapse_sites(
                psm,
                advanced={"localization_strategy": "wilson", "wilson_threshold": 0.3},
            )
            manual_gm = ap.collapse_sites(
                psm,
                advanced={"localization_strategy": "global_max"},
            )
            manual = ap.apply_wilson_filter(manual_gm, threshold=0.3)
        assert direct.n_vars == manual.n_vars
        assert list(direct.var_names) == list(manual.var_names)
        assert np.allclose(
            np.nan_to_num(direct.X, nan=-1.0),
            np.nan_to_num(manual.X, nan=-1.0),
            atol=1e-9,
        )


# ---------------------------------------------------------------------------
# wilson_threshold validation via resolve_settings
# ---------------------------------------------------------------------------


class TestWilsonThresholdValidation:
    def test_auto_string_accepted(self):
        s = ap.resolve_settings({"wilson_threshold": "auto"})
        assert s["wilson_threshold"] == "auto"

    def test_float_in_range_accepted(self):
        for t in [0.0, 0.25, 0.5, 0.9999, 1.0]:
            s = ap.resolve_settings({"wilson_threshold": t})
            assert s["wilson_threshold"] == t

    @pytest.mark.parametrize("bad", [-0.01, 1.5, 2.0, -1.0])
    def test_out_of_range_float_raises(self, bad):
        with pytest.raises(ValueError, match=r"wilson_threshold must be in \[0, 1\]"):
            ap.resolve_settings({"wilson_threshold": bad})

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="wilson_threshold string must be 'auto'"):
            ap.resolve_settings({"wilson_threshold": "elbow"})

    def test_none_raises(self):
        with pytest.raises(ValueError, match=r"must be a float or 'auto'"):
            ap.resolve_settings({"wilson_threshold": None})


# ---------------------------------------------------------------------------
# valid strategies list now includes "wilson"
# ---------------------------------------------------------------------------


def test_wilson_in_valid_strategies():
    from alphaphos.preprocess._collapse.masking import VALID_STRATEGIES

    assert "wilson" in VALID_STRATEGIES
