"""Tests for on_off detection + imputed-provenance annotation + observed-only limma.

Motivation (from a collaborator's review): the default
impute-then-limma flow inflates the "down" hit count on small-n groups
by imputing whole groups with the MNAR down-shift, then reporting a
"significant" fold-change driven entirely by invented numbers.  These
tools separate detection (feature seen in one arm) from quantification
(fold-change between two well-observed arms) and flag any imputed-driven
hits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("anndata")
import anndata as ad

from alphaphos.stats.on_off import (
    annotate_imputed_provenance,
    diff_exp_limma_observed_only,
    on_off_detection,
)

try:  # pragma: no cover - checked via skipif below
    import inmoose
    import patsy

    _HAS_LIMMA_DEPS = True
except ImportError:
    _HAS_LIMMA_DEPS = False


def _make_adata(
    *,
    n_treatment: int = 5,
    n_control: int = 5,
    n_features: int = 6,
    obs_pattern: dict[int, tuple[list[bool], list[bool]]] | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Synthesise a small AnnData with a controllable missingness pattern.

    ``obs_pattern`` maps feature-index -> (observed_treatment, observed_control)
    boolean lists.  Cells marked False are set to NaN.  Missing entries default
    to all-observed.
    """
    rng = np.random.default_rng(seed)
    n_samples = n_treatment + n_control
    # Per-feature scale so variance is heterogeneous; homogeneous variance
    # would push inmoose eBayes into its df_prior=inf degenerate branch on
    # synthetic data.
    feature_scale = rng.uniform(0.6, 1.6, size=n_features)
    X = rng.normal(loc=20.0, scale=feature_scale, size=(n_samples, n_features)).astype(float)
    obs_pattern = obs_pattern or {}
    for fi in range(n_features):
        obs_t, obs_c = obs_pattern.get(fi, ([True] * n_treatment, [True] * n_control))
        for si, present in enumerate(obs_t):
            if not present:
                X[si, fi] = np.nan
        for si, present in enumerate(obs_c):
            if not present:
                X[n_treatment + si, fi] = np.nan
    obs = pd.DataFrame(
        {"condition": (["treatment"] * n_treatment) + (["control"] * n_control)},
        index=[f"s{i:02d}" for i in range(n_samples)],
    )
    var = pd.DataFrame(index=[f"f{i:02d}" for i in range(n_features)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["intensity_log2"] = X.copy()
    return adata


class TestOnOffDetection:
    def test_call_partitions_features(self):
        adata = _make_adata(
            n_treatment=5,
            n_control=5,
            n_features=4,
            obs_pattern={
                0: ([True] * 5, [True] * 5),  # both
                1: ([True] * 5, [False] * 5),  # on_in_treatment
                2: ([False] * 5, [True] * 5),  # on_in_control
                3: ([False] * 5, [False] * 5),  # absent
            },
        )
        out = on_off_detection(
            adata,
            condition_column="condition",
            comparison=("treatment", "control"),
            min_observed_per_group=3,
        )
        assert list(out["call"]) == [
            "both",
            "on_in_treatment",
            "on_in_control",
            "absent",
        ]
        assert out.attrs["provenance"]["n_both"] == 1
        assert out.attrs["provenance"]["n_on_in_treatment"] == 1
        assert out.attrs["provenance"]["n_on_in_control"] == 1
        assert out.attrs["provenance"]["n_absent"] == 1

    def test_threshold_boundary(self):
        # Exactly at threshold on both sides -> "both"; one below -> "on_in_treatment"
        adata = _make_adata(
            n_treatment=5,
            n_control=5,
            n_features=2,
            obs_pattern={
                0: (
                    [True, True, True, False, False],
                    [True, True, True, False, False],
                ),  # 3 vs 3 -> both
                1: (
                    [True, True, True, False, False],
                    [True, True, False, False, False],
                ),  # 3 vs 2 -> on_in_treatment
            },
        )
        out = on_off_detection(
            adata,
            condition_column="condition",
            comparison=("treatment", "control"),
            min_observed_per_group=3,
        )
        assert list(out["call"]) == ["both", "on_in_treatment"]
        assert list(out["n_observed_treatment"]) == [3, 3]
        assert list(out["n_observed_control"]) == [3, 2]

    def test_bad_contrast_raises(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="Level"):
            on_off_detection(
                adata,
                condition_column="condition",
                comparison=("does_not_exist", "control"),
            )

    def test_bad_level_raises(self):
        # Contrast level actually absent from the data -- caught by validator.
        adata = _make_adata()
        adata.obs["condition"] = "third"
        with pytest.raises(ValueError, match="not in"):
            on_off_detection(
                adata,
                condition_column="condition",
                comparison=("treatment", "control"),
            )


class TestAnnotateImputedProvenance:
    def test_flag_set_when_group_below_threshold(self):
        adata = _make_adata(
            n_treatment=5,
            n_control=5,
            n_features=3,
            obs_pattern={
                0: ([True] * 5, [True] * 5),  # both fully observed
                1: ([True] * 5, [False] * 5),  # control absent -- imputed-driven
                2: ([True, True, False, False, False], [True] * 5),  # treatment underobserved
            },
        )
        # Fake a limma result -- we don't need it to be real for annotation semantics.
        fake_result = pd.DataFrame(
            {"log2fc": [0.1, 3.5, 1.2], "fdr": [0.5, 1e-8, 1e-4]},
            index=["f00", "f01", "f02"],
        )
        out = annotate_imputed_provenance(
            fake_result,
            adata_pre_imputation=adata,
            condition_column="condition",
            comparison=("treatment", "control"),
            min_observed_per_group=3,
        )
        # f00: 5/5 both -> not flagged
        # f01: 5/0 -> flagged (control below threshold)
        # f02: 2/5 -> flagged (treatment below threshold)
        assert list(out["imputed_driven"]) == [False, True, True]
        assert list(out["n_observed_treatment"]) == [5, 5, 2]
        assert list(out["n_observed_control"]) == [5, 0, 5]
        assert list(out["n_missing_treatment"]) == [0, 0, 3]
        assert list(out["n_missing_control"]) == [0, 5, 0]

    def test_original_result_untouched(self):
        adata = _make_adata()
        fake = pd.DataFrame({"log2fc": [1.0], "fdr": [0.01]}, index=["f00"])
        _ = annotate_imputed_provenance(
            fake,
            adata_pre_imputation=adata,
            condition_column="condition",
            comparison=("treatment", "control"),
        )
        assert list(fake.columns) == ["log2fc", "fdr"]  # original not mutated

    def test_missing_feature_in_adata_raises(self):
        adata = _make_adata()
        fake = pd.DataFrame({"log2fc": [1.0], "fdr": [0.01]}, index=["not_a_feature"])
        with pytest.raises(KeyError, match="not in"):
            annotate_imputed_provenance(
                fake,
                adata_pre_imputation=adata,
                condition_column="condition",
                comparison=("treatment", "control"),
            )


@pytest.mark.skipif(not _HAS_LIMMA_DEPS, reason="inmoose + patsy required for limma")
class TestObservedOnlyLimmaWrapper:
    """End-to-end: the wrapper must exclude on/off features from limma."""

    def test_on_off_features_not_in_limma_result(self):
        adata = _make_adata(
            n_treatment=5,
            n_control=5,
            n_features=5,
            obs_pattern={
                0: ([True] * 5, [True] * 5),
                1: ([True] * 5, [False] * 5),  # on_in_treatment -- must be excluded
                2: ([False] * 5, [True] * 5),  # on_in_control -- must be excluded
                3: ([True] * 5, [True] * 5),
                4: ([False] * 5, [False] * 5),  # absent -- must be excluded
            },
        )
        # Add some signal to features 0 and 3 so limma has non-zero var
        adata.X[:5, 0] += 2.0
        adata.X[:5, 3] -= 2.0
        adata.layers["intensity_log2"] = adata.X.copy()

        result, on_off = diff_exp_limma_observed_only(
            adata,
            condition_column="condition",
            comparison=("treatment", "control"),
            min_observed_per_group=3,
        )
        # On/off features are NEVER in the limma result
        assert set(result.index).isdisjoint({"f01", "f02", "f04"})
        # The "both" features are all present
        assert set(result.index) == {"f00", "f03"}
        # On/off table classifies correctly
        assert on_off.loc["f01", "call"] == "on_in_treatment"
        assert on_off.loc["f02", "call"] == "on_in_control"
        assert on_off.loc["f04", "call"] == "absent"
        # Result carries provenance columns
        assert "imputed_driven" in result.columns
        # By construction the strict filter means no imputed-driven hits
        assert not result["imputed_driven"].any()

    def test_raises_when_nothing_testable(self):
        adata = _make_adata(
            n_treatment=5,
            n_control=5,
            n_features=2,
            obs_pattern={
                0: ([True] * 5, [False] * 5),
                1: ([False] * 5, [True] * 5),
            },
        )
        with pytest.raises(ValueError, match="No features pass"):
            diff_exp_limma_observed_only(
                adata,
                condition_column="condition",
                comparison=("treatment", "control"),
                min_observed_per_group=3,
            )


class TestFilterByCompletenessMinValidN:
    """New min_valid_n mode on filter_by_completeness (reviewer's ask)."""

    def test_min_valid_n_matches_frac_semantics_on_uniform_input(self):
        from alphaphos.preprocess.filter import filter_by_completeness

        adata = _make_adata(n_treatment=5, n_control=5, n_features=3)
        # All features fully observed -> min_valid_n=1 keeps all
        a = filter_by_completeness(adata, min_valid_n=1)
        assert a.n_vars == 3

    def test_min_valid_n_per_group_each(self):
        from alphaphos.preprocess.filter import filter_by_completeness

        adata = _make_adata(
            n_treatment=5,
            n_control=5,
            n_features=3,
            obs_pattern={
                0: ([True] * 5, [True] * 5),  # both fully observed
                1: ([True] * 5, [True, True, False, False, False]),  # control underobs
                2: ([True] * 5, [False] * 5),  # control absent
            },
        )
        # min_valid_n=3 per group, both required
        a = filter_by_completeness(
            adata, min_valid_n=3, group_column="condition", keep_strategy="each"
        )
        assert list(a.var_names) == ["f00"]

    def test_rejects_both_thresholds(self):
        from alphaphos.preprocess.filter import filter_by_completeness

        adata = _make_adata()
        with pytest.raises(ValueError, match="exactly one"):
            filter_by_completeness(adata, min_valid_frac=0.5, min_valid_n=3)

    def test_rejects_neither_threshold(self):
        from alphaphos.preprocess.filter import filter_by_completeness

        adata = _make_adata()
        with pytest.raises(ValueError, match="exactly one"):
            filter_by_completeness(adata)
