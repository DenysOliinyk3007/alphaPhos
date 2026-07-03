"""Unit tests for :mod:`alphaphos.preprocess.batch_correct`.

Fixture: 12 samples x 60 sites split into 2 batches of 6.  A real batch
effect (+3 log2 on all sites in batch 2) is superimposed on a real
condition effect (+2 log2 on the first 10 sites in the 'trt' group).
ComBat should remove the batch effect while preserving the condition
effect when passed ``covariates=['condition']``.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphaphos.constants import (
    LAYER_INTENSITY_LOG2,
    LAYER_INTENSITY_LOG2_PRECOMBAT,
    UNS_ALPHAPHOS,
    UNS_BATCH_CORRECTION,
)
from alphaphos.preprocess.batch_correct import (
    _HAS_COMBAT_DEPS,
    DEFAULT_COMBAT_SETTINGS,
    batch_correct_combat,
)

pytestmark = pytest.mark.skipif(
    not _HAS_COMBAT_DEPS,
    reason="inmoose + patsy not installed; run `pip install alphaPhos[stats]`.",
)


TRUE_HITS = 10
BATCH_SHIFT = 3.0
COND_SHIFT = 2.0


@pytest.fixture
def two_batch_adata():
    rng = np.random.default_rng(0)
    n_samp, n_feat = 12, 60
    per_site_sd = rng.uniform(0.3, 0.8, n_feat)
    X = rng.normal(20.0, per_site_sd[None, :], size=(n_samp, n_feat))

    # Interleaved: half of each batch is ctrl, half is trt.
    obs = pd.DataFrame(
        {
            "batch": ["b1"] * 6 + ["b2"] * 6,
            "condition": (["ctrl"] * 3 + ["trt"] * 3) * 2,
        },
        index=[f"s{i}" for i in range(n_samp)],
    )

    X[obs["batch"].to_numpy() == "b2", :] += BATCH_SHIFT
    X[obs["condition"].to_numpy() == "trt", :TRUE_HITS] += COND_SHIFT

    var = pd.DataFrame(index=[f"site_{i:02d}" for i in range(n_feat)])
    adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    adata.layers[LAYER_INTENSITY_LOG2] = adata.X.copy()
    return adata


class TestBatchRemoval:
    def test_removes_batch_shift(self, two_batch_adata):
        batch_correct_combat(
            two_batch_adata,
            batch_column="batch",
            covariates=["condition"],
        )
        Y = two_batch_adata.layers[LAYER_INTENSITY_LOG2]
        b1_mean = Y[two_batch_adata.obs["batch"] == "b1"].mean()
        b2_mean = Y[two_batch_adata.obs["batch"] == "b2"].mean()
        # Batch effect was +3.0; after ComBat it should be near zero.
        assert abs(b2_mean - b1_mean) < 0.5

    def test_preserves_condition_effect(self, two_batch_adata):
        batch_correct_combat(
            two_batch_adata,
            batch_column="batch",
            covariates=["condition"],
        )
        Y = two_batch_adata.layers[LAYER_INTENSITY_LOG2]
        trt_mask = two_batch_adata.obs["condition"].to_numpy() == "trt"
        ctrl_mask = ~trt_mask
        # On the spiked sites, trt - ctrl should still be close to +2.0.
        delta_spiked = Y[trt_mask, :TRUE_HITS].mean() - Y[ctrl_mask, :TRUE_HITS].mean()
        assert abs(delta_spiked - COND_SHIFT) < 0.5
        # On unspiked sites, no condition difference expected.
        delta_unspiked = Y[trt_mask, TRUE_HITS:].mean() - Y[ctrl_mask, TRUE_HITS:].mean()
        assert abs(delta_unspiked) < 0.5

    def test_without_covariates_still_removes_batch(self, two_batch_adata):
        # ComBat with no covariates should still remove the batch mean shift;
        # what it CAN'T do is guarantee biological variance is preserved.
        batch_correct_combat(two_batch_adata, batch_column="batch")
        Y = two_batch_adata.layers[LAYER_INTENSITY_LOG2]
        b1_mean = Y[two_batch_adata.obs["batch"] == "b1"].mean()
        b2_mean = Y[two_batch_adata.obs["batch"] == "b2"].mean()
        assert abs(b2_mean - b1_mean) < 0.5


class TestPrecombatLayer:
    def test_default_keeps_precombat(self, two_batch_adata):
        source = two_batch_adata.layers[LAYER_INTENSITY_LOG2].copy()
        batch_correct_combat(two_batch_adata, batch_column="batch", covariates=["condition"])
        assert LAYER_INTENSITY_LOG2_PRECOMBAT in two_batch_adata.layers
        np.testing.assert_allclose(
            two_batch_adata.layers[LAYER_INTENSITY_LOG2_PRECOMBAT], source, rtol=0
        )
        # And the target layer is DIFFERENT
        assert not np.allclose(
            two_batch_adata.layers[LAYER_INTENSITY_LOG2],
            two_batch_adata.layers[LAYER_INTENSITY_LOG2_PRECOMBAT],
        )

    def test_can_disable_precombat(self, two_batch_adata):
        batch_correct_combat(
            two_batch_adata,
            batch_column="batch",
            covariates=["condition"],
            keep_precombat=False,
        )
        assert LAYER_INTENSITY_LOG2_PRECOMBAT not in two_batch_adata.layers


class TestProvenanceStamp:
    def test_stamp_present_and_complete(self, two_batch_adata):
        batch_correct_combat(
            two_batch_adata,
            batch_column="batch",
            covariates=["condition"],
        )
        stamp = two_batch_adata.uns[UNS_ALPHAPHOS][UNS_BATCH_CORRECTION]
        assert stamp["method"] == "combat"
        assert stamp["batch_column"] == "batch"
        assert stamp["covariates"] == ["condition"]
        assert stamp["layer"] == LAYER_INTENSITY_LOG2
        assert stamp["kept_precombat_layer"] is True
        assert stamp["n_batches"] == 2
        assert stamp["batch_sizes"] == {"b1": 6, "b2": 6}


class TestCopyBehavior:
    def test_copy_true_returns_and_leaves_input_untouched(self, two_batch_adata):
        source = two_batch_adata.layers[LAYER_INTENSITY_LOG2].copy()
        result = batch_correct_combat(
            two_batch_adata,
            batch_column="batch",
            covariates=["condition"],
            copy=True,
        )
        assert result is not None
        # Input is unchanged
        np.testing.assert_allclose(two_batch_adata.layers[LAYER_INTENSITY_LOG2], source, rtol=0)
        # Returned copy is corrected
        assert not np.allclose(result.layers[LAYER_INTENSITY_LOG2], source)


class TestReturnContract:
    """Must ALWAYS return the AnnData so
    ``adata = ap.batch_correct_combat(adata, batch_column=...)`` is safe."""

    def test_returns_same_object_in_place(self, two_batch_adata):
        result = batch_correct_combat(
            two_batch_adata, batch_column="batch", covariates=["condition"]
        )
        assert result is two_batch_adata

    def test_returns_copy_when_copy_true(self, two_batch_adata):
        result = batch_correct_combat(
            two_batch_adata, batch_column="batch", covariates=["condition"], copy=True
        )
        assert result is not two_batch_adata

    def test_assignment_pattern(self, two_batch_adata):
        adata = batch_correct_combat(
            two_batch_adata, batch_column="batch", covariates=["condition"]
        )
        assert adata is not None
        assert LAYER_INTENSITY_LOG2 in adata.layers


class TestValidation:
    def test_missing_batch_column(self, two_batch_adata):
        with pytest.raises(KeyError, match="batch_column"):
            batch_correct_combat(two_batch_adata, batch_column="nope")

    def test_single_batch_rejected(self, two_batch_adata):
        two_batch_adata.obs["batch"] = "only_one"
        with pytest.raises(ValueError, match=">=2 levels"):
            batch_correct_combat(two_batch_adata, batch_column="batch")

    def test_singleton_batch_rejected(self, two_batch_adata):
        # Move one sample into a batch of its own -- ComBat can't fit that.
        two_batch_adata.obs["batch"] = ["b1"] * 5 + ["b2"] * 6 + ["b3"]
        with pytest.raises(ValueError, match=">=2 samples"):
            batch_correct_combat(two_batch_adata, batch_column="batch")

    def test_covariate_equal_to_batch_rejected(self, two_batch_adata):
        with pytest.raises(ValueError, match="cannot self-adjust"):
            batch_correct_combat(
                two_batch_adata,
                batch_column="batch",
                covariates=["batch"],
            )

    def test_missing_covariate_rejected(self, two_batch_adata):
        with pytest.raises(KeyError, match="not in adata.obs"):
            batch_correct_combat(
                two_batch_adata,
                batch_column="batch",
                covariates=["nope"],
            )

    def test_bad_ref_batch_rejected(self, two_batch_adata):
        with pytest.raises(ValueError, match="ref_batch"):
            batch_correct_combat(
                two_batch_adata,
                batch_column="batch",
                advanced={"ref_batch": "b3"},
            )

    def test_nan_rejected(self, two_batch_adata):
        two_batch_adata.layers[LAYER_INTENSITY_LOG2][0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            batch_correct_combat(two_batch_adata, batch_column="batch")

    def test_linear_scale_rejected(self, two_batch_adata):
        two_batch_adata.layers[LAYER_INTENSITY_LOG2] = (
            2.0 ** two_batch_adata.layers[LAYER_INTENSITY_LOG2]
        )
        with pytest.raises(ValueError, match="linear-scale"):
            batch_correct_combat(two_batch_adata, batch_column="batch")

    def test_duplicate_var_names_rejected(self, two_batch_adata):
        two_batch_adata.var_names = ["site_0"] * two_batch_adata.n_vars
        with pytest.raises(ValueError, match="unique"):
            batch_correct_combat(two_batch_adata, batch_column="batch")

    def test_missing_layer_rejected(self, two_batch_adata):
        with pytest.raises(KeyError, match="not in adata.layers"):
            batch_correct_combat(
                two_batch_adata,
                batch_column="batch",
                layer="does_not_exist",
            )

    def test_unknown_advanced_key_rejected(self, two_batch_adata):
        with pytest.raises(ValueError, match="Unknown advanced keys"):
            batch_correct_combat(
                two_batch_adata,
                batch_column="batch",
                advanced={"bogus_flag": True},
            )


class TestDoubleCorrectionGuard:
    def test_refuses_batch_covariate_after_combat(self, two_batch_adata):
        from alphaphos.stats.diff_exp import diff_exp_limma

        batch_correct_combat(two_batch_adata, batch_column="batch", covariates=["condition"])
        with pytest.raises(ValueError, match="Double batch correction"):
            diff_exp_limma(
                two_batch_adata,
                condition_column="condition",
                comparison=("trt", "ctrl"),
                covariates=["batch"],
            )

    def test_permits_batch_covariate_on_precombat_layer(self, two_batch_adata):
        # The community-recommended path: use the pre-correction layer with a
        # batch covariate. Guard MUST NOT trigger.
        from alphaphos.stats.diff_exp import diff_exp_limma

        batch_correct_combat(two_batch_adata, batch_column="batch", covariates=["condition"])
        res = diff_exp_limma(
            two_batch_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
            covariates=["batch"],
            layer=LAYER_INTENSITY_LOG2_PRECOMBAT,
        )
        assert len(res) == two_batch_adata.n_vars

    def test_permits_combat_layer_without_batch_covariate(self, two_batch_adata):
        # Path B: run stats on the corrected layer WITHOUT a batch covariate.
        from alphaphos.stats.diff_exp import diff_exp_limma

        batch_correct_combat(two_batch_adata, batch_column="batch", covariates=["condition"])
        res = diff_exp_limma(
            two_batch_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
        )
        assert len(res) == two_batch_adata.n_vars


class TestSettings:
    def test_default_settings_shape(self):
        assert set(DEFAULT_COMBAT_SETTINGS) == {"par_prior", "mean_only", "ref_batch"}
        assert DEFAULT_COMBAT_SETTINGS["par_prior"] is True
        assert DEFAULT_COMBAT_SETTINGS["mean_only"] is False
        assert DEFAULT_COMBAT_SETTINGS["ref_batch"] is None
