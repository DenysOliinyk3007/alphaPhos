"""Tests for the phospho-specific metadata that alphaPhos writes into AnnData.

Covers:
  - _derive_motif_flags: kinase_sequence -> motif flags (p_minus_1, p_plus_1,
                       is_proline_directed, is_basophilic, is_acidic_motif)
  - _compute_site_qc:    loc_per_run    -> site QC columns
  - to_anndata end-to-end: motif flags + site QC + provenance land where
                       expected on the AnnData object
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from alphaphos import __version__ as ALPHAPHOS_VERSION
from alphaphos.preprocess.anndata import (
    _compute_site_qc,
    _derive_motif_flags,
    to_anndata,
)

# ============================================================================
# _derive_motif_flags
# ============================================================================


class TestDeriveMotifFlags:
    @pytest.mark.parametrize(
        ("window", "expected"),
        [
            # Middle of the protein, full 7+7 window
            (
                "_AAVKRGT*S*ELLIQAA_",
                {
                    "p_minus_1": "T",
                    "p_plus_1": "E",
                    "is_proline_directed": False,
                    "is_basophilic": True,  # R at -3
                    "is_acidic_motif": True,  # E at +1 (CK1-like)
                },
            ),
            # +1 = P -> proline-directed
            (
                "_AAVKRGT*S*PLLIQAA_",
                {"p_plus_1": "P", "is_proline_directed": True},
            ),
            # R-X-X-S (PKA-like) -> basophilic
            (
                "_AAVKRRX*S*YLLIQAA_",
                {"p_minus_1": "X", "is_basophilic": True},
            ),
            # No R/K in upstream window -> not basophilic
            (
                "_AAVDET*S*YLLIQAA_",
                {"is_basophilic": False},
            ),
            # +1 = D -> CK1-like acidic
            (
                "_AAVKRGT*S*DLLIQAA_",
                {"p_plus_1": "D", "is_acidic_motif": True},
            ),
            # +3 = E -> CK2-like acidic
            (
                "_AAVKRGT*S*LLELIQAA_",
                {"is_acidic_motif": True},
            ),
            # Nothing interesting anywhere
            (
                "_GGGGGGG*S*LLLLLLL_",
                {
                    "p_minus_1": "G",
                    "p_plus_1": "L",
                    "is_proline_directed": False,
                    "is_basophilic": False,
                    "is_acidic_motif": False,
                },
            ),
        ],
    )
    def test_motif_flag_matrix(self, window, expected):
        out = _derive_motif_flags(window)
        for key, val in expected.items():
            assert out[key] == val, f"{window!r}: {key} was {out[key]!r}, expected {val!r}"

    def test_padded_left_terminus(self):
        out = _derive_motif_flags("_AT*S*ELLIQAA_")
        assert out["p_minus_1"] == "T"
        assert out["is_basophilic"] is False

    def test_padded_right_terminus(self):
        out = _derive_motif_flags("_AAVKRGT*S*EL_")
        assert out["p_plus_1"] == "E"
        assert out["is_acidic_motif"] is True  # +1 = E, still CK1-like

    def test_empty_left_flank(self):
        out = _derive_motif_flags("_*S*ELLIQAA_")
        assert out["p_minus_1"] is None
        assert out["p_plus_1"] == "E"
        assert out["is_basophilic"] is False

    def test_empty_right_flank(self):
        out = _derive_motif_flags("_AAVKRGT*S*_")
        assert out["p_minus_1"] == "T"
        assert out["p_plus_1"] is None
        assert out["is_proline_directed"] is False
        assert out["is_acidic_motif"] is False

    def test_error_sentinel_strings(self):
        """PeptideCollapse error sentinels should produce empty flags."""
        for err in (
            "FASTA_ERROR: missing",
            "POSITION_ERROR: out of bounds",
            "SEQUENCE_MISMATCH: expected S found T",
            "PARSING_ERROR: bad row",
        ):
            out = _derive_motif_flags(err)
            assert all(v is None for v in out.values()), err


# ============================================================================
# _compute_site_qc
# ============================================================================


class TestComputeSiteQC:
    def test_all_metrics(self):
        # 3 sites x 4 samples
        loc = pd.DataFrame(
            [
                [0.9, 0.8, 0.95, 0.85],  # all class I
                [0.7, 0.95, 0.5, np.nan],  # mixed; one missing
                [0.4, 0.3, 0.5, 0.6],  # none class I (cutoff 0.75)
            ],
            index=["site1", "site2", "site3"],
            columns=["s1", "s2", "s3", "s4"],
        )
        qc = _compute_site_qc(loc, loc.index, list(loc.columns), classI_cutoff=0.75)

        assert qc.loc["site1", "n_samples_detected"] == 4
        assert qc.loc["site2", "n_samples_detected"] == 3
        assert qc.loc["site3", "n_samples_detected"] == 4

        assert qc.loc["site1", "n_classI_samples"] == 4
        # site2 = [0.7, 0.95, 0.5, NaN]; only 0.95 ≥ 0.75
        assert qc.loc["site2", "n_classI_samples"] == 1
        assert qc.loc["site3", "n_classI_samples"] == 0

        assert qc.loc["site1", "fraction_classI"] == 1.0
        assert qc.loc["site2", "fraction_classI"] == 0.25  # 1/4
        assert qc.loc["site3", "fraction_classI"] == 0.0

        assert qc.loc["site1", "max_loc_prob"] == 0.95
        assert qc.loc["site1", "min_loc_prob"] == 0.8
        assert abs(qc.loc["site1", "mean_loc_prob"] - 0.875) < 1e-9

    def test_classI_cutoff_threshold(self):
        loc = pd.DataFrame([[0.75]], index=["s"], columns=["sample"])
        qc = _compute_site_qc(loc, loc.index, ["sample"], classI_cutoff=0.75)
        # ≥0.75 → counts as class I
        assert qc.loc["s", "n_classI_samples"] == 1

    def test_realigns_to_requested_index(self):
        """Sites not in loc_per_run get NaN metrics gracefully."""
        loc = pd.DataFrame([[0.9, 0.8]], index=["site1"], columns=["s1", "s2"])
        qc = _compute_site_qc(
            loc, pd.Index(["site1", "site_missing"]), ["s1", "s2"], classI_cutoff=0.75
        )
        assert qc.loc["site1", "n_samples_detected"] == 2
        # Missing site reindexed → all NaN → counts are 0
        assert qc.loc["site_missing", "n_samples_detected"] == 0
        assert qc.loc["site_missing", "n_classI_samples"] == 0


# ============================================================================
# to_anndata end-to-end: (sites x samples) matrix with `|` keys -> AnnData
# ============================================================================

_KEYS = ["P12345|GENE1|S100|M1", "P12345|GENE1|T200|M1", "Q67890|GENE2|Y50|M2"]


@pytest.fixture
def mini_sites():
    """A tiny 3-site x 2-sample log2 matrix with current alphaPhos site keys."""
    df = pd.DataFrame(
        {"sample_A": [10.0, 12.0, 9.5], "sample_B": [11.0, 12.5, np.nan]},
        index=_KEYS,
    )
    df.attrs["alphaphos_pipeline"] = {
        "aggregation_method": "sum",
        "localization_strategy": "condition",
    }
    df.attrs["source_path"] = "external.tsv"
    return df


@pytest.fixture
def mini_var_meta():
    return pd.DataFrame(
        {
            "kinase_sequence": [
                "_AAVKRGT*S*ELLIQAA_",  # site1: +1 = E -> not proline-directed
                "_AAVKRGT*T*PLLIQAA_",  # site2: +1 = P -> proline-directed
                "FASTA_ERROR: missing",  # site3: error sentinel -> None flags
            ]
        },
        index=_KEYS,
    )


@pytest.fixture
def mini_loc():
    return pd.DataFrame(
        [
            [0.9, 0.8],  # site1: both classI
            [0.5, 0.4],  # site2: neither
            [0.95, np.nan],  # site3: mixed, with NaN
        ],
        index=_KEYS,
        columns=["sample_A", "sample_B"],
    )


class TestToAnnData:
    def test_shape_layers_and_index_names(self, mini_sites):
        adata = to_anndata(mini_sites)
        assert adata.shape == (2, 3)
        assert list(adata.obs_names) == ["sample_A", "sample_B"]
        assert list(adata.var_names) == _KEYS
        assert adata.var_names.name == "full_key"
        assert adata.obs_names.name == "sample"
        np.testing.assert_array_equal(adata.X, adata.layers["intensity_log2"])
        assert np.isnan(adata.X[1, 2])  # transposed correctly

    def test_key_components_parsed_into_var(self, mini_sites):
        v = to_anndata(mini_sites).var
        row = v.loc["Q67890|GENE2|Y50|M2"]
        assert row["protein_group_id"] == "Q67890"
        assert row["gene"] == "GENE2"
        assert row["site_aa"] == "Y"
        assert row["site_position"] == 50
        assert row["multiplicity"] == 2
        assert row["short_key"] == "GENE2|Y50|M2"
        assert row["pg_key"] == "Q67890|Y50|M2"

    def test_unparseable_key_warns_and_yields_nan(self, caplog):
        df = pd.DataFrame({"s": [1.0, 2.0]}, index=["P1|G1|S10|M1", "legacy~G_S1_M1"])
        with caplog.at_level(logging.WARNING, logger="alphaphos.preprocess.anndata"):
            adata = to_anndata(df)
        assert pd.isna(adata.var.loc["legacy~G_S1_M1", "protein_group_id"])
        assert adata.var.loc["P1|G1|S10|M1", "protein_group_id"] == "P1"
        assert any("do not match" in r.message for r in caplog.records)

    def test_motif_flags_from_var_meta(self, mini_sites, mini_var_meta):
        v = to_anndata(mini_sites, var_meta=mini_var_meta).var
        assert v.loc[_KEYS[0], "kinase_sequence"] == "_AAVKRGT*S*ELLIQAA_"
        assert v.loc[_KEYS[0], "is_proline_directed"] is False
        assert v.loc[_KEYS[1], "is_proline_directed"] is True
        assert v.loc[_KEYS[2], "is_proline_directed"] is None

    def test_motif_flags_absent_without_kinase_sequence(self, mini_sites):
        v = to_anndata(mini_sites).var
        assert "p_minus_1" not in v.columns
        assert "is_proline_directed" not in v.columns

    def test_var_meta_clash_raises(self, mini_sites):
        with pytest.raises(ValueError, match="clash"):
            to_anndata(mini_sites, var_meta=pd.DataFrame({"gene": ["x"] * 3}, index=_KEYS))

    def test_site_qc_and_localization_layer(self, mini_sites, mini_loc):
        adata = to_anndata(mini_sites, loc_per_run=mini_loc, classI_cutoff=0.75)
        v = adata.var
        assert v.loc[_KEYS[0], "n_classI_samples"] == 2
        assert v.loc[_KEYS[0], "fraction_classI"] == 1.0
        assert v.loc[_KEYS[1], "n_classI_samples"] == 0
        assert v.loc[_KEYS[2], "n_samples_detected"] == 1
        assert "classI_wilson_lb" in v.columns
        assert adata.layers["localization"].shape == adata.X.shape
        assert np.isnan(adata.layers["localization"][1, 2])

    def test_site_qc_absent_without_loc(self, mini_sites):
        adata = to_anndata(mini_sites)
        assert "n_classI_samples" not in adata.var.columns
        assert "localization" not in adata.layers

    def test_condition_df_joined_with_dedup_and_str_cast(self, mini_sites):
        cdf = pd.DataFrame(
            {
                "sample": ["sample_A", "sample_A", "sample_B"],
                "condition": ["ctrl", "ctrl", "trt"],
                "batch": [1, 1, 2],
            }
        )
        adata = to_anndata(mini_sites, condition_df=cdf)
        assert adata.n_obs == 2
        assert list(adata.obs["condition"]) == ["ctrl", "trt"]
        assert "batch" in adata.obs.columns

    def test_condition_df_missing_columns_raises(self, mini_sites):
        with pytest.raises(ValueError, match="condition"):
            to_anndata(mini_sites, condition_df=pd.DataFrame({"sample": ["sample_A"]}))

    def test_provenance_in_uns(self, mini_sites):
        u = to_anndata(mini_sites).uns
        a = u["alphaphos"]
        assert a["version"] == ALPHAPHOS_VERSION
        assert a["n_sites"] == 3
        assert a["n_samples"] == 2
        assert a["pipeline_params"]["aggregation_method"] == "sum"
        assert "T" in a["processing_timestamp"]
        # non-pipeline attrs land in source_attrs
        assert u["source_attrs"] == {"source_path": "external.tsv"}

    def test_explicit_pipeline_params_and_psm_attrs_win(self, mini_sites):
        u = to_anndata(mini_sites, pipeline_params={"x": 1}, psm_attrs={"n_rows_loaded": 5}).uns
        assert u["alphaphos"]["pipeline_params"] == {"x": 1}
        assert u["source_attrs"] == {"n_rows_loaded": 5}

    def test_provenance_handles_missing_attrs(self):
        adata = to_anndata(pd.DataFrame({"s": [1.0]}, index=["P1|G|S1|M1"]))
        assert adata.uns["alphaphos"]["pipeline_params"] == {}
        assert "source_attrs" not in adata.uns

    @pytest.mark.parametrize(
        ("sites", "match"),
        [
            (pd.DataFrame({"s": [1.0], "gene": ["G"]}, index=["P1|G|S1|M1"]), "numeric"),
            (pd.DataFrame({"s": [1.0, 2.0]}, index=["P1|G|S1|M1", "P1|G|S1|M1"]), "unique"),
            (pd.DataFrame({"s": []}), "non-empty"),
        ],
    )
    def test_rejects_bad_input(self, sites, match):
        with pytest.raises(ValueError, match=match):
            to_anndata(sites)
