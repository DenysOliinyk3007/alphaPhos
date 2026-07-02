"""Tests for the phospho-specific metadata that alphaPhos writes into AnnData.

Covers:
  - _derive_motif_flags: kinase_sequence -> motif flags (p_minus_1, p_plus_1,
                       is_proline_directed, is_basophilic, is_acidic_motif)
  - _compute_site_qc:    loc_per_run    -> site QC columns
  - to_anndata end-to-end: motif flags + site QC + provenance land where
                       expected on the AnnData object
"""

from __future__ import annotations

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
                    "is_basophilic": True,     # R at -3
                    "is_acidic_motif": True,   # E at +1 (CK1-like)
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
# to_anndata end-to-end (Tier 1 + Tier 2 + Tier 4)
# ============================================================================


@pytest.fixture
def mini_sites():
    """A tiny 2-sample × 3-site collapse output for to_anndata tests."""
    df = pd.DataFrame(
        {
            "PTM_Collapse_key": [
                "P12345~GENE1_S100_M1",
                "P12345~GENE1_T200_M1",
                "Q67890~GENE2_Y50_M2",
            ],
            "kinase_sequence": [
                "_AAVKRGT*S*ELLIQAA_",  # site1: proline+1? no (E)
                "_AAVKRGT*T*PLLIQAA_",  # site2: proline-directed
                "FASTA_ERROR: missing",  # site3: error sentinel
            ],
            "sample_A": [10.0, 12.0, 9.5],
            "sample_B": [11.0, 12.5, 10.0],
        }
    )
    df.attrs["alphaphos_pipeline"] = {
        "aggregation_method": "sum",
        "localization_strategy": "condition",
        "fasta_path": "/path/to/proteome.fasta",
    }
    return df


@pytest.fixture
def mini_loc():
    return pd.DataFrame(
        [
            [0.9, 0.8],  # site1: both classI
            [0.5, 0.4],  # site2: neither
            [0.95, np.nan],  # site3: mixed, with NaN
        ],
        index=[
            "P12345~GENE1_S100_M1",
            "P12345~GENE1_T200_M1",
            "Q67890~GENE2_Y50_M2",
        ],
        columns=["sample_A", "sample_B"],
    )


class TestToAnnDataMetadata:
    def test_motif_flags_in_var(self, mini_sites):
        adata = to_anndata(mini_sites)
        assert "p_minus_1" in adata.var.columns
        assert "p_plus_1" in adata.var.columns
        assert "is_proline_directed" in adata.var.columns
        # site1: S, +1 = E -> not proline-directed
        # site2: T, +1 = P -> proline-directed
        # site3: error sentinel -> None
        v = adata.var
        assert v.loc["P12345~GENE1_S100_M1", "is_proline_directed"] is False
        assert v.loc["P12345~GENE1_T200_M1", "is_proline_directed"] is True
        assert v.loc["Q67890~GENE2_Y50_M2", "is_proline_directed"] is None

    def test_site_qc_in_var(self, mini_sites, mini_loc):
        adata = to_anndata(mini_sites, loc_per_run=mini_loc, classI_cutoff=0.75)
        v = adata.var
        # site1: both samples ≥ 0.75
        assert v.loc["P12345~GENE1_S100_M1", "n_classI_samples"] == 2
        assert v.loc["P12345~GENE1_S100_M1", "fraction_classI"] == 1.0
        # site2: neither
        assert v.loc["P12345~GENE1_T200_M1", "n_classI_samples"] == 0
        # site3: one valid, one NaN
        assert v.loc["Q67890~GENE2_Y50_M2", "n_samples_detected"] == 1
        assert v.loc["Q67890~GENE2_Y50_M2", "n_classI_samples"] == 1

    def test_site_qc_absent_without_loc(self, mini_sites):
        adata = to_anndata(mini_sites)
        # No loc_per_run -> QC columns NOT added
        assert "n_classI_samples" not in adata.var.columns
        assert "mean_loc_prob" not in adata.var.columns

    def test_provenance_in_uns(self, mini_sites):
        adata = to_anndata(mini_sites)
        u = adata.uns["alphaphos"]
        assert u["version"] == ALPHAPHOS_VERSION
        assert u["n_sites"] == 3
        assert u["n_samples"] == 2
        # Pipeline params lifted from sites.attrs
        assert u["pipeline_params"]["aggregation_method"] == "sum"
        assert u["pipeline_params"]["localization_strategy"] == "condition"
        assert u["pipeline_params"]["fasta_path"] == "/path/to/proteome.fasta"
        # ISO timestamp string
        assert isinstance(u["processing_timestamp"], str)
        assert "T" in u["processing_timestamp"]

    def test_provenance_handles_missing_pipeline_attrs(self):
        """When sites.attrs is empty, uns['alphaphos'] still works."""
        df = pd.DataFrame(
            {
                "PTM_Collapse_key": ["P12345~G_S100_M1"],
                "sample_A": [10.0],
            }
        )
        adata = to_anndata(df)
        u = adata.uns["alphaphos"]
        assert u["pipeline_params"] == {}
        assert u["version"] == ALPHAPHOS_VERSION

    def test_motif_flags_absent_without_kinase_sequence(self):
        df = pd.DataFrame(
            {
                "PTM_Collapse_key": ["P12345~G_S100_M1"],
                "sample_A": [10.0],
            }
        )
        adata = to_anndata(df)
        assert "p_minus_1" not in adata.var.columns
        assert "is_proline_directed" not in adata.var.columns
