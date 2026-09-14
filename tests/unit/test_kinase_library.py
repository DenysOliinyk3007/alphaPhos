"""Tests for alphaphos.kinase.library.

Covers:
  - _strip_alphaphos_markers : alphaPhos -> Yaffe sequence format converter
  - score_kinases            : populates adata.varm + adata.uns
  - predict_kinases          : top-k DataFrame; handles missing-sequence rows
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

# Skip the whole module if kinase-library isn't installed
pytest = __import__("pytest")
pytest.importorskip("kinase_library")

from alphaphos.kinase.library import (  # noqa: E402  -- importorskip must run first
    _strip_alphaphos_markers,
    predict_kinases,
    score_kinases,
)

# ============================================================================
# _strip_alphaphos_markers
# ============================================================================


class TestStripAlphaphosMarkers:
    def test_strips_asterisks_around_phospho_residue(self):
        assert _strip_alphaphos_markers("_AAVKRGT*S*ELLIQAA_") == "_AAVKRGTSELLIQAA_"

    def test_preserves_underscore_padding(self):
        # C-terminal padding stays in place
        assert _strip_alphaphos_markers("_AAVKRGT*S*EL______") == "_AAVKRGTSEL______"

    def test_none_for_error_sentinel(self):
        # One representative sentinel; None/empty/NaN share the same guard branch.
        assert _strip_alphaphos_markers("FASTA_ERROR: not found") is None


# ============================================================================
# Helpers
# ============================================================================


def _make_adata_with_sequences(seqs: list[str | None]) -> ad.AnnData:
    """Build a tiny AnnData with the given kinase_sequence values."""
    n = len(seqs)
    X = np.random.default_rng(0).normal(10, 1, size=(3, n))
    obs = pd.DataFrame({"condition": ["A", "B", "A"]}, index=[f"s{i}" for i in range(3)])
    var = pd.DataFrame(
        {"kinase_sequence": seqs},
        index=[f"site{i}" for i in range(n)],
    )
    return ad.AnnData(X=X, obs=obs, var=var)


# Use a mix of S, T, Y phospho-centered sequences in the alphaPhos format
_VALID_SEQS = [
    "_AAVKRGT*S*ELLIQAA_",  # Ser
    "_PPRPAGT*T*PRSLEER_",  # Thr
    "_KRPAGNV*Y*HQPLNPA_",  # Tyr
    "_RRRRRGT*S*GGGGGGG_",  # Ser, contrived basophilic
]


# ============================================================================
# score_kinases
# ============================================================================


class TestScoreKinases:
    def test_populates_varm_for_ser_thr_and_tyrosine(self):
        adata = _make_adata_with_sequences(_VALID_SEQS)
        score_kinases(adata)

        # Both ser/thr and tyrosine matrices should be present
        assert "kinase_score_ser_thr" in adata.varm
        assert "kinase_score_tyrosine" in adata.varm

        st = adata.varm["kinase_score_ser_thr"]
        ty = adata.varm["kinase_score_tyrosine"]
        # Both are DataFrames with site rows + kinase columns
        assert isinstance(st, pd.DataFrame)
        assert isinstance(ty, pd.DataFrame)
        assert st.shape[0] == adata.n_vars
        assert ty.shape[0] == adata.n_vars
        # The library has 311 Ser/Thr + 78 tyrosine kinases (as of 1.5.0)
        assert st.shape[1] == 311
        assert ty.shape[1] == 78

    def test_uns_provenance(self):
        adata = _make_adata_with_sequences(_VALID_SEQS)
        score_kinases(adata)
        info = adata.uns["alphaphos"]["kinase_scores"]
        assert info["n_sites_scored"] == 4
        assert info["n_sites_dropped"] == 0
        assert info["n_ser_thr_scored"] == 3
        assert info["n_tyrosine_scored"] == 1
        assert len(info["ser_thr_kinases"]) == 311
        assert len(info["tyrosine_kinases"]) == 78

    def test_drops_invalid_sequences(self):
        seqs = [
            "_AAVKRGT*S*ELLIQAA_",  # valid
            None,  # missing
            "FASTA_ERROR: missing",  # sentinel
            "_invalid_no_asterisk_",  # no markers -> even length after strip
        ]
        adata = _make_adata_with_sequences(seqs)
        score_kinases(adata)
        info = adata.uns["alphaphos"]["kinase_scores"]
        assert info["n_sites_scored"] == 1  # only the valid one
        assert info["n_sites_dropped"] == 3
        # Dropped rows should be all-NaN in varm
        st = adata.varm["kinase_score_ser_thr"]
        assert st.iloc[0].notna().any()
        assert st.iloc[1].isna().all()
        assert st.iloc[2].isna().all()
        assert st.iloc[3].isna().all()

    def test_errors_when_sequence_col_missing(self):
        adata = _make_adata_with_sequences(_VALID_SEQS[:1])
        adata.var = adata.var.drop(columns=["kinase_sequence"])
        with pytest.raises(ValueError, match="kinase_sequence"):
            score_kinases(adata)

    def test_errors_when_no_valid_sequences(self):
        adata = _make_adata_with_sequences([None, "FASTA_ERROR: nope"])
        with pytest.raises(ValueError, match="No valid sequences"):
            score_kinases(adata)


# ============================================================================
# predict_kinases
# ============================================================================


class TestPredictKinases:
    def test_returns_top_k_dataframe(self):
        adata = _make_adata_with_sequences(_VALID_SEQS)
        result = predict_kinases(adata, top_k=3)

        # Indexed by site
        assert (result.index == adata.var.index).all()
        # Columns
        for i in range(1, 4):
            assert f"top{i}_kinase" in result.columns
            assert f"top{i}_score" in result.columns
        assert "top_kinases" in result.columns
        assert "top_scores" in result.columns
        assert "kin_type" in result.columns

        # Each valid site has a top1_kinase
        assert result["top1_kinase"].notna().all()
        # ser_thr rows tagged
        st_rows = result["kin_type"] == "ser_thr"
        assert st_rows.sum() == 3
        ty_rows = result["kin_type"] == "tyrosine"
        assert ty_rows.sum() == 1

    def test_scores_descend(self):
        adata = _make_adata_with_sequences(_VALID_SEQS)
        result = predict_kinases(adata, top_k=5)
        # For each site, the top1 score >= top2 score >= top3 ...
        for i in range(1, 5):
            valid = result[f"top{i}_score"].notna() & result[f"top{i + 1}_score"].notna()
            if valid.any():
                gte = result.loc[valid, f"top{i}_score"] >= result.loc[valid, f"top{i + 1}_score"]
                assert gte.all(), f"top{i} should >= top{i + 1}"

    def test_invalid_sites_marked_none(self):
        seqs = ["_AAVKRGT*S*ELLIQAA_", None, "FASTA_ERROR: x"]
        adata = _make_adata_with_sequences(seqs)
        result = predict_kinases(adata, top_k=2)
        assert result.iloc[0]["kin_type"] == "ser_thr"
        assert result.iloc[1]["kin_type"] == "none"
        assert result.iloc[2]["kin_type"] == "none"
        assert pd.isna(result.iloc[1]["top1_kinase"])

    def test_skips_scoring_when_varm_present(self):
        adata = _make_adata_with_sequences(_VALID_SEQS)
        score_kinases(adata)
        # Manually mark we ran -- predict_kinases shouldn't re-score
        old_st = adata.varm["kinase_percentile_ser_thr"].copy()
        adata.uns["alphaphos"]["kinase_scores"]["n_sites_scored"] = -1  # sentinel
        predict_kinases(adata, top_k=2, overwrite=False)
        pd.testing.assert_frame_equal(
            adata.varm["kinase_percentile_ser_thr"], old_st, check_exact=False
        )
        assert adata.uns["alphaphos"]["kinase_scores"]["n_sites_scored"] == -1  # not re-run

    def test_percentile_default_and_score_option(self):
        adata = _make_adata_with_sequences(_VALID_SEQS)
        pct = predict_kinases(adata, top_k=3)
        sc = predict_kinases(adata, top_k=3, metric="score")
        assert pct.attrs["metric"] == "percentile" and sc.attrs["metric"] == "score"
        assert "kinase_percentile_ser_thr" in adata.varm and "kinase_score_ser_thr" in adata.varm
        # percentiles are bounded, raw log2 scores are not
        assert float(pct["top1_score"].dropna().astype(float).max()) <= 100.0

    def test_overwrite_reruns(self):
        adata = _make_adata_with_sequences(_VALID_SEQS)
        score_kinases(adata)
        # overwrite=True should call score_kinases again (idempotent so values match)
        predict_kinases(adata, top_k=2, overwrite=True)
        info = adata.uns["alphaphos"]["kinase_scores"]
        assert info["n_sites_scored"] == 4
