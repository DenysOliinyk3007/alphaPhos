"""Unit tests for alphaphos.preprocess._collapse.selectivity."""

from __future__ import annotations

import pandas as pd
import pytest

from alphaphos.preprocess._collapse.selectivity import compute_selectivity


class TestComputeSelectivity:
    def test_all_phospho(self):
        df = pd.DataFrame(
            {
                "R.FileName": ["s1", "s1"],
                "EG.PrecursorId": [
                    "_S[Phospho (STY)]K_.2",
                    "_T[Phospho (STY)]R_.2",
                ],
            }
        )
        out = compute_selectivity(df).sort_values("sample").reset_index(drop=True)
        assert out.loc[0, "phospho_selectivity_pct"] == 100.0
        assert out.loc[0, "total_precursors"] == 2
        assert out.loc[0, "phospho_precursors"] == 2

    def test_mixed(self):
        df = pd.DataFrame(
            {
                "R.FileName": ["s1", "s1", "s2", "s2"],
                "EG.PrecursorId": [
                    "_S[Phospho (STY)]K_.2",
                    "_PEPTIDE_.2",  # non-phospho
                    "_S[Phospho (STY)]K_.2",
                    "_T[Phospho (STY)]R_.2",
                ],
            }
        )
        out = compute_selectivity(df).set_index("sample")
        assert out.loc["s1", "phospho_selectivity_pct"] == 50.0
        assert out.loc["s2", "phospho_selectivity_pct"] == 100.0

    def test_deduplicates_within_sample(self):
        # Same precursor ID appearing multiple times per sample counted once.
        df = pd.DataFrame(
            {
                "R.FileName": ["s1", "s1", "s1"],
                "EG.PrecursorId": ["_PEPTIDE_.2"] * 3,
            }
        )
        out = compute_selectivity(df).set_index("sample")
        assert out.loc["s1", "total_precursors"] == 1

    def test_missing_column_raises(self):
        df = pd.DataFrame({"R.FileName": ["s1"]})
        with pytest.raises(KeyError, match="required column"):
            compute_selectivity(df)

    def test_custom_column_names(self):
        df = pd.DataFrame(
            {
                "SampleID": ["s1", "s1"],
                "PrecursorId": [
                    "_S[Phospho (STY)]K_.2",
                    "_PEPTIDE_.2",
                ],
            }
        )
        out = compute_selectivity(df, sample_col="SampleID", precursor_col="PrecursorId").set_index(
            "sample"
        )
        assert out.loc["s1", "phospho_selectivity_pct"] == 50.0
