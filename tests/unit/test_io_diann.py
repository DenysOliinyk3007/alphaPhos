"""Unit tests for :mod:`alphaphos.io.diann`.

Two levels:

* **Pure helper tests** -- exercise ``diann_precursor_id``, ``diann_loc_probs``,
  ``first_phospho_abs_position``, ``diann_peptide_start`` with small string
  fixtures. No I/O.
* **Adapter + reader tests** -- synthesize a tiny DIA-NN-shaped DataFrame,
  serialise to parquet, read it back through ``read_diann``, and assert on
  the canonical schema, filters, and adapter math.

Note: reference DIA-NN reports from the nanoPhos revision live at
``Z:/Denys_nanoPhos/PRIDE/analysis_data/revision/figure3/diann/``. When
those files are visible, ``test_real_report_hela100`` runs an end-to-end
integration; otherwise it's skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaphos.io.diann import (
    DEFAULT_DIANN_IO_SETTINGS,
    _diann_to_psm,
    diann_loc_probs,
    diann_peptide_start,
    diann_precursor_id,
    first_phospho_abs_position,
    resolve_diann_io_settings,
)
from alphaphos.io.diann import (
    read_psm as read_diann,
)

# ---------------------------------------------------------------------------
# diann_precursor_id
# ---------------------------------------------------------------------------


class TestDiannPrecursorId:
    def test_single_phospho(self):
        assert diann_precursor_id("AAS(UniMod:21)PLK", 2) == "_AAS[Phospho (STY)]PLK_.2"

    def test_multi_phospho(self):
        s = diann_precursor_id("AAS(UniMod:21)PT(UniMod:21)K", 3)
        assert s == "_AAS[Phospho (STY)]PT[Phospho (STY)]K_.3"

    def test_strips_other_unimods(self):
        # Carbamidomethyl (UniMod:4) should be stripped entirely.
        s = diann_precursor_id("AC(UniMod:4)AS(UniMod:21)PLK", 2)
        assert s == "_ACAS[Phospho (STY)]PLK_.2"

    def test_no_phospho_still_wraps(self):
        s = diann_precursor_id("AASPLK", 2)
        assert s == "_AASPLK_.2"


# ---------------------------------------------------------------------------
# diann_loc_probs
# ---------------------------------------------------------------------------


class TestDiannLocProbs:
    def test_single_position(self):
        assert diann_loc_probs("AAS(UniMod:21){1.000000}PLK2") == "_AAS[Phospho (STY): 100%]PLK_"

    def test_multiple_positions(self):
        s = diann_loc_probs("AAS(UniMod:21){1.000000}LPT{0.848000}K2")
        assert s == "_AAS[Phospho (STY): 100%]LPT[Phospho (STY): 84.8%]K_"

    def test_missing_returns_nan(self):
        assert np.isnan(diann_loc_probs(None))
        assert np.isnan(diann_loc_probs(float("nan")))
        assert np.isnan(diann_loc_probs(42))


# ---------------------------------------------------------------------------
# first_phospho_abs_position
# ---------------------------------------------------------------------------


class TestFirstPhosphoAbsPosition:
    def test_single_sty(self):
        assert first_phospho_abs_position("[P12345:S117]") == 117

    def test_multi_sty_returns_first(self):
        assert first_phospho_abs_position("[P12345:S117,T119]") == 117

    def test_skips_carbamidomethyl_c(self):
        # DIA-NN sometimes bundles Cys sites in Protein.Sites; drop non-STY.
        assert first_phospho_abs_position("[P35221:C116,S118]") == 118

    def test_no_sty_returns_nan(self):
        assert np.isnan(first_phospho_abs_position("[P35221:C116]"))
        assert np.isnan(first_phospho_abs_position("[P00000:]"))


# ---------------------------------------------------------------------------
# diann_peptide_start
# ---------------------------------------------------------------------------


class TestDiannPeptideStart:
    def test_first_phospho(self):
        # AAS(UniMod:21)PLK with S at protein pos 100 -> peptide start 98
        assert diann_peptide_start("AAS(UniMod:21)PLK", "[P12345:S100]") == 98

    def test_second_phospho_still_anchors_first(self):
        # First (UniMod:21) is after position 3 (S). If Protein.Sites lists
        # S at 100 and T at 105, peptide_start uses the first-phospho anchor.
        assert diann_peptide_start("AAS(UniMod:21)PT(UniMod:21)K", "[P12345:S100,T105]") == 98

    def test_returns_nan_without_phospho(self):
        assert np.isnan(diann_peptide_start("AASPLK", "[P12345:S100]"))

    def test_returns_nan_without_protein_sites(self):
        assert np.isnan(diann_peptide_start("AAS(UniMod:21)PLK", "[P12345:]"))


# ---------------------------------------------------------------------------
# resolve_diann_io_settings
# ---------------------------------------------------------------------------


class TestResolveDiannIOSettings:
    def test_defaults(self):
        s = resolve_diann_io_settings(None)
        assert s == DEFAULT_DIANN_IO_SETTINGS
        assert s is not DEFAULT_DIANN_IO_SETTINGS  # copy

    def test_partial_override(self):
        s = resolve_diann_io_settings({"pg_qvalue_max": 0.01, "mbr": False})
        assert s["pg_qvalue_max"] == 0.01
        assert s["mbr"] is False
        assert s["quantity_quality_min"] == 0.5  # default kept

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown keys"):
            resolve_diann_io_settings({"pg_qvalue": 0.01})  # typo

    def test_bad_mbr_type_raises(self):
        with pytest.raises(ValueError, match="mbr must be bool"):
            resolve_diann_io_settings({"mbr": 1})

    def test_bad_qvalue_raises(self):
        with pytest.raises(ValueError, match="pg_qvalue_max"):
            resolve_diann_io_settings({"pg_qvalue_max": 1.5})

    def test_none_disables_filter(self):
        s = resolve_diann_io_settings({"pg_qvalue_max": None})
        assert s["pg_qvalue_max"] is None


# ---------------------------------------------------------------------------
# Adapter: DIA-NN raw -> Spectronaut-canonical
# ---------------------------------------------------------------------------


def _make_synthetic_diann_df() -> pd.DataFrame:
    """Small DIA-NN-shaped DataFrame -- includes edge cases and one unmappable row."""
    return pd.DataFrame(
        {
            "Run": ["s1", "s1", "s2", "s2"],
            "Modified.Sequence": [
                "AAS(UniMod:21)PLK",
                "AC(UniMod:4)S(UniMod:21)TVK",
                "PET(UniMod:21)K",
                "AAS(UniMod:21)PLK",  # duplicate of row 0's peptide, different sample
            ],
            "Precursor.Charge": [2, 3, 2, 2],
            "Precursor.Quantity": [1000.0, 500.0, 250.0, 800.0],
            "Precursor.Normalised": [1010.0, 490.0, 260.0, 790.0],
            "Ms1.Translated": [900.0, 450.0, 220.0, 720.0],
            "Ms1.Area": [910.0, 460.0, 230.0, 730.0],
            "Protein.Sites": [
                "[P12345:S100]",
                "[P54321:C50,S52]",
                "[P00000:]",  # unmappable: no S/T/Y in Protein.Sites
                "[P12345:S100]",
            ],
            "PTM.Site.Confidence": [0.95, 0.99, 0.8, 0.9],
            "Site.Occupancy.Probabilities": [
                "AAS(UniMod:21){1.000000}PLK2",
                "ACS(UniMod:21){0.980000}TVK3",
                "PET(UniMod:21){1.000000}K2",
                "AAS(UniMod:21){1.000000}PLK2",
            ],
            "Genes": ["GENE1", "GENE2;GENE2B", "GENE3", "GENE1"],
            "Protein.Group": ["P12345", "P54321", "P00000", "P12345"],
            "PG.Q.Value": [0.001, 0.001, 0.002, 0.001],
            "Global.PG.Q.Value": [0.001, 0.005, 0.008, 0.001],
            "Lib.PG.Q.Value": [0.005, 0.005, 0.005, 0.005],
            "Quantity.Quality": [0.9, 0.8, 0.7, 0.85],
            "PG.MaxLFQ.Quality": [0.9, 0.85, 0.75, 0.9],
        }
    )


class TestDiannToPsm:
    def test_canonical_columns_present(self):
        df_out, _ = _diann_to_psm(_make_synthetic_diann_df())
        expected = {
            "R.FileName",
            "EG.PrecursorId",
            "PEP.PeptidePosition",
            "EG.PTMAssayProbability",
            "EG.PTMLocalizationProbabilities",
            "PG.Genes",
            "PG.ProteinGroups",
        }
        assert expected.issubset(set(df_out.columns))

    def test_quant_columns_preserved(self):
        df_out, _ = _diann_to_psm(_make_synthetic_diann_df())
        for c in ("Precursor.Quantity", "Precursor.Normalised", "Ms1.Translated", "Ms1.Area"):
            assert c in df_out.columns, f"missing quant column {c!r}"

    def test_multi_gene_split_takes_first(self):
        df_out, _ = _diann_to_psm(_make_synthetic_diann_df())
        # Row 1 had "GENE2;GENE2B" -> should become "GENE2"
        assert "GENE2" in df_out["PG.Genes"].values
        assert "GENE2B" not in df_out["PG.Genes"].values

    def test_precursor_id_format(self):
        df_out, _ = _diann_to_psm(_make_synthetic_diann_df())
        first = df_out["EG.PrecursorId"].iloc[0]
        assert first.startswith("_") and first.endswith("_.2")
        assert "[Phospho (STY)]" in first

    def test_unmappable_row_dropped(self):
        df_out, n_unmappable = _diann_to_psm(_make_synthetic_diann_df())
        # Row 2 (PET(UniMod:21)K with Protein.Sites=[P00000:]) had no STY -> unmappable
        assert n_unmappable == 1
        # GENE3 (that row's gene) should not appear in the output.
        assert "GENE3" not in df_out["PG.Genes"].values

    def test_peptide_position_math(self):
        df_out, _ = _diann_to_psm(_make_synthetic_diann_df())
        # AAS(UniMod:21)PLK with S at 100: peptide starts at 98.
        rows_gene1 = df_out[df_out["PG.Genes"] == "GENE1"]
        assert (rows_gene1["PEP.PeptidePosition"] == 98).all()


# ---------------------------------------------------------------------------
# End-to-end: synthetic parquet -> read_diann -> collapse_sites
# ---------------------------------------------------------------------------


def _write_synthetic_parquet(tmp_path: Path) -> Path:
    p = tmp_path / "synth_diann.parquet"
    _make_synthetic_diann_df().to_parquet(p, index=False)
    return p


class TestReadDiann:
    def test_returns_canonical_schema(self, tmp_path):
        p = _write_synthetic_parquet(tmp_path)
        df = read_diann(p)
        for c in (
            "R.FileName",
            "EG.PrecursorId",
            "PEP.PeptidePosition",
            "EG.PTMAssayProbability",
            "PG.Genes",
            "PG.ProteinGroups",
        ):
            assert c in df.columns

    def test_attrs_stamped(self, tmp_path):
        p = _write_synthetic_parquet(tmp_path)
        df = read_diann(p)
        assert df.attrs["engine"] == "Diann"
        assert df.attrs["source_path"].endswith("synth_diann.parquet")
        assert df.attrs["n_rows_loaded"] == 4
        assert "n_rows_after_qc" in df.attrs
        assert "n_rows_unmappable" in df.attrs

    def test_unmappable_dropped_by_default(self, tmp_path):
        p = _write_synthetic_parquet(tmp_path)
        df = read_diann(p)
        # Row 2's gene shouldn't appear (unmappable).
        assert "GENE3" not in df["PG.Genes"].values

    def test_qc_filters_applied(self, tmp_path):
        # Tighten Quantity.Quality to drop the third row (0.7 < 0.75).
        p = _write_synthetic_parquet(tmp_path)
        df = read_diann(p, advanced={"quantity_quality_min": 0.75})
        assert df.attrs["n_rows_after_qc"] <= 3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_diann(tmp_path / "nope.parquet")

    def test_unknown_advanced_key_raises(self, tmp_path):
        p = _write_synthetic_parquet(tmp_path)
        with pytest.raises(ValueError, match="Unknown keys"):
            read_diann(p, advanced={"pg_qvalue": 0.01})


class TestCollapseSitesWithDiann:
    """End-to-end: synthetic parquet -> read_diann -> collapse_sites -> AnnData."""

    def test_collapse_returns_anndata(self, tmp_path):
        import anndata as ad

        import alphaphos as ap

        p = _write_synthetic_parquet(tmp_path)
        psm = ap.read_diann(p)
        cdf = pd.DataFrame({"sample": ["s1", "s2"], "condition": ["ctrl", "trt"]})
        adata = ap.collapse_sites(
            psm,
            condition_df=cdf,
            advanced={
                "search_engine": "Diann",
                "localization_strategy": "per_run",
                "quantification_level": "auto",  # DIA-NN's Precursor.Quantity
                "cutoff": 0.5,  # loose enough for our synthetic fixture
            },
        )
        assert isinstance(adata, ad.AnnData)
        assert adata.n_obs == 2
        assert adata.n_vars >= 1

    def test_default_ms2_falls_back_to_ms1(self, tmp_path):
        # DIA-NN has no MS2 column; MS2 should fall back to MS1 with a warning.
        import alphaphos as ap

        p = _write_synthetic_parquet(tmp_path)
        psm = ap.read_diann(p)
        cdf = pd.DataFrame({"sample": ["s1", "s2"], "condition": ["ctrl", "trt"]})
        with pytest.warns(UserWarning, match="quantification_level='MS2' unavailable"):
            adata = ap.collapse_sites(
                psm,
                condition_df=cdf,
                advanced={
                    "search_engine": "Diann",
                    "localization_strategy": "per_run",
                    "cutoff": 0.5,
                    # quantification_level defaults to "MS2"
                },
            )
        assert adata.uns["alphaphos"]["stats"]["quantification_level_used"] == "MS1"


# ---------------------------------------------------------------------------
# Optional: real DIA-NN report validation (nanoPhos revision fixtures)
# ---------------------------------------------------------------------------


_REAL_DIANN_PARQUET = Path(
    "Z:/Denys_nanoPhos/PRIDE/analysis_data/revision/figure3/diann/figure3/"
    "hela_100_cells/hela_report_100.parquet"
)


@pytest.mark.skipif(
    not _REAL_DIANN_PARQUET.exists(),
    reason="Real DIA-NN reference file not available (nanoPhos Z: share).",
)
class TestRealDiannReport:
    """Validate against the manuscript reference report (hela_100_cells)."""

    def test_reader_produces_reasonable_row_count(self):
        df = read_diann(_REAL_DIANN_PARQUET)
        # Notebook funnel for hela_100_cells reported ~11,563 localizable phospho rows.
        # We're less strict here; just require substantial output.
        assert len(df) > 1000, f"suspiciously few rows: {len(df)}"
        assert df.attrs["n_rows_loaded"] > df.attrs["n_rows_returned"]
        assert df.attrs["n_rows_phospho"] > 0

    def test_canonical_schema_valid(self):
        df = read_diann(_REAL_DIANN_PARQUET)
        # Every non-null EG.PrecursorId must contain the phospho marker.
        assert df["EG.PrecursorId"].str.contains(r"\[Phospho \(STY\)\]", regex=True).all()

    def test_peptide_positions_all_positive(self):
        df = read_diann(_REAL_DIANN_PARQUET)
        assert (df["PEP.PeptidePosition"] > 0).all()
