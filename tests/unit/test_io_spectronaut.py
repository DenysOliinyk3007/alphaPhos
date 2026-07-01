"""Unit tests for :mod:`alphaphos.io.spectronaut`.

Focus: settings validation, column-pruning at read time, and filter behavior.
Uses tiny synthetic TSV files (no external data) so tests are portable.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from alphaphos.io.spectronaut import (
    DEFAULT_IO_SETTINGS,
    _to_dotted_column,
    read_psm,
    resolve_io_settings,
)

# ---------------------------------------------------------------------------
# Column-name normalization
# ---------------------------------------------------------------------------


class TestToDottedColumn:
    def test_underscore_form(self):
        assert _to_dotted_column("R_FileName") == "R.FileName"

    def test_dotted_passthrough(self):
        assert _to_dotted_column("R.FileName") == "R.FileName"

    def test_flag_suffix_converted(self):
        assert _to_dotted_column("EG_TotalQuantity_(Settings)") == "EG.TotalQuantity (Settings)"


# ---------------------------------------------------------------------------
# Settings resolution + validation
# ---------------------------------------------------------------------------


class TestResolveIOSettings:
    def test_defaults(self):
        s = resolve_io_settings(None)
        assert s == DEFAULT_IO_SETTINGS
        # Returned copy
        assert s is not DEFAULT_IO_SETTINGS

    def test_partial_override(self):
        s = resolve_io_settings({"drop_decoys": False, "eg_qvalue_max": 0.01})
        assert s["drop_decoys"] is False
        assert s["eg_qvalue_max"] == 0.01
        assert s["drop_contaminants"] is True  # default preserved

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown keys"):
            resolve_io_settings({"drop_decys": True})  # typo

    def test_non_dict_raises(self):
        with pytest.raises(TypeError):
            resolve_io_settings("not a dict")

    def test_bad_bool_raises(self):
        with pytest.raises(ValueError, match="drop_decoys must be bool"):
            resolve_io_settings({"drop_decoys": 1})  # int not bool

    def test_bad_qvalue_raises(self):
        with pytest.raises(ValueError, match="eg_qvalue_max"):
            resolve_io_settings({"eg_qvalue_max": 1.5})
        with pytest.raises(ValueError, match="pg_qvalue_max"):
            resolve_io_settings({"pg_qvalue_max": -0.1})

    def test_qvalue_none_allowed(self):
        s = resolve_io_settings({"eg_qvalue_max": None})
        assert s["eg_qvalue_max"] is None


# ---------------------------------------------------------------------------
# Synthetic TSV builder for column-pruning + filter tests
# ---------------------------------------------------------------------------


def _make_synthetic_tsv(tmp_path: Path, *, extra_columns: bool = True) -> Path:
    """Write a small Spectronaut-shaped TSV. Includes decoys + contaminants
    for filter tests. Optionally includes many extra unused columns so we
    can assert column pruning drops them.
    """
    core_cols = [
        "R.FileName",
        "EG.PrecursorId",
        "EG.TotalQuantity (Settings)",
        "PEP.PeptidePosition",
        "EG.PTMAssayProbability",
        "EG.PTMLocalizationProbabilities",
        "EG.IsDecoy",
        "PG.Genes",
        "PG.ProteinGroups",
        "EG.Qvalue",
    ]
    junk_cols = (
        [
            # These should be pruned at read time
            "EG.CScore",
            "EG.PEP",
            "EG.Charge",
            "EG.Cutoff",
            "R.Instrument",
            "R.CalibrationRTValue",
            "FG.PrecMz",
            "FG.PrecursorMZ",
            "PG.CellularComponent",
            "PG.MolecularFunction",
            "PG.BiologicalProcess",
        ]
        if extra_columns
        else []
    )

    all_cols = core_cols + junk_cols

    rows = [
        # Real phospho row on a real protein
        (
            "s1",
            "_S[Phospho (STY)]TSK_.2",
            1000.0,
            "42",
            0.95,
            "_S[Phospho (STY): 95.0%]TSK_",
            "False",
            "AKT1",
            "P31749",
            0.001,
        ),
        # Decoy row (should be dropped when drop_decoys=True)
        ("s1", "_DECOY_.2", 100.0, "1", 0.5, "_DECOY_", "True", "DecoyGene", "DecoyPG", 0.5),
        # Contaminant row (starts with CON__)
        (
            "s2",
            "_S[Phospho (STY)]K_.2",
            500.0,
            "10",
            0.9,
            "_S[Phospho (STY): 90.0%]K_",
            "False",
            "TRYP",
            "CON__P00761",
            0.02,
        ),
    ]
    # Fill junk_cols with placeholder values so column widths match
    padded_rows = [tuple(list(r) + ["x"] * len(junk_cols)) for r in rows]

    tsv_path = tmp_path / "synth.tsv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(all_cols) + "\n")
        for r in padded_rows:
            f.write("\t".join(str(v) for v in r) + "\n")
    return tsv_path


# ---------------------------------------------------------------------------
# Column pruning
# ---------------------------------------------------------------------------


class TestColumnPruning:
    def test_prunes_unused_columns_at_read_time(self, tmp_path):
        tsv = _make_synthetic_tsv(tmp_path, extra_columns=True)
        df = read_psm(tsv, advanced={"drop_contaminants": False})
        # Junk columns should NOT be in the returned DataFrame
        for junk in ("EG.CScore", "EG.PEP", "PG.CellularComponent", "R.Instrument"):
            assert junk not in df.columns, f"junk column {junk!r} leaked past pruning"

    def test_attrs_report_columns_dropped(self, tmp_path):
        tsv = _make_synthetic_tsv(tmp_path, extra_columns=True)
        df = read_psm(tsv, advanced={"drop_contaminants": False})
        assert df.attrs["columns_dropped"] >= 5  # 11 junk cols

    def test_keeps_all_quant_candidates_present(self, tmp_path):
        # Build a report with two quant columns and make sure BOTH are kept
        # (collapse picks one later; reader keeps all).
        header = [
            "R.FileName",
            "EG.PrecursorId",
            "PEP.PeptidePosition",
            "EG.PTMAssayProbability",
            "PG.Genes",
            "PG.ProteinGroups",
            "EG.TotalQuantity (Settings)",
            "FG.MS2Quantity",
        ]
        row = ["s1", "_S[Phospho (STY)]K_.2", "10", "0.9", "AKT1", "P31749", "1000.0", "800.0"]
        tsv = tmp_path / "twoquant.tsv"
        with open(tsv, "w", encoding="utf-8") as f:
            f.write("\t".join(header) + "\n")
            f.write("\t".join(row) + "\n")
        df = read_psm(tsv, advanced={"drop_contaminants": False})
        assert "EG.TotalQuantity (Settings)" in df.columns
        assert "FG.MS2Quantity" in df.columns


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestFilters:
    def test_decoy_dropped_by_default(self, tmp_path):
        tsv = _make_synthetic_tsv(tmp_path)
        df = read_psm(tsv, advanced={"drop_contaminants": False})
        # No row should have EG.IsDecoy == True
        assert not df["EG.IsDecoy"].astype(bool).any()

    def test_decoy_kept_when_disabled(self, tmp_path):
        tsv = _make_synthetic_tsv(tmp_path)
        df = read_psm(tsv, advanced={"drop_decoys": False, "drop_contaminants": False})
        # Decoy row should still be there
        assert df["EG.IsDecoy"].astype(bool).any()

    def test_contaminant_dropped_by_default(self, tmp_path):
        tsv = _make_synthetic_tsv(tmp_path)
        df = read_psm(tsv)
        # CON__ prefix row must have been dropped
        assert not df["PG.ProteinGroups"].astype(str).str.startswith("CON__").any()

    def test_qvalue_cutoff(self, tmp_path):
        tsv = _make_synthetic_tsv(tmp_path)
        df = read_psm(
            tsv,
            advanced={
                "drop_contaminants": False,
                "eg_qvalue_max": 0.01,  # only the first row passes
            },
        )
        assert (df["EG.Qvalue"] <= 0.01).all()

    def test_attrs_lineage_stamped(self, tmp_path):
        tsv = _make_synthetic_tsv(tmp_path)
        df = read_psm(tsv)
        assert df.attrs["engine"] == "SN"
        assert df.attrs["source_path"].endswith("synth.tsv")
        assert df.attrs["n_rows_loaded"] == 3
        assert "n_rows_after_decoys" in df.attrs
        assert "n_rows_after_contaminants" in df.attrs


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_psm(tmp_path / "nope.tsv")

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "foo.csv"
        p.write_text("R.FileName\ts1\n")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            read_psm(p)

    def test_missing_required_column_raises(self, tmp_path):
        # File lacks PG.Genes -> should raise on validate
        header = [
            "R.FileName",
            "EG.PrecursorId",
            "PEP.PeptidePosition",
            "EG.PTMAssayProbability",
            "PG.ProteinGroups",
            "EG.TotalQuantity (Settings)",
        ]
        row = ["s1", "_S[Phospho (STY)]K_.2", "10", "0.9", "P31749", "1000.0"]
        tsv = tmp_path / "bad.tsv"
        with open(tsv, "w", encoding="utf-8") as f:
            f.write("\t".join(header) + "\n")
            f.write("\t".join(row) + "\n")
        with pytest.raises(ValueError, match="missing required column"):
            read_psm(tsv, advanced={"drop_contaminants": False})
