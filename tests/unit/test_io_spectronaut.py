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
    _decoy_mask,
    _to_dotted_column,
    read_psm,
    resolve_io_settings,
)

# ---------------------------------------------------------------------------
# Column-name normalization
# ---------------------------------------------------------------------------


class TestToDottedColumn:
    @pytest.mark.parametrize(
        ("input_col", "expected"),
        [
            ("R_FileName", "R.FileName"),
            ("R.FileName", "R.FileName"),
            ("EG_TotalQuantity_(Settings)", "EG.TotalQuantity (Settings)"),
        ],
    )
    def test_normalization(self, input_col, expected):
        assert _to_dotted_column(input_col) == expected


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

    @pytest.mark.parametrize(
        ("bad_settings", "exc_type", "match"),
        [
            ({"drop_decys": True}, ValueError, "Unknown keys"),  # typo
            ({"drop_decoys": 1}, ValueError, "drop_decoys must be bool"),
            ({"eg_qvalue_max": 1.5}, ValueError, "eg_qvalue_max"),
            ({"pg_qvalue_max": -0.1}, ValueError, "pg_qvalue_max"),
            # bool is an int subclass; must not silently pass as 1.0
            ({"eg_qvalue_max": True}, ValueError, "eg_qvalue_max"),
            ({"contaminant_prefixes": ("CON__", 3)}, ValueError, "contaminant_prefixes"),
        ],
    )
    def test_validation_raises(self, bad_settings, exc_type, match):
        with pytest.raises(exc_type, match=match):
            resolve_io_settings(bad_settings)

    def test_qvalue_none_allowed(self):
        s = resolve_io_settings({"eg_qvalue_max": None})
        assert s["eg_qvalue_max"] is None


# ---------------------------------------------------------------------------
# Synthetic TSV builder for column-pruning + filter tests
# ---------------------------------------------------------------------------


def _make_synthetic_tsv(
    tmp_path: Path,
    *,
    extra_columns: bool = True,
    bom: bool = False,
    decoy_values: tuple[str, str, str] = ("False", "True", "False"),
) -> Path:
    """Write a small Spectronaut-shaped TSV. Includes decoys + contaminants
    for filter tests. Optionally includes many extra unused columns so we
    can assert column pruning drops them.

    ``bom=True`` prefixes the file with a UTF-8 BOM (as some Windows exports
    do). ``decoy_values`` sets the three ``EG.IsDecoy`` cells (row order:
    real, decoy, contaminant) so string / numeric / missing encodings can be
    exercised.
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
            decoy_values[0],
            "AKT1",
            "P31749",
            0.001,
        ),
        # Decoy row (should be dropped when drop_decoys=True)
        (
            "s1",
            "_DECOY_.2",
            100.0,
            "1",
            0.5,
            "_DECOY_",
            decoy_values[1],
            "DecoyGene",
            "DecoyPG",
            0.5,
        ),
        # Contaminant row (starts with CON__)
        (
            "s2",
            "_S[Phospho (STY)]K_.2",
            500.0,
            "10",
            0.9,
            "_S[Phospho (STY): 90.0%]K_",
            decoy_values[2],
            "TRYP",
            "CON__P00761",
            0.02,
        ),
    ]
    # Fill junk_cols with placeholder values so column widths match
    padded_rows = [tuple(list(r) + ["x"] * len(junk_cols)) for r in rows]

    tsv_path = tmp_path / "synth.tsv"
    with open(tsv_path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
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
# Decoy-column encodings
# ---------------------------------------------------------------------------


class TestDecoyMask:
    def test_bool_dtype_passthrough(self):
        s = pd.Series([False, True, False])
        assert _decoy_mask(s).tolist() == [False, True, False]

    def test_object_strings_with_missing(self):
        # "False" as a *string* must not be truthy; NaN counts as not-decoy.
        s = pd.Series(["False", None, "True", " false "], dtype=object)
        assert _decoy_mask(s).tolist() == [False, False, True, False]

    def test_numeric_zero_one(self):
        s = pd.Series([0, 1, 0])
        assert _decoy_mask(s).tolist() == [False, True, False]

    def test_unrecognized_values_raise(self):
        with pytest.raises(ValueError, match="Unrecognized values"):
            _decoy_mask(pd.Series(["yes", "no"], dtype=object))

    def test_reader_keeps_rows_when_decoy_column_has_missing_values(self, tmp_path):
        # A blank cell forces object dtype: before the fix, astype(bool)
        # turned "False" into True and dropped every row.
        tsv = _make_synthetic_tsv(tmp_path, decoy_values=("False", "True", ""))
        df = read_psm(tsv, advanced={"drop_contaminants": False})
        assert len(df) == 2
        assert "_DECOY_.2" not in df["EG.PrecursorId"].values

    def test_reader_handles_numeric_decoy_column(self, tmp_path):
        tsv = _make_synthetic_tsv(tmp_path, decoy_values=("0", "1", "0"))
        df = read_psm(tsv, advanced={"drop_contaminants": False})
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Encoding quirks
# ---------------------------------------------------------------------------


class TestEncoding:
    def test_utf8_bom_header_is_recognized(self, tmp_path):
        tsv = _make_synthetic_tsv(tmp_path, bom=True)
        df = read_psm(tsv, advanced={"drop_contaminants": False})
        # Without BOM handling the first column would be "﻿R.FileName"
        # and the reader would report R.FileName as missing.
        assert "R.FileName" in df.columns
        assert len(df) == 2


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
