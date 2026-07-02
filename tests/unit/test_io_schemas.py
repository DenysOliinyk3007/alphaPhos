"""Unit tests for :mod:`alphaphos.io.schemas`.

The schema dictionaries are the single source of truth for
"which columns belong to which search engine at which quant level".
These tests lock in the shape so accidental edits are caught fast.
"""

from __future__ import annotations

import pytest

from alphaphos.io import schemas


class TestSchemaShape:
    def test_required_present_for_SN(self):
        assert "SN" in schemas.REQUIRED_COLUMNS
        req = schemas.REQUIRED_COLUMNS["SN"]
        # Non-negotiable columns for the pipeline
        for col in (
            "R.FileName",
            "EG.PrecursorId",
            "PEP.PeptidePosition",
            "EG.PTMAssayProbability",
            "PG.Genes",
            "PG.ProteinGroups",
        ):
            assert col in req, (
                f"required column {col!r} missing from schemas.REQUIRED_COLUMNS['SN']"
            )

    def test_optional_has_loc_string(self):
        # The per-position loc column is optional but hugely valuable when present.
        assert "EG.PTMLocalizationProbabilities" in schemas.OPTIONAL_COLUMNS["SN"]

    def test_quant_candidates_have_three_levels(self):
        assert set(schemas.QUANT_COLUMN_CANDIDATES["SN"]) == {"auto", "MS1", "MS2"}

    def test_ms2_has_at_least_one_candidate(self):
        assert len(schemas.QUANT_COLUMN_CANDIDATES["SN"]["MS2"]) >= 1

    def test_fallback_chains_default(self):
        # MS2 is the default -> deepest chain
        assert schemas.FALLBACK_CHAINS["MS2"] == ("MS2", "MS1", "auto")
        assert schemas.FALLBACK_CHAINS["MS1"] == ("MS1", "auto")
        assert schemas.FALLBACK_CHAINS["auto"] == ("auto",)


class TestAllNeededColumns:
    def test_union_contains_required(self):
        needed = schemas.all_needed_columns("SN")
        for c in schemas.REQUIRED_COLUMNS["SN"]:
            assert c in needed

    def test_union_contains_optional(self):
        needed = schemas.all_needed_columns("SN")
        for c in schemas.OPTIONAL_COLUMNS["SN"]:
            assert c in needed

    def test_union_contains_all_quant_candidates(self):
        needed = schemas.all_needed_columns("SN")
        for level_cols in schemas.QUANT_COLUMN_CANDIDATES["SN"].values():
            for c in level_cols:
                assert c in needed

    # test_unknown_engine_raises removed here (near-duplicate of the
    # TestResolveQuantColumn.test_unknown_engine_raises below).


class TestResolveQuantColumn:
    @pytest.mark.parametrize(
        ("available", "expected_col", "expected_level"),
        [
            (
                {"FG.MS2Quantity", "R.FileName"},
                "FG.MS2Quantity",
                "MS2",
            ),  # direct MS2 hit, no fallback
            (
                {"FG.MS1Quantity", "R.FileName"},
                "FG.MS1Quantity",
                "MS1",
            ),  # MS2 -> MS1 fallback
            (
                {"EG.TotalQuantity (Settings)"},
                "EG.TotalQuantity (Settings)",
                "auto",
            ),  # MS2 -> MS1 -> auto fallback
        ],
    )
    def test_ms2_request_fallback_chain(self, available, expected_col, expected_level):
        col, level = schemas.resolve_quant_column(
            available_columns=available, engine="SN", requested_level="MS2"
        )
        assert col == expected_col
        assert level == expected_level

    def test_prefers_first_candidate_within_level(self):
        # Both FG.MS2Quantity and FG.MS2RawQuantity present -> the ordered
        # first one wins.
        col, level = schemas.resolve_quant_column(
            available_columns={"FG.MS2Quantity", "FG.MS2RawQuantity"},
            engine="SN",
            requested_level="MS2",
        )
        assert col == "FG.MS2Quantity"
        assert level == "MS2"

    def test_auto_does_not_fall_back(self):
        # auto has no further fallback -> hard error if not found
        with pytest.raises(ValueError, match="No quantification column"):
            schemas.resolve_quant_column(
                available_columns={"R.FileName"},
                engine="SN",
                requested_level="auto",
            )

    def test_nothing_found_raises(self):
        with pytest.raises(ValueError, match="No quantification column"):
            schemas.resolve_quant_column(
                available_columns={"R.FileName"},
                engine="SN",
                requested_level="MS2",
            )

    def test_unknown_engine_raises(self):
        with pytest.raises(KeyError, match="Unknown engine"):
            schemas.resolve_quant_column(
                available_columns=set(),
                engine="Fragpipe",
                requested_level="MS2",
            )

    def test_unknown_level_raises(self):
        with pytest.raises(KeyError, match="Unknown quantification_level"):
            schemas.resolve_quant_column(
                available_columns=set(),
                engine="SN",
                requested_level="MS3",
            )
