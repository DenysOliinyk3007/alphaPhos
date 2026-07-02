"""Unit tests for alphaphos.preprocess._collapse.parsing.

Every test focuses on a single, well-defined behavior of one parser. The
functions here are pure -- no I/O, no state -- so tests use small string
literals as inputs and assert on returned dicts / lists / dicts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alphaphos.preprocess._collapse.parsing import (
    calculate_phospho_positions,
    extract_first_valid_position,
    extract_sequence_modifications,
    parse_localization_probabilities,
    rank_select_positions,
)

# ---------------------------------------------------------------------------
# extract_sequence_modifications
# ---------------------------------------------------------------------------


class TestExtractSequenceModifications:
    @pytest.mark.parametrize(
        ("precid", "expected"),
        [
            (
                "_PEPTIDE_.2",
                {"clean_sequence": "PEPTIDE", "phospho_positions": [], "phospho_count": 0},
            ),
            (
                "_S[Phospho (STY)]TSK_.3",
                {"clean_sequence": "STSK", "phospho_positions": [1], "phospho_count": 1},
            ),
            (
                "_S[Phospho (STY)]TS[Phospho (STY)]K_.3",
                {"clean_sequence": "STSK", "phospho_positions": [1, 3], "phospho_count": 2},
            ),
            (
                "*.S[Phospho (STY)]TSK.*.2",  # alt Spectronaut *..*.charge wrapper
                {"clean_sequence": "STSK", "phospho_positions": [1], "phospho_count": 1},
            ),
        ],
    )
    def test_precid_extraction(self, precid, expected):
        r = extract_sequence_modifications(precid)
        for key, val in expected.items():
            assert r[key] == val

    def test_mixed_mods_captured(self):
        r = extract_sequence_modifications("_S[Phospho (STY)]TC[Carbamidomethyl (C)]K_.2")
        assert r["clean_sequence"] == "STCK"
        assert r["phospho_positions"] == [1]
        assert set(r["all_modifications"]) == {"Phospho (STY)", "Carbamidomethyl (C)"}


# ---------------------------------------------------------------------------
# calculate_phospho_positions
# ---------------------------------------------------------------------------


class TestCalculatePhosphoPositions:
    @pytest.mark.parametrize(
        ("sequence", "expected"),
        [
            ("S[Phospho (STY)]TSK", [1]),
            ("S[Phospho (STY)]TS[Phospho (STY)]K", [1, 3]),
            ("PEPTIDES[Phospho (STY)]", [8]),
        ],
    )
    def test_positions_extracted(self, sequence, expected):
        assert calculate_phospho_positions(sequence) == expected


# ---------------------------------------------------------------------------
# extract_first_valid_position
# ---------------------------------------------------------------------------


class TestExtractFirstValidPosition:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("42", 42),
            ("37;188;295", 37),
            ("37,999;188", 37),
        ],
    )
    def test_valid_forms(self, raw, expected):
        assert extract_first_valid_position(raw) == expected

    def test_leading_none_string_treated_as_missing(self):
        assert extract_first_valid_position("None;42") is None

    def test_garbage(self):
        assert extract_first_valid_position("abc") is None


# ---------------------------------------------------------------------------
# parse_localization_probabilities
# ---------------------------------------------------------------------------


class TestParseLocalizationProbabilities:
    def test_single_position(self):
        s = "_PVS[Phospho (STY): 92.3%]PSK_"
        r = parse_localization_probabilities(s)
        assert set(r.keys()) == {3}
        assert r[3] == pytest.approx(0.923)

    def test_multiple_positions(self):
        s = "_PVS[Phospho (STY): 92.3%]PS[Phospho (STY): 7.6%]K_"
        r = parse_localization_probabilities(s)
        assert set(r.keys()) == {3, 5}
        assert r[3] == pytest.approx(0.923)
        assert r[5] == pytest.approx(0.076)

    def test_ignores_non_phospho_brackets(self):
        s = "_C[Carbamidomethyl (C)]S[Phospho (STY): 88.4%]TK_"
        r = parse_localization_probabilities(s)
        assert set(r.keys()) == {2}
        assert r[2] == pytest.approx(0.884)

    def test_zero_prob(self):
        s = "_S[Phospho (STY): 0.0%]TK_"
        assert parse_localization_probabilities(s) == {1: 0.0}


# ---------------------------------------------------------------------------
# rank_select_positions
# ---------------------------------------------------------------------------


class TestRankSelectPositions:
    def test_basic_ordering(self):
        pos, prob = rank_select_positions({3: 0.9, 8: 0.1}, 1)
        assert pos == [3]
        assert prob == [0.9]

    def test_ties_break_by_position(self):
        pos, _prob = rank_select_positions({8: 0.5, 3: 0.5, 12: 0.2}, 2)
        # Both 0.5 -- ascending position wins the tie
        assert pos == [3, 8]

    def test_n_larger_than_available(self):
        pos, prob = rank_select_positions({3: 0.7}, 5)
        assert pos == [3]
        assert prob == [0.7]

    def test_empty_input(self):
        assert rank_select_positions({}, 3) == ([], [])
        assert rank_select_positions({3: 0.9}, 0) == ([], [])
