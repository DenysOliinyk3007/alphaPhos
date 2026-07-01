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
    def test_unmodified_peptide(self):
        r = extract_sequence_modifications("_PEPTIDE_.2")
        assert r["clean_sequence"] == "PEPTIDE"
        assert r["phospho_positions"] == []
        assert r["phospho_count"] == 0
        assert r["all_modifications"] == []

    def test_single_phospho(self):
        r = extract_sequence_modifications("_S[Phospho (STY)]TSK_.3")
        assert r["clean_sequence"] == "STSK"
        assert r["phospho_positions"] == [1]
        assert r["phospho_count"] == 1
        assert r["all_modifications"] == ["Phospho (STY)"]

    def test_multi_phospho(self):
        r = extract_sequence_modifications("_S[Phospho (STY)]TS[Phospho (STY)]K_.3")
        assert r["clean_sequence"] == "STSK"
        assert r["phospho_positions"] == [1, 3]
        assert r["phospho_count"] == 2

    def test_mixed_mods(self):
        r = extract_sequence_modifications("_S[Phospho (STY)]TC[Carbamidomethyl (C)]K_.2")
        assert r["clean_sequence"] == "STCK"
        assert r["phospho_positions"] == [1]
        assert r["phospho_count"] == 1
        # Both mods captured; non-phospho ignored for position derivation
        assert set(r["all_modifications"]) == {"Phospho (STY)", "Carbamidomethyl (C)"}

    def test_wrapper_stars(self):
        """Alt Spectronaut wrapper: ``*..*.charge``."""
        r = extract_sequence_modifications("*.S[Phospho (STY)]TSK.*.2")
        assert r["clean_sequence"] == "STSK"
        assert r["phospho_positions"] == [1]


# ---------------------------------------------------------------------------
# calculate_phospho_positions
# ---------------------------------------------------------------------------


class TestCalculatePhosphoPositions:
    def test_empty_string(self):
        assert calculate_phospho_positions("") == []

    def test_no_phospho_marker(self):
        assert calculate_phospho_positions("PEPTIDE") == []

    def test_single(self):
        assert calculate_phospho_positions("S[Phospho (STY)]TSK") == [1]

    def test_multiple(self):
        assert calculate_phospho_positions("S[Phospho (STY)]TS[Phospho (STY)]K") == [1, 3]

    def test_terminal_position(self):
        assert calculate_phospho_positions("PEPTIDES[Phospho (STY)]") == [8]


# ---------------------------------------------------------------------------
# extract_first_valid_position
# ---------------------------------------------------------------------------


class TestExtractFirstValidPosition:
    def test_single_int(self):
        assert extract_first_valid_position("42") == 42

    def test_semicolon_separated(self):
        assert extract_first_valid_position("37;188;295") == 37

    def test_comma_within_first(self):
        assert extract_first_valid_position("37,999;188") == 37

    def test_leading_empty(self):
        assert extract_first_valid_position("None;42") is None  # First is 'None'

    def test_missing_forms(self):
        assert extract_first_valid_position(None) is None
        assert extract_first_valid_position(float("nan")) is None
        assert extract_first_valid_position("") is None
        assert extract_first_valid_position("None") is None

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
        # Carbamidomethyl bracket is skipped but occupies the 'C' residue.
        assert set(r.keys()) == {2}
        assert r[2] == pytest.approx(0.884)

    def test_zero_prob(self):
        s = "_S[Phospho (STY): 0.0%]TK_"
        assert parse_localization_probabilities(s) == {1: 0.0}

    def test_missing_or_bad_input(self):
        assert parse_localization_probabilities(None) == {}
        assert parse_localization_probabilities(float("nan")) == {}
        assert parse_localization_probabilities(42) == {}
        assert parse_localization_probabilities("") == {}


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
