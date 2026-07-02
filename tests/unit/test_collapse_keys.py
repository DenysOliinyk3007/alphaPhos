"""Unit tests for alphaphos.preprocess._collapse.keys.

These functions build the canonical ``Protein|Gene|Site|Mult`` identifier
and its variants. The tests verify format stability (downstream code
relies on the exact ``|``-delimited layout) and the collision-resolution
determinism (suffixing is done in a well-defined order).
"""

from __future__ import annotations

import pytest

from alphaphos.preprocess._collapse.keys import (
    build_full_key,
    build_modified_sequence,
    build_pg_key,
    build_short_key,
    get_phospho_amino_acid,
    resolve_short_key_collisions,
)

# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------


class TestBuildFullKey:
    def test_format(self):
        k = build_full_key("A0A0B4J2F2", "SIK1B", "S", 575, 1)
        assert k == "A0A0B4J2F2|SIK1B|S575|M1"

    def test_multi_gene_input_kept_verbatim(self):
        # Multi-value strings should have been split BEFORE this function
        # (caller responsibility). If not, we don't re-split; we produce
        # the key with the raw string. This keeps the function pure.
        k = build_full_key("P1;P2", "GENE1;GENE2", "T", 42, 2)
        assert k == "P1;P2|GENE1;GENE2|T42|M2"

    def test_error_on_bad_position(self):
        assert build_full_key("PG", "G", "S", "not-an-int", 1) == "Error_key"

    def test_error_on_bad_multiplicity(self):
        assert build_full_key("PG", "G", "S", 100, "not-int") == "Error_key"


class TestBuildShortKey:
    def test_format(self):
        assert build_short_key("SIK1B", "S", 575, 1) == "SIK1B|S575|M1"

    def test_error_on_bad_input(self):
        assert build_short_key("G", "S", "?", 1) == "Error_short_key"


class TestBuildPgKey:
    def test_format(self):
        assert build_pg_key("A0A0B4J2F2", "S", 575, 1) == "A0A0B4J2F2|S575|M1"


# ---------------------------------------------------------------------------
# Modified sequence + AA lookup
# ---------------------------------------------------------------------------


class TestBuildModifiedSequence:
    @pytest.mark.parametrize(
        ("sequence", "pos", "expected"),
        [
            ("PEPTIDE", 4, "PEPt*IDE"),  # middle
            ("PEPTIDE", 1, "p*EPTIDE"),  # start
            ("PEPTIDE", 7, "PEPTIDe*"),  # end
            ("PEPTIDE", 0, "PEPTIDE"),   # out of range left: no-op
            ("PEPTIDE", 999, "PEPTIDE"), # out of range right: no-op
        ],
    )
    def test_star_marker_placement(self, sequence, pos, expected):
        assert build_modified_sequence(sequence, pos) == expected


class TestGetPhosphoAminoAcid:
    def test_valid(self):
        assert get_phospho_amino_acid("PEPTIDES", 1) == "P"
        assert get_phospho_amino_acid("PEPTIDES", 8) == "S"

    def test_out_of_range(self):
        assert get_phospho_amino_acid("PEP", 0) == "X"
        assert get_phospho_amino_acid("PEP", 99) == "X"


# ---------------------------------------------------------------------------
# Collision resolution
# ---------------------------------------------------------------------------


class TestResolveShortKeyCollisions:
    def test_no_duplicates(self):
        shorts = ["A|S1|M1", "B|T2|M1", "C|Y3|M1"]
        fulls = ["P1|A|S1|M1", "P2|B|T2|M1", "P3|C|Y3|M1"]
        resolved, collisions = resolve_short_key_collisions(shorts, fulls)
        assert resolved == shorts
        assert collisions == []

    def test_single_duplicate_pair(self):
        # Two rows collide -- first (by full_key ordering) keeps the plain
        # short_key; the other gets '#2'.
        shorts = ["G|S1|M1", "G|S1|M1", "OTHER|T2|M1"]
        fulls = ["P2|G|S1|M1", "P1|G|S1|M1", "P3|OTHER|T2|M1"]
        resolved, collisions = resolve_short_key_collisions(shorts, fulls)
        # P1 came first alphabetically, so its short stays; P2 gets #2.
        assert resolved == ["G|S1|M1#2", "G|S1|M1", "OTHER|T2|M1"]
        assert collisions == [("G|S1|M1", ["P1|G|S1|M1", "P2|G|S1|M1"])]

    def test_three_way_collision(self):
        shorts = ["G|S1|M1"] * 3
        fulls = ["P3|G|S1|M1", "P1|G|S1|M1", "P2|G|S1|M1"]
        resolved, _collisions = resolve_short_key_collisions(shorts, fulls)
        assert resolved == ["G|S1|M1#3", "G|S1|M1", "G|S1|M1#2"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            resolve_short_key_collisions(["A", "B"], ["X"])
