"""Tests for alphaphos.preprocess.contaminants.

Covers:
  - parse_fasta_accessions       : header parsing across the 4 MaxQuant
                                   header formats (UniProt, TREMBL with
                                   secondary IDs, ENSEMBL, REFSEQ)
  - filter_contaminants          : prefix detection, FASTA detection,
                                   require_all_contaminant semantics,
                                   missing-column safety
  - get_default_contaminants_fasta : the bundled FASTA is present and parses
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaphos.preprocess.contaminants import (
    DEFAULT_CONTAMINANT_PREFIXES,
    filter_contaminants,
    get_default_contaminants_fasta,
    parse_fasta_accessions,
)

# ============================================================================
# get_default_contaminants_fasta — the bundled FASTA exists and is parseable
# ============================================================================


class TestBundledFasta:
    def test_default_fasta_exists(self):
        p = get_default_contaminants_fasta()
        assert p.exists(), f"Bundled FASTA missing at {p}"
        assert p.stat().st_size > 50_000  # ~150 KB MaxQuant FASTA

    def test_default_fasta_known_contaminants(self):
        """The bundled FASTA should contain the canonical lab contaminants."""
        p = get_default_contaminants_fasta()
        acc = parse_fasta_accessions(p)
        # Sanity check: at least the household names should be present
        for known in ("P00761", "P02769", "P19013"):  # trypsin, BSA, keratin
            assert known in acc, f"{known} missing from bundled contaminants FASTA"
        # Bundled is the MaxQuant 246-entry list — expect at least 200 accessions
        # (266 actual including secondary IDs, but be lenient for future bundles).
        assert len(acc) >= 200


# ============================================================================
# parse_fasta_accessions — header parsing
# ============================================================================


class TestParseFastaAccessions:
    @staticmethod
    def _write_fasta(tmp_path: Path, body: str) -> Path:
        p = tmp_path / "test.fasta"
        p.write_text(body, encoding="utf-8")
        return p

    def test_uniprot_header(self, tmp_path):
        p = self._write_fasta(
            tmp_path,
            ">P00761 SWISS-PROT:P00761|TRYP_PIG Trypsin - Sus scrofa (Pig).\nMKFLAA\n",
        )
        acc = parse_fasta_accessions(p)
        assert "P00761" in acc
        # TRYP_PIG is a name not an accession; should not be in the set
        assert "TRYP_PIG" not in acc

    def test_trembl_with_secondary_ids(self, tmp_path):
        p = self._write_fasta(
            tmp_path,
            ">Q32MB2 TREMBL:Q32MB2;Q86Y46 Tax_Id=9606 Gene_Symbol=KRT73 Keratin-73\nMSEQ\n",
        )
        acc = parse_fasta_accessions(p)
        assert "Q32MB2" in acc
        assert "Q86Y46" in acc  # secondary ID captured

    def test_ensembl_header(self, tmp_path):
        p = self._write_fasta(
            tmp_path,
            ">ENSEMBL:ENSBTAP00000034412 (Bos taurus) similar to C4b-binding protein\nMSEQ\n",
        )
        acc = parse_fasta_accessions(p)
        assert "ENSBTAP00000034412" in acc

    def test_refseq_header(self, tmp_path):
        p = self._write_fasta(
            tmp_path,
            ">REFSEQ:XP_585019 (Bos taurus) similar to afamin\nMSEQ\n",
        )
        acc = parse_fasta_accessions(p)
        assert "XP_585019" in acc

    def test_multiple_entries(self, tmp_path):
        p = self._write_fasta(
            tmp_path,
            ">P00761 trypsin\nMKFL\n>P02769 BSA\nMSEQ\n>P19013 keratin\nKER\n",
        )
        acc = parse_fasta_accessions(p)
        assert acc == {"P00761", "P02769", "P19013"}

    def test_prefixed_accession_also_yields_bare(self, tmp_path):
        # Newer MaxQuant fastas use "CON__P02769" as the accession token;
        # we must also index the bare "P02769" so that Spectronaut's
        # ambiguous "Cont_P02769;P02769" is fully caught by the fasta
        # lookup rather than surviving the filter.
        p = self._write_fasta(
            tmp_path,
            ">CON__P02769 SWISS-PROT:CON__P02769|ALBU_BOVIN Bovine serum albumin\nMSEQ\n"
            ">Cont_P05787 Keratin 8\nMSEQ\n"
            ">contam_P00761 Trypsin\nMKFL\n",
        )
        acc = parse_fasta_accessions(p)
        # Both prefixed AND bare forms are in the accession set
        assert "CON__P02769" in acc and "P02769" in acc
        assert "Cont_P05787" in acc and "P05787" in acc
        assert "contam_P00761" in acc and "P00761" in acc

    def test_empty_fasta(self, tmp_path):
        p = self._write_fasta(tmp_path, "")
        assert parse_fasta_accessions(p) == set()

    def test_skips_sequence_lines(self, tmp_path):
        """Sequence lines (no leading >) must not be parsed as accessions."""
        p = self._write_fasta(
            tmp_path,
            ">P00761\nMKFLAAVLLALA\nVLLSGSGAQA\n>P02769\nMQTRRLF\n",
        )
        acc = parse_fasta_accessions(p)
        assert acc == {"P00761", "P02769"}


# ============================================================================
# filter_contaminants
# ============================================================================


@pytest.fixture
def mini_fasta(tmp_path):
    """A two-entry FASTA: trypsin + BSA."""
    p = tmp_path / "mini.fasta"
    p.write_text(
        ">P00761 SWISS-PROT:P00761|TRYP_PIG Trypsin\n"
        "MKFL\n"
        ">P02769 SWISS-PROT:P02769|ALBU_BOVIN Serum albumin\n"
        "MTRRLF\n",
        encoding="utf-8",
    )
    return p


class TestFilterContaminants:
    def test_drops_pure_contaminant_row_by_fasta(self, mini_fasta):
        df = pd.DataFrame({"PG.ProteinGroups": ["P00761", "P12345"]})
        out = filter_contaminants(df, contaminants_fasta=mini_fasta)
        assert len(out) == 1
        assert out["PG.ProteinGroups"].tolist() == ["P12345"]

    def test_drops_pure_contaminant_row_by_prefix(self, mini_fasta):
        df = pd.DataFrame({"PG.ProteinGroups": ["CON__SomeUnknown", "P12345"]})
        out = filter_contaminants(df, contaminants_fasta=mini_fasta)
        assert out["PG.ProteinGroups"].tolist() == ["P12345"]

    def test_keeps_mixed_group_when_require_all(self, mini_fasta):
        """Default (require_all_contaminant=True) keeps mixed groups."""
        df = pd.DataFrame({"PG.ProteinGroups": ["P12345;P00761", "P12345;P02769"]})
        out = filter_contaminants(df, contaminants_fasta=mini_fasta)
        assert len(out) == 2

    def test_drops_mixed_group_when_not_require_all(self, mini_fasta):
        df = pd.DataFrame({"PG.ProteinGroups": ["P12345;P00761", "P67890"]})
        out = filter_contaminants(df, contaminants_fasta=mini_fasta, require_all_contaminant=False)
        assert out["PG.ProteinGroups"].tolist() == ["P67890"]

    def test_drops_all_contaminant_group(self, mini_fasta):
        df = pd.DataFrame({"PG.ProteinGroups": ["P00761;P02769", "P12345"]})
        out = filter_contaminants(df, contaminants_fasta=mini_fasta)
        assert out["PG.ProteinGroups"].tolist() == ["P12345"]

    def test_keeps_rows_with_unknown_protein_group(self, mini_fasta):
        # Empty and NaN both trip the same "no info to decide, default keep"
        # branch.  Test with a mixed input to cover both in one shot.
        df = pd.DataFrame({"PG.ProteinGroups": ["", float("nan"), "P12345"]})
        out = filter_contaminants(df, contaminants_fasta=mini_fasta)
        assert len(out) == 3

    def test_missing_protein_groups_column_is_no_op(self, mini_fasta):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        out = filter_contaminants(df, contaminants_fasta=mini_fasta)
        assert len(out) == 3
        assert out.equals(df)

    def test_resets_index(self, mini_fasta):
        df = pd.DataFrame(
            {"PG.ProteinGroups": ["P00761", "P12345", "P02769", "P67890"]},
            index=[10, 20, 30, 40],
        )
        out = filter_contaminants(df, contaminants_fasta=mini_fasta)
        assert out.index.tolist() == [0, 1]

    def test_uses_bundled_fasta_when_none(self):
        """Passing contaminants_fasta=None should load the bundled MaxQuant FASTA."""
        df = pd.DataFrame({"PG.ProteinGroups": ["P00761", "P12345"]})
        # Trypsin (P00761) is in the bundled FASTA → should be dropped
        out = filter_contaminants(df, contaminants_fasta=None)
        assert out["PG.ProteinGroups"].tolist() == ["P12345"]

    def test_empty_prefixes_disables_prefix_detection(self, mini_fasta):
        df = pd.DataFrame({"PG.ProteinGroups": ["CON__SomeUnknown", "P12345"]})
        out = filter_contaminants(df, contaminants_fasta=mini_fasta, prefix_patterns=())
        # CON__SomeUnknown not in fasta and no prefix detection → kept
        assert len(out) == 2

    def test_all_three_prefixes_recognized(self, mini_fasta):
        df = pd.DataFrame(
            {
                "PG.ProteinGroups": [
                    "CON__X",
                    "Cont_Y",
                    "contam_Z",
                    "P12345",
                ]
            }
        )
        out = filter_contaminants(df, contaminants_fasta=mini_fasta)
        assert out["PG.ProteinGroups"].tolist() == ["P12345"]

    def test_default_prefixes_constant(self):
        """The DEFAULT_CONTAMINANT_PREFIXES tuple is the public contract."""
        assert DEFAULT_CONTAMINANT_PREFIXES == ("CON__", "Cont_", "contam_")

    def test_custom_column_name(self, mini_fasta):
        df = pd.DataFrame({"my_proteins": ["P00761", "P12345"]})
        out = filter_contaminants(
            df, contaminants_fasta=mini_fasta, protein_groups_column="my_proteins"
        )
        assert out["my_proteins"].tolist() == ["P12345"]
