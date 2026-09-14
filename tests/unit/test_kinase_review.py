"""Regression tests from the 2026-09-14 review of :mod:`alphaphos.kinase`.

These run WITHOUT the optional ``kinase_library`` package: FASTA window
annotation (``annotation.py`` had no direct tests), the pure helpers of the
KSEA wrappers, and the import-side-effect guard.
"""

from __future__ import annotations

import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphaphos.kinase.annotation import (
    add_kinase_windows,
    extract_window,
    load_fasta,
    resolve_fasta_accession,
)
from alphaphos.kinase.enrichment import _build_seq_frame, _merge_in_seq, _run_pools
from alphaphos.kinase.library import _strip_alphaphos_markers, kl_context, require_kinase_library

SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHYVQGRLNTF"


@pytest.fixture
def fasta(tmp_path):
    p = tmp_path / "test.fasta"
    p.write_text(
        ">sp|P00001|PROT1_HUMAN Test protein OS=Homo sapiens\n"
        + "\n".join(SEQ[i : i + 60] for i in range(0, len(SEQ), 60))
        + "\n\n>P00002 bare header protein\nMSSSTTTYYY\n"
        + ">tr|A0A000|A0A000_HUMAN lower case seq\nmkt\n"
    )
    return p


class TestLoadFasta:
    def test_header_conventions_and_joining(self, fasta):
        d = load_fasta(fasta)
        assert set(d) == {"P00001", "P00002", "A0A000"}
        assert d["P00001"] == SEQ  # continuation lines joined
        assert d["A0A000"] == "MKT"  # upper-cased

    def test_empty_raises(self, tmp_path):
        p = tmp_path / "e.fasta"
        p.write_text("\n")
        with pytest.raises(ValueError, match="No sequences"):
            load_fasta(p)
        with pytest.raises(FileNotFoundError):
            load_fasta(tmp_path / "missing.fasta")


class TestExtractWindow:
    def test_interior_site_is_19_chars_uppercase_centre(self):
        d = {"P": SEQ}
        pos = 13  # 'S' in MKTAYIAKQRQIS (1-indexed)
        assert SEQ[pos - 1] == "S"
        w = extract_window(d, "P", pos, "S")
        assert w == "_" + SEQ[pos - 8 : pos - 1] + "*S*" + SEQ[pos : pos + 7] + "_"
        assert len(w) == 19 and len(_strip_alphaphos_markers(w)) == 17

    def test_terminal_padding(self):
        d = {"P": "MSKTAYIAKQRQISF"}
        w_n = extract_window(d, "P", 2, "S")  # N-terminal: 6 missing residues
        assert w_n.startswith("_" + "_" * 6 + "M*S*")
        assert len(w_n) == 19
        w_c = extract_window(d, "P", 14, "S")  # C-terminal: 6 missing residues
        assert w_c.endswith("*S*F" + "_" * 6 + "_") and len(w_c) == 19

    def test_sentinels(self):
        d = {"P": SEQ}
        assert extract_window(d, "NOPE", 5, "S").startswith("FASTA_ERROR:")
        assert extract_window(d, "P", 0, "S").startswith("POSITION_ERROR:")
        assert extract_window(d, "P", len(SEQ) + 1, "S").startswith("POSITION_ERROR:")
        assert extract_window(d, "P", 13, "T").startswith("SEQUENCE_MISMATCH:")
        for s in ("FASTA_ERROR: x", "POSITION_ERROR: y", None, ""):
            assert _strip_alphaphos_markers(s) is None


class TestResolveAccession:
    def test_fallbacks(self):
        d = {"P00441": "MATK", "P00533": "MRPS"}
        assert resolve_fasta_accession(d, "P00441") == ("P00441", "exact")
        assert resolve_fasta_accession(d, "cRAP-P00441") == ("P00441", "contaminant_tag")
        assert resolve_fasta_accession(d, "Cont_P00441") == ("P00441", "contaminant_tag")
        assert resolve_fasta_accession(d, "P00533-2") == ("P00533", "isoform")
        assert resolve_fasta_accession(d, "cRAP-P00533-3") == ("P00533", "contaminant_tag+isoform")
        assert resolve_fasta_accession(d, "Q99999") == (None, "missing")


class TestAddKinaseWindows:
    def _adata(self):
        var = pd.DataFrame(
            {
                "protein_group_id": [
                    "P00001",
                    "cRAP-P00001",
                    "P00001-2",
                    "P00002",
                    "P00001",
                    "ZZZ",
                ],
                "site_position": [13, 13, 13, 4, 12, 3],
                "site_aa": ["S", "S", "S", "S", "S", "S"],  # pos 12 is I -> mismatch
            },
            index=[f"k{i}" for i in range(6)],
        )
        return ad.AnnData(X=np.zeros((2, 6)), var=var)

    def test_counts_fallbacks_and_provenance(self, fasta):
        out = add_kinase_windows(self._adata(), fasta_path=fasta)
        ks = out.var["kinase_sequence"]
        assert ks["k0"] == ks["k1"] == ks["k2"]  # tag / isoform fallbacks give the same window
        assert ks["k3"].startswith("_" + "_" * 4 + "MSS*S*TTTYYY")  # bare-header protein
        assert ks["k4"].startswith("SEQUENCE_MISMATCH:")
        assert ks["k5"].startswith("FASTA_ERROR:")
        prov = out.uns["alphaphos"]["kinase_annotation"]
        assert prov["n_ok"] == 4 and prov["n_sequence_mismatch"] == 1 and prov["n_fasta_error"] == 1
        assert prov["n_fallback_contaminant_tag"] == 1 and prov["n_fallback_isoform"] == 1

    def test_copy_semantics_and_missing_column(self, fasta):
        a = self._adata()
        out = add_kinase_windows(a, fasta_path=fasta, copy=True)
        assert "kinase_sequence" not in a.var.columns and "kinase_sequence" in out.var.columns
        add_kinase_windows(a, fasta_path=fasta, copy=False)
        assert "kinase_sequence" in a.var.columns
        with pytest.raises(KeyError, match="site_aa"):
            add_kinase_windows(a[:, :].copy().T.T, fasta_path=fasta, aa_col="nope")


class TestEnrichmentHelpers:
    def test_merge_in_seq_index_default_and_dedup(self):
        seqs = pd.Series(
            {
                "P1|G|S10|M1": "_AAVKRGT*S*ELLIQAA_",
                "P1|G|S10|M2": "_AAVKRGT*S*ELLIQAA_",  # multiplicity variant, same window
                "P2|G|T5|M1": "_PPRPAGT*T*PRSLEER_",
                "P3|G|S1|M1": "FASTA_ERROR: nope",
            }
        )
        df = pd.DataFrame({"t_stat": [1.0, -4.0, 2.0, 9.0]}, index=seqs.index)
        out = _merge_in_seq(df, seqs, None, dedup_sequences=False)
        assert len(out) == 3  # sentinel dropped, both variants kept
        out = _merge_in_seq(df, seqs, None, dedup_sequences=True, strength_col="t_stat")
        assert len(out) == 2
        assert out.loc[out["seq"] == "_AAVKRGTSELLIQAA_", "t_stat"].iloc[0] == -4.0  # largest |t|
        out = _merge_in_seq(
            df.assign(p=[0.5, 0.01, 0.2, 0.3]),
            seqs,
            None,
            dedup_sequences=True,
            strength_col="p",
            strength="min",
        )
        assert out.loc[out["seq"] == "_AAVKRGTSELLIQAA_", "p"].iloc[0] == 0.01

    def test_merge_in_seq_id_col_and_errors(self):
        seqs = pd.Series({"a": "_AAVKRGT*S*ELLIQAA_"})
        df = pd.DataFrame({"protein": ["a"], "log2fc": [1.0]})
        assert len(_merge_in_seq(df, seqs, "protein")) == 1
        with pytest.raises(KeyError, match="id_col='nope'"):
            _merge_in_seq(df, seqs, "nope")
        dup = pd.Series(["_AAVKRGT*S*ELLIQAA_", "_AAVKRGT*S*ELLIQAA_"], index=["a", "a"])
        with pytest.raises(ValueError, match="unique"):
            _merge_in_seq(df, dup, "protein")

    def test_build_seq_frame_dedup(self):
        seqs = pd.Series(
            {"a": "_AAVKRGT*S*ELLIQAA_", "b": "_AAVKRGT*S*ELLIQAA_", "c": "FASTA_ERROR: x"}
        )
        assert len(_build_seq_frame(["a", "b", "c"], seqs, dedup_sequences=False)) == 2
        assert len(_build_seq_frame(["a", "b", "c"], seqs, dedup_sequences=True)) == 1

    def test_run_pools_warns_and_raises(self, caplog):
        def fn(kt):
            if kt == "tyrosine":
                raise ValueError("too few sites")
            return pd.DataFrame({"NES": [1.0]})

        out = _run_pools("x", ("ser_thr", "tyrosine"), fn)
        assert list(out) == ["ser_thr"]
        assert any("tyrosine" in r.message for r in caplog.records)
        with pytest.raises(RuntimeError, match="every kinase pool failed"):
            _run_pools("x", ("tyrosine",), fn)

    def test_kl_context_quiet_swallows_stdout_and_stderr(self, capsys):
        with kl_context(quiet=True):
            print("noise")
            print("bar", file=sys.stderr)
        with kl_context(quiet=False):
            print("signal")
        with pytest.raises(ValueError), kl_context(quiet=True):
            raise ValueError("propagates")
        captured = capsys.readouterr()
        assert captured.out == "signal\n" and captured.err == ""

    def test_require_kinase_library_message(self):
        pytest.importorskip("kinase_library", reason="skip when installed")  # inverse guard
        # (only reached when kinase_library IS importable; nothing to assert)

    def test_require_message_when_missing(self, monkeypatch):
        import builtins

        real = builtins.__import__

        def fake(name, *a, **k):
            if name == "kinase_library":
                raise ImportError("no")
            return real(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake)
        with pytest.raises(ImportError, match="--no-deps kinase-library"):
            require_kinase_library("score_kinases")


def test_import_alphaphos_does_not_touch_streams():
    code = (
        "import sys; b=(sys.stdout.errors, sys.stderr.errors); "
        "import alphaphos, alphaphos.kinase.enrichment; "
        "print(b == (sys.stdout.errors, sys.stderr.errors))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"
