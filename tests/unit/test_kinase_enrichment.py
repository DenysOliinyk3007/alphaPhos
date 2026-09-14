"""Tests for alphaphos.kinase.enrichment.

Covers the three KSEA / kinase-activity wrappers:
  - kinase_enrichment_from_diffexp : Fisher per direction (up/down)
  - kinase_mea                      : GSEA-style weighted K-S
  - kinase_enrichment_binary        : foreground/background Fisher
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Skip the whole module if kinase-library isn't installed
pytest = __import__("pytest")
pytest.importorskip("kinase_library")

from alphaphos.kinase.enrichment import (  # noqa: E402  importorskip first
    _build_seq_frame,
    _merge_in_seq,
    kinase_enrichment_binary,
    kinase_enrichment_from_diffexp,
    kinase_mea,
)

# ---- shared fixtures ------------------------------------------------------


def _make_diff_results(n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic diff_exp table + matching sequence lookup with diverse sequences.

    Builds n unique 19-char sequences in alphaPhos format
    ``"_<7 left aa>*<X>*<7 right aa>_"`` so they survive the marker-strip
    (becoming 17-char odd strings) and have varied flanking residues so
    GSEA's min_size=5 is satisfied per kinase family.
    """
    rng = np.random.default_rng(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    ids = [f"site{i}" for i in range(n)]
    df = pd.DataFrame(
        {
            "protein": ids,
            "log2fc": rng.normal(0, 2, n),
            "fdr": rng.uniform(0, 1, n) ** 3,
        }
    )

    def _rand_flank() -> str:
        return "".join(rng.choice(list(aa), 7))

    # 75% S/T centers, 25% Y centers — typical phospho ratio
    centers = rng.choice(["S", "T", "Y"], n, p=[0.5, 0.25, 0.25])
    seqs = {}
    for sid, c in zip(ids, centers, strict=True):
        seqs[sid] = f"_{_rand_flank()}*{c}*{_rand_flank()}_"
    return df, pd.Series(seqs)


# ============================================================================
# _build_seq_frame
# ============================================================================


class TestBuildSeqFrame:
    def test_strips_markers_and_filters_invalid(self):
        ids = ["a", "b", "c", "d"]
        # alphaPhos format: 19 chars (1 + 7 + 3 + 7 + 1), 17 after strip (odd)
        seqs = pd.Series(
            {
                "a": "_AAVKRGT*S*ELLIQAA_",  # valid 19-mer -> 17 after strip
                "b": "FASTA_ERROR: nope",
                "c": None,
                "d": "_PPRPAGT*T*PRSLEER_",  # valid
            }
        )
        frame = _build_seq_frame(ids, seqs)
        assert len(frame) == 2
        assert set(frame["id"]) == {"a", "d"}
        assert frame.iloc[0]["seq"] == "_AAVKRGTSELLIQAA_"


# ============================================================================
# _merge_in_seq
# ============================================================================


class TestMergeInSeq:
    def test_adds_seq_and_drops_invalid_rows(self):
        df = pd.DataFrame(
            {
                "protein": ["a", "b", "c"],
                "log2fc": [1.0, 2.0, 3.0],
            }
        )
        seqs = pd.Series(
            {
                "a": "_AAVKRGT*S*ELLIQAA_",  # valid 19-mer
                "b": "FASTA_ERROR: nope",
                "c": "_PPRPAGT*T*PRSLEER_",  # valid
            }
        )
        out = _merge_in_seq(df, seqs, id_col="protein")
        assert len(out) == 2
        assert "seq" in out.columns
        assert set(out["protein"]) == {"a", "c"}


# ============================================================================
# kinase_enrichment_from_diffexp (DiffPhos KSEA)
# ============================================================================


class TestDiffphosKsea:
    def test_returns_per_kintype_with_combined_columns(self):
        df, seq = _make_diff_results()
        out = kinase_enrichment_from_diffexp(df, seq, id_col="protein")

        assert isinstance(out, dict)
        # ser_thr should always come back (the synthetic data has S/T sites)
        assert "ser_thr" in out
        st = out["ser_thr"]
        assert isinstance(st, pd.DataFrame)
        # 311 ser_thr kinases in the library
        assert len(st) == 311
        # Key columns we promise
        for col in (
            "fg_counts_upreg",
            "fg_counts_downreg",
            "log2_freq_factor_upreg",
            "log2_freq_factor_downreg",
            "fisher_adj_pval_upreg",
            "fisher_adj_pval_downreg",
            "most_sig_direction",
            "most_sig_log2_freq_factor",
            "most_sig_fisher_adj_pval",
        ):
            assert col in st.columns, col

    def test_threshold_arguments_pass_through(self):
        df, seq = _make_diff_results()
        # Tight thresholds — fewer sites should hit fg, so log2_freq_factors
        # will move (we don't pin exact numbers, just that the call succeeds)
        out = kinase_enrichment_from_diffexp(
            df, seq, id_col="protein", lfc_thresh=1.5, pval_thresh=0.01, kl_thresh=95
        )
        assert "ser_thr" in out

    def test_dropped_invalid_sequences_dont_crash(self):
        df, seq = _make_diff_results()
        # Set every fourth site's seq to an error sentinel
        bad = {sid: "FASTA_ERROR: x" for i, sid in enumerate(df["protein"]) if i % 4 == 3}
        seq = pd.concat([seq, pd.Series(bad)])
        seq = seq[~seq.index.duplicated(keep="last")]
        out = kinase_enrichment_from_diffexp(df, seq, id_col="protein")
        assert "ser_thr" in out


# ============================================================================
# kinase_mea (GSEA)
# ============================================================================


class TestKinaseMea:
    def test_returns_per_kintype_with_nes_columns(self):
        df, seq = _make_diff_results()
        out = kinase_mea(df, seq, id_col="protein", permutation_num=50, threads=1)
        assert isinstance(out, dict)
        assert "ser_thr" in out
        st = out["ser_thr"]
        # Index is kinase names
        assert pd.api.types.is_string_dtype(st.index) or st.index.dtype == object
        # Headline statistic
        assert "NES" in st.columns
        assert "ES" in st.columns
        assert "p-value" in st.columns
        assert "FDR" in st.columns

    def test_min_size_filters(self):
        df, seq = _make_diff_results()
        # min_size=1000 means no kinase should have enough substrates
        # No kinase can pass min_size=1000: kinase_library either returns
        # all-NaN NES or fails per pool; when every pool fails alphaPhos now
        # raises instead of returning an empty dict.
        try:
            out = kinase_mea(
                df, seq, id_col="protein", permutation_num=20, min_size=1000, threads=1
            )
        except RuntimeError as exc:
            assert "every kinase pool failed" in str(exc)
            return
        for df_kt in out.values():
            assert df_kt["NES"].isna().all()


# ============================================================================
# kinase_enrichment_binary
# ============================================================================


class TestKinaseBinary:
    def test_basic_call(self):
        df, seq = _make_diff_results()
        all_sites = df["protein"].tolist()
        fg = all_sites[: len(all_sites) // 3]
        out = kinase_enrichment_binary(fg, all_sites, seq)
        assert isinstance(out, dict)
        assert "ser_thr" in out
        st = out["ser_thr"]
        # Standard binary enrichment output columns
        assert "log2_freq_factor" in st.columns
        assert "fisher_adj_pval" in st.columns

    def test_handles_invalid_sequences_silently(self):
        df, seq = _make_diff_results()
        all_sites = df["protein"].tolist()
        fg = all_sites[: len(all_sites) // 3]
        # Wipe out half the sequences
        seq_partial = seq.iloc[: len(seq) // 2]
        out = kinase_enrichment_binary(fg, all_sites, seq_partial)
        # Should still return something for both kin_types we asked
        assert "ser_thr" in out
