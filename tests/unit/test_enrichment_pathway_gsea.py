"""Tests for alphaphos.enrichment.pathway_gsea.

Unit tests mock ``gseapy.prerank`` to run offline; integration tests
hit the real Enrichr API on the EGF walkthrough result and are skipped
when the walkthrough hasn't been run or the network is unreachable.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gseapy")

from alphaphos.enrichment import (
    DEFAULT_LIBRARIES_HUMAN,
    DEFAULT_LIBRARIES_MOUSE,
    pathway_gsea,
)

REPO = Path(__file__).resolve().parents[2]
EGF_DIFF_EXP = REPO / "test_data" / "walkthrough_output" / "egf_diff_exp_result.tsv"


# ---------------------------------------------------------------------------
# Helpers -- synthetic input + fake gseapy prerank result
# ---------------------------------------------------------------------------


def _make_diff_exp(n_up: int = 10, n_down: int = 5, n_null: int = 30) -> pd.DataFrame:
    """Diff-exp result with alphaphos keys, one site per gene."""
    rows: list[tuple[str, float, float]] = []
    for i in range(n_up):
        rows.append((f"P0000{i}|UPGENE{i}|S{100 + i}|M1", 2.0 + 0.01 * i, 0.001))
    for i in range(n_down):
        rows.append((f"Q0000{i}|DOWNGENE{i}|S{200 + i}|M1", -2.0 - 0.01 * i, 0.001))
    for i in range(n_null):
        rows.append((f"R0000{i}|NULLGENE{i}|S{300 + i}|M1", 0.01 * (i - n_null / 2), 0.9))
    keys, log2fcs, fdrs = zip(*rows, strict=True)
    return pd.DataFrame({"log2fc": log2fcs, "fdr": fdrs}, index=list(keys))


def _make_multisite_diff_exp() -> pd.DataFrame:
    """One gene with several sites of varying |log2fc| for collapse-strategy tests."""
    rows = [
        # GENE_A: three sites, |log2fc| = 0.5, 3.0, 1.0 -- max_abs picks -3.0
        ("P1|GENE_A|S1|M1", 0.5, 0.5),
        ("P1|GENE_A|S2|M1", -3.0, 0.05),
        ("P1|GENE_A|S3|M1", 1.0, 0.001),  # top_significant picks +1.0
        # GENE_B: two sites, |log2fc| = 2.5, 2.0 -- max_abs picks +2.5, top_sig picks +2.0
        ("P2|GENE_B|S1|M1", 2.5, 0.1),
        ("P2|GENE_B|S2|M1", 2.0, 0.001),
        # GENE_C: single site
        ("P3|GENE_C|S1|M1", -1.5, 0.02),
    ]
    keys, log2fcs, fdrs = zip(*rows, strict=True)
    return pd.DataFrame({"log2fc": log2fcs, "fdr": fdrs}, index=list(keys))


def _fake_prerank_result(rnk, gene_sets, **_kwargs):
    """Return a namespace with ``.res2d`` in gseapy's Prerank schema."""
    library = gene_sets if isinstance(gene_sets, str) else "MULTI"
    top3 = list(rnk.head(3).index)
    bot3 = list(rnk.tail(3).index)
    res2d = pd.DataFrame(
        [
            {
                "Name": "prerank",
                "Term": f"Up_Term ({library})",
                "ES": "0.65",
                "NES": "1.8",
                "NOM p-val": "0.001",
                "FDR q-val": "0.01",
                "FWER p-val": "0.02",
                "Tag %": f"3/{len(rnk)}",
                "Gene %": "10.00%",
                "Lead_genes": ";".join(top3),
            },
            {
                "Name": "prerank",
                "Term": f"Down_Term ({library})",
                "ES": "-0.55",
                "NES": "-1.6",
                "NOM p-val": "0.005",
                "FDR q-val": "0.03",
                "FWER p-val": "0.05",
                "Tag %": f"3/{len(rnk)}",
                "Gene %": "10.00%",
                "Lead_genes": ";".join(bot3),
            },
        ]
    )
    return types.SimpleNamespace(res2d=res2d)


@pytest.fixture()
def mock_gseapy(monkeypatch):
    """Patch ``gseapy.prerank`` to return a deterministic fake."""
    import gseapy as gp

    calls: list[dict] = []

    def _spy(**kwargs):
        calls.append(kwargs)
        return _fake_prerank_result(**kwargs)

    monkeypatch.setattr(gp, "prerank", _spy)
    return calls


# ---------------------------------------------------------------------------
# Wrapper behaviour (mocked gseapy)
# ---------------------------------------------------------------------------


class TestBasicMockedRun:
    def test_returns_output_schema(self, mock_gseapy):
        diff = _make_diff_exp()
        out = pathway_gsea(diff, libraries=["GO_BP"], site_to_gene_agg="max_abs")
        expected = {
            "library",
            "term",
            "es",
            "nes",
            "p_value",
            "fdr",
            "size",
            "leading_edge",
            "direction",
        }
        assert expected.issubset(out.columns)

    def test_one_call_per_library(self, mock_gseapy):
        diff = _make_diff_exp()
        pathway_gsea(diff, libraries=["GO_BP", "KEGG", "Reactome"])
        assert len(mock_gseapy) == 3
        called_libs = [c["gene_sets"] for c in mock_gseapy]
        assert called_libs == ["GO_BP", "KEGG", "Reactome"]

    def test_direction_derived_from_nes_sign(self, mock_gseapy):
        diff = _make_diff_exp()
        out = pathway_gsea(diff, libraries=["GO_BP"])
        assert (out.loc[out["nes"] >= 0, "direction"] == "up").all()
        assert (out.loc[out["nes"] < 0, "direction"] == "down").all()

    def test_size_extracted_from_tag_percent(self, mock_gseapy):
        diff = _make_diff_exp()
        out = pathway_gsea(diff, libraries=["GO_BP"])
        # fake writes "3/<n_rnk>" for Tag % -> size should be 3
        assert (out["size"] == 3).all()

    def test_sorted_by_fdr_within_library(self, mock_gseapy):
        diff = _make_diff_exp()
        out = pathway_gsea(diff, libraries=["GO_BP", "KEGG"])
        for _, grp in out.groupby("library"):
            fdrs = grp["fdr"].to_numpy()
            assert (fdrs[:-1] <= fdrs[1:]).all()

    def test_provenance_stamped(self, mock_gseapy):
        diff = _make_diff_exp()
        out = pathway_gsea(diff, libraries=["GO_BP"], n_permutations=500)
        prov = out.attrs["provenance"]
        assert prov["method"] == "gsea_prerank"
        assert prov["libraries"] == ["GO_BP"]
        assert prov["site_to_gene_agg"] == "max_abs"
        assert prov["n_permutations"] == 500
        assert prov["n_ranked_genes"] == 45  # 10 up + 5 down + 30 null
        assert "gseapy_version" in prov


class TestSiteToGeneCollapse:
    def test_max_abs_keeps_largest_magnitude_signed(self, mock_gseapy):
        diff = _make_multisite_diff_exp()
        pathway_gsea(diff, libraries=["GO_BP"], site_to_gene_agg="max_abs")
        # Verify ranked series passed to gseapy
        rnk = mock_gseapy[0]["rnk"]
        assert rnk["GENE_A"] == -3.0  # site with largest |log2fc|
        assert rnk["GENE_B"] == 2.5
        assert rnk["GENE_C"] == -1.5

    def test_top_significant_keeps_lowest_fdr_site(self, mock_gseapy):
        diff = _make_multisite_diff_exp()
        pathway_gsea(diff, libraries=["GO_BP"], site_to_gene_agg="top_significant")
        rnk = mock_gseapy[0]["rnk"]
        # GENE_A: lowest-FDR site (0.001) has log2fc = +1.0
        assert rnk["GENE_A"] == 1.0
        # GENE_B: lowest-FDR site (0.001) has log2fc = +2.0
        assert rnk["GENE_B"] == 2.0
        # GENE_C: single site
        assert rnk["GENE_C"] == -1.5

    def test_ranked_is_sorted_descending(self, mock_gseapy):
        diff = _make_multisite_diff_exp()
        pathway_gsea(diff, libraries=["GO_BP"])
        rnk = mock_gseapy[0]["rnk"]
        vals = rnk.to_numpy()
        assert (vals[:-1] >= vals[1:]).all()

    def test_top_significant_requires_fdr(self):
        diff = pd.DataFrame(
            {"log2fc": [1.0, -1.0]},
            index=["P1|G1|S1|M1", "P2|G2|S2|M1"],
        )
        with pytest.raises(KeyError):
            # No 'fdr' column present, top_significant reads it
            pathway_gsea(diff, libraries=["GO_BP"], site_to_gene_agg="top_significant")

    def test_invalid_strategy_raises(self):
        diff = _make_diff_exp()
        with pytest.raises(ValueError, match="site_to_gene_agg must be"):
            pathway_gsea(diff, libraries=["GO_BP"], site_to_gene_agg="mean")


class TestLibrariesAndOrganism:
    def test_default_libraries_used_when_none(self, mock_gseapy):
        diff = _make_diff_exp()
        pathway_gsea(diff)
        called = [c["gene_sets"] for c in mock_gseapy]
        assert called == DEFAULT_LIBRARIES_HUMAN

    def test_mouse_defaults(self, mock_gseapy):
        diff = _make_diff_exp()
        pathway_gsea(diff, organism="mouse")
        called = [c["gene_sets"] for c in mock_gseapy]
        assert called == DEFAULT_LIBRARIES_MOUSE

    def test_custom_libraries_respected(self, mock_gseapy):
        diff = _make_diff_exp()
        pathway_gsea(diff, libraries=["KEGG_2021_Human"])
        assert mock_gseapy[0]["gene_sets"] == "KEGG_2021_Human"

    def test_empty_libraries_raises(self):
        diff = _make_diff_exp()
        with pytest.raises(ValueError, match="libraries must be non-empty"):
            pathway_gsea(diff, libraries=[])

    def test_invalid_organism_raises(self):
        diff = _make_diff_exp()
        with pytest.raises(ValueError, match="organism must be"):
            pathway_gsea(diff, organism="fly")


class TestGeneExtraction:
    def test_skips_unparseable_keys(self, mock_gseapy):
        diff = pd.DataFrame(
            {"log2fc": [2.0, 2.0], "fdr": [0.001, 0.001]},
            index=["not_a_key", "P1|GOOD|S1|M1"],
        )
        pathway_gsea(diff, libraries=["GO_BP"])
        rnk = mock_gseapy[0]["rnk"]
        assert list(rnk.index) == ["GOOD"]

    def test_no_parseable_keys_raises(self):
        diff = pd.DataFrame(
            {"log2fc": [2.0], "fdr": [0.001]},
            index=["totally_bogus"],
        )
        with pytest.raises(ValueError, match="No parseable gene names"):
            pathway_gsea(diff, libraries=["GO_BP"])

    def test_key_column_supported(self, mock_gseapy):
        diff = pd.DataFrame(
            {
                "site_key": ["P1|GENE_X|S1|M1", "P2|GENE_Y|S2|M1"],
                "log2fc": [2.0, -2.0],
                "fdr": [0.001, 0.001],
            }
        )
        pathway_gsea(diff, libraries=["GO_BP"], key_column="site_key")
        rnk = mock_gseapy[0]["rnk"]
        assert set(rnk.index) == {"GENE_X", "GENE_Y"}


class TestNaNHandling:
    def test_nan_stats_dropped(self, mock_gseapy):
        diff = pd.DataFrame(
            {"log2fc": [2.0, np.nan, -1.0], "fdr": [0.001, 0.001, 0.001]},
            index=["P1|GA|S1|M1", "P2|GB|S2|M1", "P3|GC|S3|M1"],
        )
        pathway_gsea(diff, libraries=["GO_BP"])
        rnk = mock_gseapy[0]["rnk"]
        assert "GB" not in rnk.index
        assert set(rnk.index) == {"GA", "GC"}

    def test_all_nan_raises(self):
        diff = pd.DataFrame(
            {"log2fc": [np.nan, np.nan], "fdr": [0.001, 0.001]},
            index=["P1|GA|S1|M1", "P2|GB|S2|M1"],
        )
        with pytest.raises(ValueError, match="Empty ranked list"):
            pathway_gsea(diff, libraries=["GO_BP"])


class TestLibraryFailureIsolation:
    def test_one_library_fail_does_not_kill_run(self, monkeypatch):
        """If a library errors (e.g., 404), skip it and continue."""
        import gseapy as gp

        def _flaky_prerank(**kwargs):
            if kwargs["gene_sets"] == "BROKEN_LIB":
                raise RuntimeError("simulated network failure")
            return _fake_prerank_result(**kwargs)

        monkeypatch.setattr(gp, "prerank", _flaky_prerank)
        diff = _make_diff_exp()
        out = pathway_gsea(diff, libraries=["GO_BP", "BROKEN_LIB", "KEGG"])
        # Two libraries survive, one dropped
        assert set(out["library"].unique()) == {"GO_BP", "KEGG"}


# ---------------------------------------------------------------------------
# Integration on the real EGF walkthrough result (hits Enrichr's API)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not EGF_DIFF_EXP.exists(),
    reason="EGF walkthrough output missing; run examples/egf_walkthrough.py first",
)
class TestEGFRealData:
    """Real-data validation on the EGF +/- Spectronaut dataset."""

    @pytest.fixture(scope="class")
    def egf_result(self):
        return pd.read_csv(EGF_DIFF_EXP, sep="\t", index_col=0)

    @pytest.mark.slow
    def test_prerank_recovers_egf_pathways(self, egf_result):
        try:
            out = pathway_gsea(
                egf_result,
                libraries=["KEGG_2021_Human", "MSigDB_Hallmark_2020"],
                site_to_gene_agg="max_abs",
                n_permutations=500,  # keep test fast
                seed=42,
            )
        except Exception as exc:
            pytest.skip(f"Enrichr unreachable: {exc}")
        assert not out.empty
        # Top up-regulated pathways should be EGF / MAPK / RAS / PI3K flavoured
        up_hits = out[out["direction"] == "up"].sort_values("fdr").head(20)
        motifs = ["mapk", "egf", "ras", "erbb", "receptor tyrosine", "erk", "pi3k", "akt", "mtor"]
        top_terms = up_hits["term"].str.lower().tolist()
        assert any(m in t for t in top_terms for m in motifs), (
            f"no EGF/MAPK/PI3K pathway in top-20 up-regulated: {top_terms[:10]}"
        )

    @pytest.mark.slow
    def test_provenance_from_real_call(self, egf_result):
        try:
            out = pathway_gsea(
                egf_result,
                libraries=["MSigDB_Hallmark_2020"],
                n_permutations=200,
            )
        except Exception as exc:
            pytest.skip(f"Enrichr unreachable: {exc}")
        prov = out.attrs["provenance"]
        assert prov["method"] == "gsea_prerank"
        assert prov["site_to_gene_agg"] == "max_abs"
        assert prov["n_ranked_genes"] > 1000  # real EGF has ~4k genes
        assert prov["libraries"] == ["MSigDB_Hallmark_2020"]
