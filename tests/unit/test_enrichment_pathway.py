"""Tests for alphaphos.enrichment.pathway.

Unit tests mock ``gseapy.enrichr`` so they run offline; integration
tests hit the real Enrichr API on the EGF walkthrough result and are
skipped when the walkthrough hasn't been run or the network is
unreachable.
"""

from __future__ import annotations

import types
import warnings
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("gseapy")

from alphaphos.enrichment import (
    DEFAULT_LIBRARIES_HUMAN,
    DEFAULT_LIBRARIES_MOUSE,
    pathway_enrichment,
)

REPO = Path(__file__).resolve().parents[2]
EGF_DIFF_EXP = REPO / "test_data" / "walkthrough_output" / "egf_diff_exp_result.tsv"


# ---------------------------------------------------------------------------
# Helpers -- synthetic input + fake gseapy for mocked unit tests
# ---------------------------------------------------------------------------


def _make_diff_exp(n_up: int = 10, n_down: int = 5, n_null: int = 20) -> pd.DataFrame:
    """A tiny diff-exp result with known up / down / null genes."""
    rows: list[tuple[str, float, float]] = []
    for i in range(n_up):
        rows.append((f"P0000{i}|UPGENE{i}|S{100 + i}|M1", 2.0 + 0.01 * i, 0.001))
    for i in range(n_down):
        rows.append((f"Q0000{i}|DOWNGENE{i}|S{200 + i}|M1", -2.0 - 0.01 * i, 0.001))
    for i in range(n_null):
        rows.append((f"R0000{i}|NULLGENE{i}|S{300 + i}|M1", 0.01 * i, 0.9))
    keys, log2fcs, fdrs = zip(*rows, strict=True)
    return pd.DataFrame({"log2fc": log2fcs, "fdr": fdrs}, index=list(keys))


def _fake_enrichr_result(gene_sets, gene_list, background=None, **_kwargs):
    """Return a namespace with a ``.results`` DataFrame in Enrichr's schema."""
    libs = [gene_sets] if isinstance(gene_sets, str) else list(gene_sets)
    rows = []
    for lib in libs:
        rows.append(
            {
                "Gene_set": lib,
                "Term": f"Term_A ({lib})",
                "Overlap": f"{min(3, len(gene_list))}/50",
                "P-value": 0.001,
                "Adjusted P-value": 0.01,
                "Odds Ratio": 5.0,
                "Combined Score": 25.0,
                "Genes": ";".join(list(gene_list)[:3]),
            }
        )
        rows.append(
            {
                "Gene_set": lib,
                "Term": f"Term_B ({lib})",
                "Overlap": f"{min(2, len(gene_list))}/80",
                "P-value": 0.05,
                "Adjusted P-value": 0.1,
                "Odds Ratio": 2.0,
                "Combined Score": 6.0,
                "Genes": ";".join(list(gene_list)[:2]),
            }
        )
    return types.SimpleNamespace(results=pd.DataFrame(rows))


@pytest.fixture()
def mock_gseapy(monkeypatch):
    """Patch ``gseapy.enrichr`` with a deterministic fake so unit tests
    never touch the network."""
    import gseapy as gp

    calls: list[dict] = []

    def _spy(**kwargs):
        calls.append(kwargs)
        return _fake_enrichr_result(**kwargs)

    monkeypatch.setattr(gp, "enrichr", _spy)
    return calls


# ---------------------------------------------------------------------------
# Wrapper behaviour (mocked gseapy)
# ---------------------------------------------------------------------------


class TestBasicMockedRun:
    def test_split_returns_up_and_down_rows(self, mock_gseapy):
        diff = _make_diff_exp(n_up=10, n_down=5, n_null=20)
        out = pathway_enrichment(diff, libraries=["GO_BP"], background="phosphoproteome")
        assert set(out["direction"].unique()) == {"up", "down"}
        assert "library" in out.columns and "fdr" in out.columns
        # Two calls: one for 'up', one for 'down'
        assert len(mock_gseapy) == 2

    def test_output_schema(self, mock_gseapy):
        diff = _make_diff_exp()
        out = pathway_enrichment(diff, libraries=["GO_BP"], background="phosphoproteome")
        expected = {
            "direction",
            "library",
            "term",
            "overlap",
            "p_value",
            "fdr",
            "odds_ratio",
            "combined_score",
            "genes",
            "n_foreground",
            "n_background",
        }
        assert expected.issubset(out.columns)

    def test_sorted_by_fdr_within_group(self, mock_gseapy):
        diff = _make_diff_exp()
        out = pathway_enrichment(diff, libraries=["GO_BP"], background="phosphoproteome")
        for (_, _), grp in out.groupby(["direction", "library"]):
            fdrs = grp["fdr"].to_numpy()
            assert (fdrs[:-1] <= fdrs[1:]).all(), f"not sorted within group: {fdrs}"

    def test_provenance_is_stamped(self, mock_gseapy):
        diff = _make_diff_exp()
        out = pathway_enrichment(diff, libraries=["GO_BP"], background="phosphoproteome")
        prov = out.attrs["provenance"]
        assert prov["libraries"] == ["GO_BP"]
        assert prov["background_type"] == "phosphoproteome"
        assert prov["direction"] == "split"
        assert prov["n_foreground_per_direction"]["up"] == 10
        assert prov["n_foreground_per_direction"]["down"] == 5
        assert "gseapy_version" in prov


class TestDirectionOptions:
    def test_up_only(self, mock_gseapy):
        diff = _make_diff_exp()
        out = pathway_enrichment(
            diff, libraries=["GO_BP"], background="phosphoproteome", direction="up"
        )
        assert set(out["direction"].unique()) == {"up"}
        assert len(mock_gseapy) == 1

    def test_down_only(self, mock_gseapy):
        diff = _make_diff_exp()
        out = pathway_enrichment(
            diff, libraries=["GO_BP"], background="phosphoproteome", direction="down"
        )
        assert set(out["direction"].unique()) == {"down"}
        assert len(mock_gseapy) == 1

    def test_both_lumps_signs(self, mock_gseapy):
        diff = _make_diff_exp(n_up=10, n_down=5)
        pathway_enrichment(
            diff, libraries=["GO_BP"], background="phosphoproteome", direction="both"
        )
        # single call whose gene_list has 15 genes (up + down)
        assert len(mock_gseapy) == 1
        assert len(mock_gseapy[0]["gene_list"]) == 15

    def test_invalid_direction_raises(self):
        diff = _make_diff_exp()
        with pytest.raises(ValueError, match="direction must be"):
            pathway_enrichment(diff, direction="sideways")


class TestBackgroundResolution:
    def test_phosphoproteome_default(self, mock_gseapy):
        diff = _make_diff_exp(n_up=10, n_down=5, n_null=20)
        pathway_enrichment(diff, libraries=["GO_BP"])
        # background = every parseable gene across the whole diff_exp
        bg = mock_gseapy[0]["background"]
        assert bg is not None
        assert len(bg) == 35  # n_up + n_down + n_null

    def test_custom_list(self, mock_gseapy):
        diff = _make_diff_exp()
        custom = ["GENEX", "GENEY", "GENEZ"]
        pathway_enrichment(diff, libraries=["GO_BP"], background=custom)
        assert set(mock_gseapy[0]["background"]) == set(custom)

    def test_custom_dataframe_single_column(self, mock_gseapy):
        diff = _make_diff_exp()
        bg_df = pd.DataFrame({"my_genes": ["A", "B", "C"]})
        pathway_enrichment(diff, libraries=["GO_BP"], background=bg_df)
        assert set(mock_gseapy[0]["background"]) == {"A", "B", "C"}

    def test_custom_dataframe_named_gene_column(self, mock_gseapy):
        diff = _make_diff_exp()
        bg_df = pd.DataFrame({"gene": ["A", "B", "C"], "protein": ["P1", "P2", "P3"]})
        pathway_enrichment(diff, libraries=["GO_BP"], background=bg_df)
        assert set(mock_gseapy[0]["background"]) == {"A", "B", "C"}

    def test_genome_emits_warning(self, mock_gseapy):
        diff = _make_diff_exp()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pathway_enrichment(diff, libraries=["GO_BP"], background="genome")
        assert any(issubclass(w.category, UserWarning) for w in caught)
        assert mock_gseapy[0]["background"] is None

    def test_invalid_background_string_raises(self):
        diff = _make_diff_exp()
        with pytest.raises(ValueError, match="background str must be"):
            pathway_enrichment(diff, libraries=["GO_BP"], background="whole_universe")

    def test_invalid_background_type_raises(self):
        diff = _make_diff_exp()
        with pytest.raises(TypeError, match="background must be"):
            pathway_enrichment(diff, libraries=["GO_BP"], background=42)

    def test_empty_background_list_raises(self):
        diff = _make_diff_exp()
        with pytest.raises(ValueError, match="empty"):
            pathway_enrichment(diff, libraries=["GO_BP"], background=[])


class TestLibrariesOption:
    def test_default_libraries_used_when_none(self, mock_gseapy):
        diff = _make_diff_exp()
        pathway_enrichment(diff, background="phosphoproteome")
        assert mock_gseapy[0]["gene_sets"] == DEFAULT_LIBRARIES_HUMAN

    def test_mouse_uses_mouse_defaults(self, mock_gseapy):
        diff = _make_diff_exp()
        pathway_enrichment(diff, background="phosphoproteome", organism="mouse")
        assert mock_gseapy[0]["gene_sets"] == DEFAULT_LIBRARIES_MOUSE

    def test_custom_libraries_respected(self, mock_gseapy):
        diff = _make_diff_exp()
        pathway_enrichment(diff, libraries=["KEGG_2021_Human"], background="phosphoproteome")
        assert mock_gseapy[0]["gene_sets"] == ["KEGG_2021_Human"]

    def test_empty_libraries_raises(self):
        diff = _make_diff_exp()
        with pytest.raises(ValueError, match="libraries must be non-empty"):
            pathway_enrichment(diff, libraries=[], background="phosphoproteome")


class TestEmptyForeground:
    def test_no_significant_up_hits_returns_only_down_rows(self, mock_gseapy):
        # All positive log2fcs are non-significant
        diff = pd.DataFrame(
            {
                "log2fc": [2.0, 2.0, -2.0, -2.0, -2.0],
                "fdr": [0.9, 0.9, 0.001, 0.001, 0.001],
            },
            index=[
                "P1|UP1|S1|M1",
                "P2|UP2|S2|M1",
                "P3|DN1|S3|M1",
                "P4|DN2|S4|M1",
                "P5|DN3|S5|M1",
            ],
        )
        out = pathway_enrichment(diff, libraries=["GO_BP"], background="phosphoproteome")
        assert set(out["direction"].unique()) == {"down"}
        assert len(mock_gseapy) == 1

    def test_no_significant_hits_returns_empty(self, mock_gseapy):
        diff = pd.DataFrame(
            {"log2fc": [0.1, -0.1], "fdr": [0.9, 0.9]},
            index=["P1|G1|S1|M1", "P2|G2|S2|M1"],
        )
        out = pathway_enrichment(diff, libraries=["GO_BP"], background="phosphoproteome")
        assert out.empty
        assert len(mock_gseapy) == 0


class TestGeneExtraction:
    def test_skips_unparseable_keys(self, mock_gseapy):
        diff = pd.DataFrame(
            {"log2fc": [2.0, 2.0], "fdr": [0.001, 0.001]},
            index=["not_a_key", "P1|GOOD|S1|M1"],
        )
        pathway_enrichment(diff, libraries=["GO_BP"], background="phosphoproteome")
        assert mock_gseapy[0]["gene_list"] == ["GOOD"]

    def test_no_parseable_keys_raises(self):
        diff = pd.DataFrame(
            {"log2fc": [2.0], "fdr": [0.001]},
            index=["totally_bogus"],
        )
        with pytest.raises(ValueError, match="No parseable alphaPhos keys"):
            pathway_enrichment(diff, libraries=["GO_BP"], background="phosphoproteome")

    def test_key_column_supported(self, mock_gseapy):
        diff = pd.DataFrame(
            {
                "site_key": ["P1|GENE_X|S1|M1", "P2|GENE_Y|S2|M1"],
                "log2fc": [2.0, -2.0],
                "fdr": [0.001, 0.001],
            }
        )
        pathway_enrichment(
            diff, libraries=["GO_BP"], background="phosphoproteome", key_column="site_key"
        )
        # Two calls (up + down)
        assert len(mock_gseapy) == 2


# ---------------------------------------------------------------------------
# Integration on the real EGF walkthrough result (hits Enrichr's API)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not EGF_DIFF_EXP.exists(),
    reason="EGF walkthrough output missing; run examples/egf_walkthrough.py first",
)
class TestEGFRealData:
    """Real-data validation on the EGF +/- Spectronaut dataset.

    Requires network access to Enrichr; marked ``slow`` and will
    skip on connection errors.
    """

    @pytest.fixture(scope="class")
    def egf_result(self):
        return pd.read_csv(EGF_DIFF_EXP, sep="\t", index_col=0)

    @pytest.mark.slow
    def test_up_regulated_hits_egfr_pathway_terms(self, egf_result):
        try:
            out = pathway_enrichment(
                egf_result,
                libraries=["KEGG_2021_Human", "Reactome_2022", "MSigDB_Hallmark_2020"],
                background="phosphoproteome",
                direction="up",
                fdr_threshold=0.05,
            )
        except Exception as exc:
            pytest.skip(f"Enrichr unreachable: {exc}")
        assert not out.empty, "no enriched terms returned for up-regulated set"
        # Top hits should include something EGF- / MAPK- / RAS-flavoured
        top_terms = out.sort_values("fdr").head(30)["term"].str.lower().tolist()
        motifs = ["mapk", "egf", "ras", "erbb", "receptor tyrosine", "erk"]
        assert any(m in t for t in top_terms for m in motifs), (
            f"no MAPK/EGF/RAS pathway in top-30 up-regulated hits: {top_terms[:10]}"
        )

    @pytest.mark.slow
    def test_provenance_from_real_call(self, egf_result):
        try:
            out = pathway_enrichment(
                egf_result,
                libraries=["MSigDB_Hallmark_2020"],
                background="phosphoproteome",
                direction="up",
            )
        except Exception as exc:
            pytest.skip(f"Enrichr unreachable: {exc}")
        prov = out.attrs["provenance"]
        assert prov["background_type"] == "phosphoproteome"
        assert prov["direction"] == "up"
        assert prov["n_background_genes"] > 1000  # phospho universe is large
        assert prov["libraries"] == ["MSigDB_Hallmark_2020"]
