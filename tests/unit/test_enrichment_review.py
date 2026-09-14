"""Regression tests from the 2026-09-13 review of :mod:`alphaphos.enrichment`.

Everything here runs without gseapy / decoupler except the schema test for the
Enrichr normaliser, which feeds the real gseapy 1.3.1 background-mode frame
(no ``Overlap`` column) into the pure-pandas normaliser directly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alphaphos.enrichment import canonicalise_site_ids, gsea, library_redundancy, ora
from alphaphos.enrichment.enrich import _compute_es, _leading_edge
from alphaphos.enrichment.matching import parse_alphaphos_key
from alphaphos.enrichment.pathway.enrichment import _normalise_enrichr_output
from alphaphos.enrichment.pathway_gsea.gsea import _normalise_prerank_output
from alphaphos.enrichment.validation import _compute_directional_auc, score_against_ev3


def _universe(n=300):
    return [f"P{i:05d}_S{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# A1 -- gsea refuses duplicated site ids
# ---------------------------------------------------------------------------


class TestGseaDuplicateIds:
    def test_duplicates_raise_with_aggregation_hint(self):
        idx = _universe()
        idx[200:260] = idx[:60]
        s = pd.Series(np.random.default_rng(0).normal(size=300), index=idx)
        with pytest.raises(ValueError, match="duplicated site ids.*groupby"):
            gsea(s, {"L": {"A": idx[:40]}}, n_permutations=50)

    def test_aggregated_input_matches_clean_input(self):
        rng = np.random.default_rng(0)
        idx = _universe()
        vals = rng.normal(size=300)
        vals[:40] += 2
        clean = gsea(pd.Series(vals, index=idx), {"L": {"A": idx[:40]}}, n_permutations=200)
        dup = pd.Series(np.r_[vals, vals[:60] * 0.5], index=idx + idx[:60])
        agg = dup.groupby(level=0).agg(lambda v: v.iloc[int(v.abs().to_numpy().argmax())])
        res = gsea(agg, {"L": {"A": idx[:40]}}, n_permutations=200)
        assert res["ES"].iloc[0] == pytest.approx(clean["ES"].iloc[0])


# ---------------------------------------------------------------------------
# A3 -- ORA filters on set size, tests depletion, warns on inconsistent min_overlap
# ---------------------------------------------------------------------------


class TestOraFiltering:
    def test_zero_overlap_set_is_tested_as_depleted_by_default(self):
        bg = _universe()
        hits = bg[:60]
        far = bg[140:200]  # 60 members, 0 hits, expected 12
        out = ora(hits, bg, {"L": {"far": far, "near": bg[:30]}})
        row = out.set_index("set_name").loc["far"]
        assert row["direction"] == "depleted" and row["n_overlap"] == 0
        assert row["p_value"] == pytest.approx(stats.fisher_exact([[0, 60], [60, 180]])[1])
        assert out.attrs["n_sets_skipped_overlap"] == 0

    def test_min_set_size_is_hit_independent(self):
        bg = _universe()
        hits = bg[:60]
        tiny = bg[:3]  # 3 members, all hits -- still skipped (size, not overlap)
        out = ora(hits, bg, {"L": {"tiny": tiny, "ok": bg[:30]}}, min_set_size=5)
        assert set(out["set_name"]) == {"ok"}
        assert out.attrs["n_sets_skipped_size"] == 1
        out2 = ora(hits, bg, {"L": {"tiny": tiny}}, min_set_size=3)
        assert set(out2["set_name"]) == {"tiny"}

    def test_min_overlap_with_two_sided_warns(self):
        bg = _universe()
        hits = bg[:60]
        with pytest.warns(UserWarning, match="most depleted"):
            out = ora(hits, bg, {"L": {"far": bg[140:200], "near": bg[:30]}}, min_overlap=2)
        assert set(out["set_name"]) == {"near"}
        assert out.attrs["n_sets_skipped_overlap"] == 1

    def test_min_overlap_one_sided_does_not_warn(self):
        import warnings

        bg = _universe()
        hits = bg[:60]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ora(hits, bg, {"L": {"near": bg[:30]}}, min_overlap=2, alternative="greater")


# ---------------------------------------------------------------------------
# A4 / C1 -- leading edge respects weight; max_set_size default and skip report
# ---------------------------------------------------------------------------


class TestGseaInternals:
    def test_leading_edge_uses_weight(self):
        rng = np.random.default_rng(1)
        ranks = np.sort(rng.normal(0, 3, 400))[::-1]
        ranks[:5] = 50.0  # a few huge values make weight matter
        idx = np.array(sorted([*rng.choice(400, 40, replace=False).tolist(), 0, 1, 2, 3, 4]))
        idx = np.unique(idx)
        es0, _ = _compute_es(ranks, idx, weight=0.0)
        es1, _ = _compute_es(ranks, idx, weight=1.0)
        le0 = _leading_edge(ranks, idx, es0, weight=0.0)
        le1 = _leading_edge(ranks, idx, es1, weight=1.0)
        assert le0 != le1  # would be identical if weight were ignored

    def test_no_default_size_cap_and_skips_reported(self, caplog):
        idx = _universe(1000)
        s = pd.Series(np.random.default_rng(0).normal(size=1000), index=idx)
        libs = {"L": {"huge": idx[:700], "small": idx[:3], "ok": idx[100:200]}}
        with caplog.at_level(logging.INFO, logger="alphaphos.enrichment.enrich"):
            out = gsea(s, libs, n_permutations=50)
        assert set(out["set_name"]) == {"huge", "ok"}  # 700 members no longer skipped
        assert "L/small" in out.attrs["skipped_sets"]
        capped = gsea(s, libs, n_permutations=50, max_set_size=500)
        assert set(capped["set_name"]) == {"ok"}
        assert "L/huge" in capped.attrs["skipped_sets"]


# ---------------------------------------------------------------------------
# C2 -- key regex accepts empty gene; dropped keys are logged
# ---------------------------------------------------------------------------


class TestCanonicalise:
    def test_empty_gene_key_parses(self):
        parsed = parse_alphaphos_key("P12345||S10|M1")
        assert parsed is not None and parsed.gene == "" and parsed.canonical_site_id == "P12345_S10"

    def test_dropped_keys_are_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="alphaphos.enrichment.matching"):
            out = canonicalise_site_ids(["P1|G|S1|M1", "garbage", "P2|G|Y2|M2"])
        assert out == ["P1_S1", "P2_Y2"]
        assert any("dropped 1 unparseable" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# A2 / C3 -- gseapy output schemas
# ---------------------------------------------------------------------------


class TestGseapySchemas:
    def test_enrichr_background_mode_without_overlap_column(self):
        # Exact column set returned by gseapy 1.3.1 enrichr(background=[...]).
        raw = pd.DataFrame(
            {
                "Gene_set": ["KEGG", "KEGG"],
                "Term": ["ErbB signaling pathway", "Other"],
                "P-value": [1e-5, 0.2],
                "Adjusted P-value": [1e-4, 0.3],
                "Old P-value": [0, 0],
                "Old adjusted P-value": [0, 0],
                "Odds Ratio": [3.0, 1.1],
                "Combined Score": [20.0, 1.0],
                "Genes": ["EGFR;MAPK1;MAPK3", "AKT1"],
            }
        )
        out = _normalise_enrichr_output(
            raw,
            direction="up",
            n_foreground=10,
            n_background=100,
            term_sizes={"KEGG": {"ErbB signaling pathway": 85}},
        )
        assert list(out["n_overlap"]) == [3, 1]
        assert out["n_term"].tolist()[0] == 85 and np.isnan(out["n_term"].iloc[1])
        assert list(out["overlap"]) == ["3/85", "1/?"]
        assert {"direction", "library", "term", "overlap", "p_value", "fdr", "genes"} <= set(
            out.columns
        )

    def test_enrichr_web_mode_overlap_column_is_parsed(self):
        raw = pd.DataFrame(
            {
                "Gene_set": ["GO"],
                "Term": ["t"],
                "Overlap": ["4/120"],
                "P-value": [0.01],
                "Adjusted P-value": [0.05],
                "Odds Ratio": [2.0],
                "Combined Score": [9.0],
                "Genes": ["A;B;C;D"],
            }
        )
        out = _normalise_enrichr_output(raw, direction="down", n_foreground=5, n_background=0)
        assert (
            out["n_overlap"].iloc[0] == 4
            and out["n_term"].iloc[0] == 120
            and out["overlap"].iloc[0] == "4/120"
        )

    def test_enrichr_schema_drift_raises(self):
        raw = pd.DataFrame({"Gene_set": ["x"], "Term": ["t"], "P-value": [0.1]})
        with pytest.raises(ValueError, match="missing columns"):
            _normalise_enrichr_output(raw, direction="up", n_foreground=1, n_background=1)

    def test_prerank_tag_split(self):
        raw = pd.DataFrame(
            {
                "Name": ["p"],
                "Term": ["t"],
                "ES": ["0.5"],
                "NES": ["1.4"],
                "NOM p-val": ["0.01"],
                "FDR q-val": ["0.1"],
                "FWER p-val": ["0.2"],
                "Tag %": ["8/20"],
                "Gene %": ["10%"],
                "Lead_genes": ["A;B"],
            }
        )
        out = _normalise_prerank_output(raw, library="L")
        assert out["n_leading_edge"].iloc[0] == 8 and out["n_set"].iloc[0] == 20
        assert "size" not in out.columns


# ---------------------------------------------------------------------------
# C7 / C8 -- validation AUC ties, duplicate kinase index, redundancy helper
# ---------------------------------------------------------------------------


class TestValidationHelpers:
    def test_auc_counts_ties_as_half(self):
        pk = pd.DataFrame(
            {
                "observed_score": [1.0, 1.0, 1.0, 1.0],
                "expected_direction": ["up", "up", "down", "down"],
            }
        )
        assert _compute_directional_auc(pk) == pytest.approx(0.5)

    def test_duplicate_kinase_index_raises(self):
        ev3 = pd.DataFrame(
            {
                "Condition": ["EGF", "EGF"],
                "Kinase": ["MAPK1", "AKT1"],
                "Directionality": ["up", "up"],
            }
        )
        scores = pd.Series([1.0, 2.0, 3.0], index=["MAPK1", "MAPK1", "AKT1"])
        with pytest.raises(ValueError, match="duplicated kinases"):
            score_against_ev3(scores, "EGF", ev3=ev3)

    def test_library_redundancy(self):
        libs = {"L": {"A": ["s1", "s2", "s3", "s4"], "B": ["s3", "s4", "s5"], "C": ["s9"]}}
        red = library_redundancy(libs)
        top = red.iloc[0]
        assert (top["set_a"], top["set_b"]) == ("A", "B") and top["jaccard"] == pytest.approx(2 / 5)
        assert (red["jaccard"].diff().dropna() <= 0).all()
        restricted = library_redundancy(libs, background=["s3", "s4"], min_jaccard=0.5)
        assert restricted["jaccard"].iloc[0] == pytest.approx(1.0)
