"""Tests for alphaphos.enrichment.ksea.

Combines fast unit tests on synthetic inputs with a couple of
integration-style checks on the real EGF walkthrough result (skipped if
the walkthrough hasn't been run locally).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("decoupler")

from alphaphos.enrichment import (
    TEST_FIXTURE_PATH,
    kinase_activity,
    load_ptm_ks_network,
)
from alphaphos.enrichment.ksea.network import validate_network

REPO = Path(__file__).resolve().parents[2]
EGF_DIFF_EXP = REPO / "test_data" / "walkthrough_output" / "egf_diff_exp_result.tsv"
OMNIPATH_CACHE = REPO / "test_data" / "walkthrough_output" / "omnipath_ks_human.parquet"


# ---------------------------------------------------------------------------
# Network validation
# ---------------------------------------------------------------------------


class TestValidateNetwork:
    def test_accepts_minimal_schema(self):
        net = pd.DataFrame({"source": ["K1", "K1", "K2"], "target": ["S1", "S2", "S3"]})
        out = validate_network(net)
        assert "weight" in out.columns
        assert (out["weight"] == 1.0).all()

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            validate_network(pd.DataFrame({"kinase": ["K1"], "site": ["S1"]}))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_network(pd.DataFrame({"source": [], "target": [], "weight": []}))

    def test_non_dataframe_raises(self):
        with pytest.raises(ValueError, match="DataFrame"):
            validate_network({"source": ["K1"], "target": ["S1"]})

    def test_preserves_extra_columns(self):
        net = pd.DataFrame(
            {
                "source": ["K1"],
                "target": ["S1"],
                "confidence": ["high"],
                "n_sources": [3],
            }
        )
        out = validate_network(net)
        assert "confidence" in out.columns
        assert "n_sources" in out.columns


# ---------------------------------------------------------------------------
# PTM-DB network loader (needs mini fixture)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not TEST_FIXTURE_PATH.exists(),
    reason=f"PTM DB fixture missing at {TEST_FIXTURE_PATH}",
)
class TestLoadPtmKsNetwork:
    def test_returns_decoupler_schema(self):
        net = load_ptm_ks_network(db_path=TEST_FIXTURE_PATH, require_curated=False)
        assert not net.empty
        for col in ("source", "target", "weight"):
            assert col in net.columns
        assert net["weight"].dtype == float

    def test_deduplicated_on_source_target(self):
        net = load_ptm_ks_network(db_path=TEST_FIXTURE_PATH, require_curated=False)
        dup = net.duplicated(subset=["source", "target"]).sum()
        assert dup == 0

    def test_min_curation_confidence_filter(self):
        # 'high' should be strictly a subset of 'low'.
        net_low = load_ptm_ks_network(
            db_path=TEST_FIXTURE_PATH,
            require_curated=False,
            min_curation_confidence="low",
        )
        net_hi = load_ptm_ks_network(
            db_path=TEST_FIXTURE_PATH,
            require_curated=False,
            min_curation_confidence="high",
        )
        assert len(net_hi) <= len(net_low)


# ---------------------------------------------------------------------------
# kinase_activity — synthetic
# ---------------------------------------------------------------------------


class TestKinaseActivitySynthetic:
    """Hand-crafted BYO network + diff-exp with known expected direction."""

    def _make_synthetic_result(self, kinase_up: str, kinase_down: str, seed: int = 0):
        rng = np.random.default_rng(seed)
        substrates_up = [f"UP_S{i}" for i in range(30)]  # substrates of `kinase_up`
        substrates_down = [f"DN_S{i}" for i in range(30)]
        random_sites = [f"RND_S{i}" for i in range(200)]
        all_sites = substrates_up + substrates_down + random_sites
        # log2fc: +3 for kinase_up substrates, -3 for kinase_down, noise otherwise
        log2fc = (
            [3.0] * len(substrates_up)
            + [-3.0] * len(substrates_down)
            + list(rng.normal(0, 0.5, len(random_sites)))
        )
        diff_exp = pd.DataFrame({"log2fc": log2fc}, index=all_sites)
        net = pd.DataFrame(
            {
                "source": [kinase_up] * len(substrates_up)
                + [kinase_down] * len(substrates_down)
                + ["NOISE_K"] * 10,
                "target": substrates_up + substrates_down + random_sites[:10],
                "weight": 1.0,
            }
        )
        return diff_exp, net

    def test_ulm_recovers_expected_direction(self):
        diff_exp, net = self._make_synthetic_result("KIN_UP", "KIN_DOWN")
        result = kinase_activity(
            diff_exp,
            stat_col="log2fc",
            network=net,
            method="ulm",
            min_substrates=5,
        )
        row_up = result[result["kinase"] == "KIN_UP"].iloc[0]
        row_dn = result[result["kinase"] == "KIN_DOWN"].iloc[0]
        assert row_up["score"] > 0 and row_up["direction"] == "up"
        assert row_dn["score"] < 0 and row_dn["direction"] == "down"
        assert row_up["fdr"] < 0.05
        assert row_dn["fdr"] < 0.05

    def test_mlm_also_recovers_direction(self):
        diff_exp, net = self._make_synthetic_result("KIN_UP2", "KIN_DOWN2")
        result = kinase_activity(
            diff_exp,
            stat_col="log2fc",
            network=net,
            method="mlm",
            min_substrates=5,
        )
        assert (result["kinase"] == "KIN_UP2").any()
        assert (result["kinase"] == "KIN_DOWN2").any()
        row_up = result[result["kinase"] == "KIN_UP2"].iloc[0]
        row_dn = result[result["kinase"] == "KIN_DOWN2"].iloc[0]
        assert row_up["score"] > 0
        assert row_dn["score"] < 0

    def test_min_substrates_filter(self):
        diff_exp, net = self._make_synthetic_result("KIN_UP3", "KIN_DOWN3")
        # NOISE_K has only 10 substrates in the network; asking for 20 min drops it
        r_low = kinase_activity(
            diff_exp, stat_col="log2fc", network=net, method="ulm", min_substrates=5
        )
        r_hi = kinase_activity(
            diff_exp, stat_col="log2fc", network=net, method="ulm", min_substrates=20
        )
        assert "NOISE_K" in r_low["kinase"].values
        assert "NOISE_K" not in r_hi["kinase"].values

    def test_bad_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            kinase_activity(
                pd.DataFrame({"log2fc": [1.0]}, index=["S1"]),
                network=pd.DataFrame({"source": ["K"], "target": ["S1"]}),
                method="viper",
            )

    def test_bad_network_string_raises(self):
        with pytest.raises(ValueError, match="network must be"):
            kinase_activity(
                pd.DataFrame({"log2fc": [1.0]}, index=["S1"]),
                network="phosphositeplus",
            )

    def test_bad_fdr_method_raises(self):
        with pytest.raises(ValueError, match="fdr_method must be"):
            kinase_activity(
                pd.DataFrame({"log2fc": [1.0]}, index=["S1"]),
                network=pd.DataFrame({"source": ["K"], "target": ["S1"]}),
                method="ulm",
                fdr_method="by",
            )

    def test_anova_shape_raises_helpful_error(self):
        # diff_exp_anova output: 'F' + 'fdr', no 'log2fc' -> signed
        # KSEA is undefined, so we want a clear pointer to the fix.
        anova = pd.DataFrame(
            {"F": [15.0, 2.0], "fdr": [0.001, 0.5]},
            index=["S1", "S2"],
        )
        with pytest.raises(ValueError, match="ANOVA|diff_exp_anova"):
            kinase_activity(
                anova,
                network=pd.DataFrame({"source": ["K"], "target": ["S1"]}),
                method="ulm",
            )

    def test_missing_stat_col_without_F_raises_plain_error(self):
        df = pd.DataFrame({"log2fc": [1.0]}, index=["S1"])
        with pytest.raises(ValueError, match="not in diff_exp_result columns"):
            kinase_activity(
                df,
                network=pd.DataFrame({"source": ["K"], "target": ["S1"]}),
                method="ulm",
                stat_col="nonexistent",
            )

    def test_provenance_stamped(self):
        diff_exp, net = self._make_synthetic_result("KIN_UP4", "KIN_DOWN4")
        result = kinase_activity(diff_exp, network=net, method="ulm", min_substrates=5)
        prov = result.attrs["provenance"]
        assert prov["method"] == "ulm"
        assert prov["network"] == "user_supplied"
        assert prov["n_kinases_tested"] == len(result)
        assert "decoupler_version" in prov


# ---------------------------------------------------------------------------
# kinase_activity — alphaphos key handling
# ---------------------------------------------------------------------------


class TestAlphaphosKeyAutoConversion:
    def test_auto_canonicalises_alphaphos_keys(self):
        # Diff exp indexed by alphaphos keys (Protein|Gene|Site|Mult).
        # kinase_activity should convert them to Protein_AApos before
        # matching the network.
        diff_exp = pd.DataFrame(
            {"log2fc": [3.0, 3.0, 3.0, 3.0, 3.0, -0.1, -0.2, 0.05, -0.03, 0.1]},
            index=[
                "P00001|GENE1|S100|M1",
                "P00001|GENE1|S110|M1",
                "P00001|GENE1|S120|M1",
                "P00001|GENE1|S130|M1",
                "P00001|GENE1|S140|M1",
                "P00002|GENE2|S200|M1",
                "P00002|GENE2|S210|M1",
                "P00002|GENE2|S220|M1",
                "P00003|GENE3|S300|M1",
                "P00003|GENE3|S310|M1",
            ],
        )
        # Network: MY_KIN targets sites via Protein_AApos IDs
        net = pd.DataFrame(
            {
                "source": ["MY_KIN"] * 5,
                "target": [f"P00001_S{i}" for i in (100, 110, 120, 130, 140)],
                "weight": 1.0,
            }
        )
        result = kinase_activity(diff_exp, network=net, method="ulm", min_substrates=5)
        # If canonicalisation worked, MY_KIN should surface with a strong positive score
        assert (result["kinase"] == "MY_KIN").any()
        row = result[result["kinase"] == "MY_KIN"].iloc[0]
        assert row["score"] > 0
        assert row["direction"] == "up"


# ---------------------------------------------------------------------------
# Integration on the real EGF walkthrough result
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not EGF_DIFF_EXP.exists() or not OMNIPATH_CACHE.exists(),
    reason="EGF walkthrough outputs missing; run examples/egf_walkthrough.py first",
)
class TestEGFRealData:
    """Real-data validation on the EGF +/- Spectronaut dataset."""

    @pytest.fixture(scope="class")
    def egf_result(self):
        return pd.read_csv(EGF_DIFF_EXP, sep="\t", index_col=0)

    def test_ulm_omnipath_finds_canonical_egf_kinases(self, egf_result):
        ksea = kinase_activity(
            egf_result,
            stat_col="log2fc",
            network="omnipath",
            method="ulm",
            min_substrates=5,
            cache_path=OMNIPATH_CACHE,
        )
        # Expected canonical EGF-signaling kinases in the top-20
        expected = {"EGF", "BRAF", "MAP2K3", "MAPKAPK2", "MAP3K8", "RAF1"}
        top20 = set(ksea.sort_values("score", ascending=False).head(20)["kinase"])
        matched = top20 & expected
        assert len(matched) >= 4, f"only {len(matched)} of {expected} in top-20: {matched}"

    def test_gsk3b_down_regulated(self, egf_result):
        # GSK3B is inhibited by AKT downstream of EGF — expected in the bottom.
        ksea = kinase_activity(
            egf_result,
            stat_col="log2fc",
            network="omnipath",
            method="ulm",
            min_substrates=5,
            cache_path=OMNIPATH_CACHE,
        )
        assert "GSK3B" in ksea["kinase"].values, "GSK3B missing from OmniPath scoring"
        gsk3b = ksea[ksea["kinase"] == "GSK3B"].iloc[0]
        assert gsk3b["score"] < 0, f"GSK3B expected down; got score={gsk3b['score']}"
        assert gsk3b["direction"] == "down"

    def test_ptm_db_network_produces_similar_top20(self, egf_result):
        ksea_omni = kinase_activity(
            egf_result,
            method="ulm",
            network="omnipath",
            min_substrates=5,
            cache_path=OMNIPATH_CACHE,
        )
        ksea_ptm = kinase_activity(egf_result, method="ulm", network="ptm_db", min_substrates=5)
        top20_o = set(ksea_omni.sort_values("score", ascending=False).head(20)["kinase"])
        top20_p = set(ksea_ptm.sort_values("score", ascending=False).head(20)["kinase"])
        overlap = top20_o & top20_p
        # Two different networks should agree on at least 10/20 top kinases on
        # a strong signal like EGF stimulation.
        assert len(overlap) >= 10, (
            f"top-20 overlap between OmniPath and PTM-DB is only {len(overlap)}"
        )

    def test_mlm_error_message_when_singular(self, egf_result):
        # OmniPath at default filtering is dense enough that MLM's design
        # matrix is singular. The error message must point users at ULM.
        with pytest.raises(RuntimeError, match="ulm"):
            kinase_activity(
                egf_result,
                method="mlm",
                network="omnipath",
                min_substrates=5,
                cache_path=OMNIPATH_CACHE,
            )

    def test_mlm_succeeds_on_high_confidence_ptm_db(self, egf_result):
        # A smaller / higher-confidence network is what MLM wants.
        small_net = load_ptm_ks_network(require_curated=True, min_curation_confidence="high")
        ksea = kinase_activity(egf_result, method="mlm", network=small_net, min_substrates=5)
        assert not ksea.empty
        top10 = set(ksea.sort_values("score", ascending=False).head(10)["kinase"])
        expected = {"MAPKAPK2", "EGFR", "MAPK1", "MAP2K1", "BRAF"}
        matched = top10 & expected
        assert len(matched) >= 3, f"MLM top-10 missed too many EGF-canonical kinases: {matched}"
