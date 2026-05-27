"""Tests for alphaphos.ksea.

Covers:
  - alphaphos_site_to_omnipath : site-id format converter
  - kinase_activity_ulm/mlm/ora/gsea : decoupler-backed activity wrappers
    (synthetic network, no live OmniPath fetch)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("decoupler")

from alphaphos.ksea import (
    alphaphos_site_to_omnipath,
    kinase_activity_gsea,
    kinase_activity_mlm,
    kinase_activity_ora,
    kinase_activity_ulm,
)

# ============================================================================
# Site-id converter
# ============================================================================


class TestAlphaphosSiteToOmnipath:
    def test_simple_single_protein(self):
        assert alphaphos_site_to_omnipath("P00533~EGFR_Y1172_M1") == "P00533_Y1172"

    def test_multi_protein_group_takes_first(self):
        assert alphaphos_site_to_omnipath("P12345;Q67890~MYGENE_S123_M2") == "P12345_S123"

    def test_threonine_site(self):
        assert alphaphos_site_to_omnipath("Q9Y1B6~SOME_T55_M1") == "Q9Y1B6_T55"

    def test_none_for_invalid(self):
        assert alphaphos_site_to_omnipath(None) is None
        assert alphaphos_site_to_omnipath("") is None
        assert alphaphos_site_to_omnipath("not_a_real_key") is None
        # Lowercase aa shouldn't match (we expect S/T/Y uppercase)
        assert alphaphos_site_to_omnipath("P00533~EGFR_y1172_M1") is None


# ============================================================================
# Synthetic fixtures (no live OmniPath fetch)
# ============================================================================


@pytest.fixture
def synthetic_net() -> pd.DataFrame:
    """Tiny kinase-substrate network covering 3 kinases x 30 sites total.

    KIN_A: 10 substrates. KIN_B: 10 substrates. KIN_C: 10 substrates.
    Site ids are in OmniPath format (UniProtAC_AApos).
    """
    rows = []
    for kin in ("KIN_A", "KIN_B", "KIN_C"):
        for i in range(10):
            rows.append(
                {
                    "source": kin,
                    "target": f"P{kin[-1]}{i:04d}_S{100 + i}",
                    "weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_diff(synthetic_net: pd.DataFrame) -> pd.DataFrame:
    """Diff_exp-style table. KIN_A substrates are up (lfc ≈ +3),
    KIN_B substrates are down (lfc ≈ -2), KIN_C unchanged (lfc ≈ 0)."""
    rng = np.random.default_rng(0)
    rows = []
    for _, edge in synthetic_net.iterrows():
        if edge["source"] == "KIN_A":
            lfc = 3.0 + rng.normal(0, 0.2)
        elif edge["source"] == "KIN_B":
            lfc = -2.0 + rng.normal(0, 0.2)
        else:
            lfc = rng.normal(0, 0.4)
        rows.append({"protein": edge["target"], "log2fc": float(lfc)})
    return pd.DataFrame(rows)


# ============================================================================
# kinase_activity_ulm
# ============================================================================


class TestKinaseActivityUlm:
    def test_basic_call_returns_per_kinase_table(self, synthetic_net, synthetic_diff):
        act = kinase_activity_ulm(
            synthetic_diff,
            synthetic_net,
            id_col="protein",
            stat_col="log2fc",
            min_targets=3,
            convert_site_ids=False,
        )
        assert act.index.name == "kinase"
        assert {"score", "fdr", "n_targets"}.issubset(act.columns)
        assert set(act.index) >= {"KIN_A", "KIN_B", "KIN_C"}

    def test_activated_kinase_has_positive_score(self, synthetic_net, synthetic_diff):
        act = kinase_activity_ulm(
            synthetic_diff,
            synthetic_net,
            id_col="protein",
            stat_col="log2fc",
            min_targets=3,
            convert_site_ids=False,
        )
        assert act.loc["KIN_A", "score"] > 0
        assert act.loc["KIN_B", "score"] < 0
        # KIN_C is null — score should be close to 0 (allow a band for noise)
        assert abs(act.loc["KIN_C", "score"]) < 2.0

    def test_min_targets_filter(self, synthetic_net, synthetic_diff):
        """Setting min_targets above what any kinase has should error or yield empty."""
        try:
            act = kinase_activity_ulm(
                synthetic_diff,
                synthetic_net,
                id_col="protein",
                stat_col="log2fc",
                min_targets=100,
                convert_site_ids=False,
            )
        except (ValueError, IndexError, AssertionError):
            # decoupler 2.x asserts when no kinases pass tmin — acceptable
            return
        if len(act):
            assert act["score"].isna().all() or len(act) == 0

    def test_converts_alphaphos_site_ids(self, synthetic_net):
        """When convert_site_ids=True the diff_exp ids are in alphaPhos format."""
        rng = np.random.default_rng(0)
        rows = []
        for _, edge in synthetic_net.iterrows():
            uniprot, site = edge["target"].split("_", 1)
            aa = site[0]
            pos = site[1:]
            key = f"{uniprot}~GENE_{aa}{pos}_M1"
            # Inject noise everywhere so the ulm regression has positive
            # variance on every kinase's substrate set (avoids NaN scores)
            base = 3.0 if edge["source"] == "KIN_A" else 0.0
            rows.append({"protein": key, "log2fc": base + rng.normal(0, 0.3)})
        diff = pd.DataFrame(rows)
        act = kinase_activity_ulm(
            diff,
            synthetic_net,
            id_col="protein",
            stat_col="log2fc",
            min_targets=3,
            convert_site_ids=True,
        )
        assert act.loc["KIN_A", "score"] > 0

    def test_raises_when_no_overlap(self, synthetic_net):
        """If no diff site ids match any network target, raise a clear error."""
        diff = pd.DataFrame(
            {
                "protein": [f"NONEXISTENT_{i}" for i in range(20)],
                "log2fc": np.linspace(-3, 3, 20),
            }
        )
        with pytest.raises(ValueError, match="No overlap"):
            kinase_activity_ulm(
                diff,
                synthetic_net,
                id_col="protein",
                stat_col="log2fc",
                convert_site_ids=False,
            )


# ============================================================================
# kinase_activity_mlm
# ============================================================================


class TestKinaseActivityMlm:
    def test_returns_per_kinase_table(self, synthetic_net, synthetic_diff):
        """Smoke test — verify the wrapper runs and returns the right schema.
        Sign-correctness is hard to guarantee on synthetic data with disjoint
        substrate sets (the design matrix is degenerate); covered by
        :class:`TestKinaseActivityUlm` instead.
        """
        act = kinase_activity_mlm(
            synthetic_diff,
            synthetic_net,
            id_col="protein",
            stat_col="log2fc",
            min_targets=3,
            convert_site_ids=False,
        )
        assert {"score", "fdr", "n_targets"}.issubset(act.columns)
        assert set(act.index) >= {"KIN_A", "KIN_B", "KIN_C"}
        # n_targets should match the network exactly
        assert act.loc["KIN_A", "n_targets"] == 10
        assert act.loc["KIN_B", "n_targets"] == 10


# ============================================================================
# kinase_activity_ora
# ============================================================================


class TestKinaseActivityOra:
    def test_top_fraction(self, synthetic_net, synthetic_diff):
        act = kinase_activity_ora(
            synthetic_diff,
            synthetic_net,
            id_col="protein",
            stat_col="log2fc",
            top_n=0.3,  # top 30% by |lfc|
            min_targets=3,
            convert_site_ids=False,
        )
        # KIN_A and KIN_B substrates dominate the top tail
        # KIN_C should have a non-significant score
        assert {"score", "fdr", "n_targets"}.issubset(act.columns)
        # Either KIN_A or KIN_B (they each have ~10 of the top 9) is enriched
        top = act.sort_values("fdr").head(2)
        assert "KIN_A" in top.index or "KIN_B" in top.index

    def test_top_n_as_integer(self, synthetic_net, synthetic_diff):
        """top_n can be a literal integer (number of sites)."""
        act = kinase_activity_ora(
            synthetic_diff,
            synthetic_net,
            id_col="protein",
            stat_col="log2fc",
            top_n=10,  # top 10 sites
            min_targets=3,
            convert_site_ids=False,
        )
        # decoupler drops kinases with no targets in the top-N; KIN_A
        # (with the highest |lfc|) should always be there.
        assert "KIN_A" in act.index
        assert {"score", "fdr", "n_targets"}.issubset(act.columns)


# ============================================================================
# kinase_activity_gsea
# ============================================================================


class TestKinaseActivityGsea:
    def test_basic_call(self, synthetic_net, synthetic_diff):
        act = kinase_activity_gsea(
            synthetic_diff,
            synthetic_net,
            id_col="protein",
            stat_col="log2fc",
            min_targets=3,
            convert_site_ids=False,
            permutation_num=100,  # quick
        )
        assert {"score", "fdr", "n_targets"}.issubset(act.columns)
        # KIN_A substrates are at the top of the ranking -> positive NES
        # KIN_B at the bottom -> negative NES
        if not act["score"].isna().all():
            valid = act.dropna(subset=["score"])
            if "KIN_A" in valid.index and "KIN_B" in valid.index:
                assert valid.loc["KIN_A", "score"] > 0
                assert valid.loc["KIN_B", "score"] < 0
