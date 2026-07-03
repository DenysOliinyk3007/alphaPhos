"""Validation tests for :mod:`alphaphos.enrichment.validation`.

Two categories:

1. **Unit tests** on the loaders + kinase-substrate library builder
   using the small PTM DB fixture.
2. **Synthetic-AUC engine validation**: construct N synthetic
   "conditions" with known ground truth (a specific kinase's substrates
   spiked with a big positive / negative log2fc, all other sites
   random), run GSEA, and confirm the engine assigns the correct sign
   to the target kinase across the panel.  Reports an ROC AUC on the
   engine's kinase-direction calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaphos.enrichment import (
    TEST_FIXTURE_PATH,
    build_kinase_substrate_library,
    ev3_expectations_for_condition,
    gsea,
    load_ev3,
    score_against_ev3,
)

pytestmark = pytest.mark.skipif(
    not TEST_FIXTURE_PATH.exists(),
    reason=f"mini PTM DB fixture missing at {TEST_FIXTURE_PATH}",
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestBuildKinaseSubstrateLibrary:
    def test_returns_dict_of_lists(self):
        lib = build_kinase_substrate_library(
            db_path=TEST_FIXTURE_PATH, min_set_size=2, require_curated=False
        )
        assert isinstance(lib, dict)
        for kinase, members in lib.items():
            assert isinstance(kinase, str)
            assert isinstance(members, list)
            assert len(members) >= 2

    def test_min_set_size_enforced(self):
        big = build_kinase_substrate_library(
            db_path=TEST_FIXTURE_PATH, min_set_size=1, require_curated=False
        )
        small = build_kinase_substrate_library(
            db_path=TEST_FIXTURE_PATH, min_set_size=5, require_curated=False
        )
        assert len(small) <= len(big)
        for members in small.values():
            assert len(members) >= 5

    def test_curated_only_shrinks_or_equals(self):
        all_ = build_kinase_substrate_library(
            db_path=TEST_FIXTURE_PATH, min_set_size=1, require_curated=False
        )
        cur = build_kinase_substrate_library(
            db_path=TEST_FIXTURE_PATH, min_set_size=1, require_curated=True
        )
        for kinase, members in cur.items():
            if kinase in all_:
                assert len(members) <= len(all_[kinase])


class TestLoadEV3:
    def test_shape_and_columns(self):
        ev3 = load_ev3()
        assert not ev3.empty
        for col in ["Condition ID", "Condition", "Kinase", "Directionality"]:
            assert col in ev3.columns
        assert set(ev3["Directionality"]).issubset({"up", "down"})

    def test_egf_condition_filter(self):
        ev3 = load_ev3()
        egf_exact = ev3[ev3["Condition"] == "EGF"]
        assert len(egf_exact) >= 1
        assert (egf_exact["Kinase"] == "EGFR").all()
        assert (egf_exact["Directionality"] == "up").all()

    def test_substring_filter_helper(self):
        ev3 = load_ev3()
        # 'EGF' substring pulls in VEGF, EGFRi, EGF+U0126 — that's expected.
        subs = ev3_expectations_for_condition("EGF", ev3=ev3)
        exact = ev3[ev3["Condition"] == "EGF"]
        assert len(subs) > len(exact)


class TestScoreAgainstEV3:
    def test_perfect_score_on_known_upregulation(self):
        # Fake kinase-activity vector: EGFR gets +2.0 NES.
        kinase_scores = pd.Series({"EGFR": 2.0, "OTHER_KINASE": 0.1})
        report = score_against_ev3(kinase_scores, "EGF", exact_condition="EGF")
        assert report["n_expected"] == 1
        assert report["n_recovered"] == 1
        assert report["recall_direction"] == 1.0

    def test_wrong_direction_scored_as_miss(self):
        kinase_scores = pd.Series({"EGFR": -2.0})  # wrong direction
        report = score_against_ev3(kinase_scores, "EGF", exact_condition="EGF")
        assert report["n_recovered"] == 0

    def test_missing_kinase_scored_as_miss(self):
        # Kinase not in our scores at all
        report = score_against_ev3(pd.Series(dtype=float), "EGF", exact_condition="EGF")
        assert report["n_recovered"] == 0

    def test_auc_none_with_single_direction(self):
        # EGF condition has only "up" expectations -> AUC undefined
        kinase_scores = pd.Series({"EGFR": 2.0})
        report = score_against_ev3(kinase_scores, "EGF", exact_condition="EGF")
        assert report["auc"] is None


# ---------------------------------------------------------------------------
# Synthetic-AUC engine validation
# ---------------------------------------------------------------------------


class TestSyntheticAUC:
    """Build N synthetic conditions with known ground truth; confirm
    GSEA assigns the correct sign to each condition's target kinase."""

    @pytest.fixture(scope="class")
    def kinase_lib(self):
        # Use mini fixture -> smaller library but plenty of kinases with 5+ subs.
        lib = build_kinase_substrate_library(
            db_path=TEST_FIXTURE_PATH,
            min_set_size=5,
            require_curated=False,
        )
        assert len(lib) >= 3, "test fixture doesn't have enough kinases with 5+ substrates"
        return lib

    def test_engine_recovers_direction_across_conditions(self, kinase_lib):
        # Build the universe: union of all sites in the library plus random padding.
        rng = np.random.default_rng(42)
        universe = set()
        for members in kinase_lib.values():
            universe.update(members)
        # Pad with 500 non-set sites so background is meaningful.
        padding = [f"NONSET_{i}" for i in range(500)]
        universe.update(padding)
        universe = sorted(universe)

        # For each kinase with >=5 substrates, generate 2 conditions:
        # one where its substrates are up-spiked (+3 log2fc),
        # one where they are down-spiked (-3 log2fc).
        # All other sites: N(0, 1).
        results: list[dict] = []
        target_kinases = list(kinase_lib.keys())[:10]  # cap for test speed

        for kinase in target_kinases:
            substrates = set(kinase_lib[kinase])
            for direction, spike in [("up", 3.0), ("down", -3.0)]:
                ranks = pd.Series(
                    rng.normal(0, 1, len(universe)),
                    index=universe,
                )
                ranks[list(substrates)] = spike
                gsea_out = gsea(
                    ranks,
                    {"kinase_substrate": kinase_lib},
                    min_set_size=5,
                    max_set_size=100_000,
                    n_permutations=200,
                    seed=1,
                )
                nes = gsea_out.set_index("set_name")["NES"].get(kinase, np.nan)
                results.append(
                    {
                        "kinase": kinase,
                        "expected_direction": direction,
                        "observed_nes": nes,
                        "correct_direction": (
                            (nes > 0 and direction == "up") or (nes < 0 and direction == "down")
                        ),
                    }
                )

        results_df = pd.DataFrame(results)
        # ROC AUC: perfect engine should give AUC = 1.0.
        labels = (results_df["expected_direction"] == "up").astype(int).to_numpy()
        scores = results_df["observed_nes"].to_numpy()
        valid = ~np.isnan(scores)
        scores = scores[valid]
        labels = labels[valid]
        order = np.argsort(scores)
        ranks = np.empty_like(scores, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

        n_correct = results_df["correct_direction"].sum()
        n_total = len(results_df)
        print(
            f"\nSynthetic-AUC engine validation: AUC={auc:.3f}  correct_dir={n_correct}/{n_total}"
        )
        # Beltrao paper reports mean AUC ~0.72 for the best inference
        # methods on the real 132-pair benchmark.  Our SYNTHETIC AUC
        # tests only the engine (with perfect ground truth), so it
        # should be much higher.  0.9 is a generous floor.
        assert auc >= 0.9, f"engine AUC {auc:.3f} < 0.9 on synthetic conditions"
        assert n_correct >= 0.8 * n_total, (
            f"only {n_correct}/{n_total} conditions got correct direction"
        )
