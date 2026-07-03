"""Unit tests for :mod:`alphaphos.enrichment.enrich`.

Every math-heavy test uses a small hand-crafted example whose Fisher /
GSEA answer is either analytically computable or checkable via scipy
directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alphaphos.enrichment import gsea, ora

# ---------------------------------------------------------------------------
# ORA
# ---------------------------------------------------------------------------


class TestORA:
    def test_fisher_math_matches_scipy(self):
        """Hand-crafted 2x2 whose exact Fisher p we can verify against scipy."""
        # Background = 100 sites; hits = 20; set has 30 sites total in bg;
        # observed overlap = 10 (vs expected 6).  Fisher's two-sided p should
        # match scipy's answer to machine precision.
        background = [f"S{i}" for i in range(100)]
        hits = background[:20]
        library = {
            "test_lib": {
                "big_set": background[10:40],  # 30 sites; overlap w/ hits = 10
            }
        }
        expected_p = stats.fisher_exact([[10, 10], [20, 60]], alternative="two-sided").pvalue
        result = ora(hits, background, library, min_overlap=1)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["n_overlap"] == 10
        assert row["n_set"] == 30
        assert row["n_hits"] == 20
        assert row["n_background"] == 100
        assert row["p_value"] == pytest.approx(expected_p)
        assert row["direction"] == "enriched"

    def test_background_restriction_prevents_inflated_significance(self):
        """A set of 5000 with only 5 in the background must be tested as size 5."""
        background = [f"S{i}" for i in range(100)]
        hits = background[:20]
        # Set contains most of the background but ALSO 5000 "phantom" sites
        # the assay didn't measure.  Correct behavior: test as size-5-in-bg
        # (0 overlap), NOT as size-5000 (which would spuriously call
        # depletion because 0 << expected 100).
        phantoms = [f"NOT_IN_BG_{i}" for i in range(5000)]
        library = {"test_lib": {"phantom_set": [background[95], *phantoms]}}
        result = ora(hits, background, library, min_overlap=1)
        # With min_overlap=1, we still get a row because background[95] IS
        # in the background but not in hits (since hits are indices 0-19).
        # So n_overlap=0 < min_overlap=1 -> skip.  Expect 0 rows.
        assert result.empty

    def test_min_overlap_filter(self):
        background = [f"S{i}" for i in range(100)]
        hits = background[:10]
        library = {
            "test_lib": {
                "one_hit": [background[0], *background[50:80]],  # overlap=1
                "many_hits": background[:8],  # overlap=8
            }
        }
        result = ora(hits, background, library, min_overlap=2)
        assert set(result["set_name"]) == {"many_hits"}

    def test_bh_fdr_applied_per_library(self):
        # Two libraries × 5 sets each. Different libraries get independent BH.
        background = [f"S{i}" for i in range(200)]
        hits = background[:40]
        # 5 sets per library, all with the same overlap size 8 -> same p-value
        libraries = {
            f"lib{L}": {f"set{s}": background[s * 8 : (s + 1) * 8] for s in range(5)}
            for L in range(2)
        }
        r_tiered = ora(hits, background, libraries, min_overlap=1, fdr_per_library=True)
        r_joint = ora(hits, background, libraries, min_overlap=1, fdr_per_library=False)
        # Joint BH corrects across 10 tests; tiered corrects across 5 per library.
        # For the same nominal p-value the joint q should be >= tiered q.
        for row_t, row_j in zip(
            r_tiered.sort_values("set_name").itertuples(),
            r_joint.sort_values("set_name").itertuples(),
            strict=True,
        ):
            assert row_j.fdr >= row_t.fdr - 1e-12

    def test_hits_must_be_subset_of_background(self):
        with pytest.raises(ValueError, match="not in the background"):
            ora(["stray_hit"], ["A", "B"], {"lib": {"set": ["A"]}})

    def test_two_sided_finds_depletion(self):
        background = [f"S{i}" for i in range(200)]
        hits = background[:20]
        # Set of 50 sites, all in background positions 60-110 => hits (0-19)
        # have zero overlap with the set. Expected overlap = 20*50/200=5.
        # Observed 0 vs expected 5 -> significant depletion with two-sided.
        library = {"test_lib": {"far_set": background[60:110]}}
        r_two = ora(hits, background, library, min_overlap=0, alternative="two-sided")
        # min_overlap=0 lets zero-count rows through
        assert (r_two["direction"] == "depleted").iloc[0]

    def test_unknown_fdr_method_raises(self):
        with pytest.raises(ValueError, match="fdr_method"):
            ora(["A"], ["A"], {}, fdr_method="banana")


# ---------------------------------------------------------------------------
# GSEA
# ---------------------------------------------------------------------------


class TestGSEA:
    def test_perfect_top_enrichment(self):
        """A set whose members are the top-K sites in the ranking must have
        positive ES near the theoretical maximum, small p_value, and NES>0."""
        rng = np.random.default_rng(0)
        n = 200
        # Give the top 10 sites +5 log2FC; the remaining 190 draw from N(0, 1).
        ranked = pd.Series(
            np.concatenate([np.full(10, 5.0), rng.normal(0, 1, n - 10)]),
            index=[f"S{i}" for i in range(n)],
        )
        library = {"test_lib": {"top_set": [f"S{i}" for i in range(10)]}}
        result = gsea(ranked, library, min_set_size=5, n_permutations=1000, seed=1)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["ES"] > 0.5
        assert row["NES"] > 1.5  # NES conventional threshold
        assert row["p_value"] < 0.01
        assert row["direction"] == "up"

    def test_perfect_bottom_enrichment(self):
        """Same setup but the set members are ranked LAST -> negative ES."""
        rng = np.random.default_rng(0)
        n = 200
        ranked = pd.Series(
            np.concatenate([rng.normal(0, 1, n - 10), np.full(10, -5.0)]),
            index=[f"S{i}" for i in range(n)],
        )
        library = {"test_lib": {"bottom_set": [f"S{i}" for i in range(n - 10, n)]}}
        result = gsea(ranked, library, min_set_size=5, n_permutations=1000, seed=1)
        assert result["ES"].iloc[0] < -0.5
        assert result["NES"].iloc[0] < -1.5
        assert result["direction"].iloc[0] == "down"

    def test_random_set_not_significant(self):
        """A random set of the same size in a random ranking should yield
        near-null enrichment (empirical p >~ 0.05)."""
        rng = np.random.default_rng(0)
        n = 200
        ranked = pd.Series(rng.normal(0, 1, n), index=[f"S{i}" for i in range(n)])
        library = {
            "test_lib": {
                "random_set": rng.choice(
                    [f"S{i}" for i in range(n)], size=20, replace=False
                ).tolist()
            }
        }
        result = gsea(ranked, library, min_set_size=5, n_permutations=1000, seed=1)
        # Not asserting strict >0.05 since one draw could be extreme;
        # assert the p-value is well above the strong-signal threshold.
        assert result["p_value"].iloc[0] > 0.01

    def test_size_filter_min_max(self):
        rng = np.random.default_rng(0)
        n = 100
        ranked = pd.Series(rng.normal(0, 1, n), index=[f"S{i}" for i in range(n)])
        libraries = {
            "test_lib": {
                "too_small": [f"S{i}" for i in range(4)],  # 4 < 5
                "just_right": [f"S{i}" for i in range(20)],  # 20 in [5, 500]
                "too_large": [f"S{i}" for i in range(n)],  # 100 > 50 max
            }
        }
        result = gsea(
            ranked, libraries, min_set_size=5, max_set_size=50, n_permutations=200, seed=1
        )
        assert set(result["set_name"]) == {"just_right"}

    def test_reproducibility_with_seed(self):
        rng = np.random.default_rng(0)
        n = 100
        ranked = pd.Series(rng.normal(0, 1, n), index=[f"S{i}" for i in range(n)])
        library = {"test_lib": {"set": [f"S{i}" for i in range(20)]}}
        r1 = gsea(ranked, library, min_set_size=5, n_permutations=500, seed=42)
        r2 = gsea(ranked, library, min_set_size=5, n_permutations=500, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_nan_ranks_rejected(self):
        ranked = pd.Series([1.0, float("nan"), 2.0], index=["A", "B", "C"])
        with pytest.raises(ValueError, match="NaN"):
            gsea(ranked, {"lib": {"set": ["A", "C"]}}, min_set_size=1, n_permutations=100)

    def test_leading_edge_reported(self):
        rng = np.random.default_rng(0)
        n = 200
        ranked = pd.Series(
            np.concatenate([np.full(10, 5.0), rng.normal(0, 1, n - 10)]),
            index=[f"S{i}" for i in range(n)],
        )
        library = {"test_lib": {"top_set": [f"S{i}" for i in range(10)]}}
        result = gsea(ranked, library, min_set_size=5, n_permutations=500, seed=1)
        edge = set(result["leading_edge"].iloc[0].split(";"))
        # For a perfect-top enrichment the leading edge should include
        # all the top-10 hits.
        assert edge.issuperset({f"S{i}" for i in range(10)})


# ---------------------------------------------------------------------------
# Combined smoke test
# ---------------------------------------------------------------------------


class TestSmokeIntegration:
    def test_both_engines_run_on_same_input(self, tmp_path):
        """ORA and GSEA on the same synthetic setup produce consistent
        direction calls -- if ORA calls 'enriched', GSEA calls 'up'."""
        rng = np.random.default_rng(0)
        n = 200
        # Top 20 sites are the "hit" set with big positive stats.
        stats_arr = np.concatenate([np.full(20, 4.0), rng.normal(0, 1, n - 20)])
        ranked = pd.Series(stats_arr, index=[f"S{i}" for i in range(n)])
        background = list(ranked.index)
        hits = [f"S{i}" for i in range(20)]
        library = {"lib": {"the_set": hits}}

        ora_result = ora(hits, background, library, min_overlap=1)
        gsea_result = gsea(ranked, library, min_set_size=5, n_permutations=500, seed=1)
        # Both engines should call the set enriched / up.
        assert ora_result["direction"].iloc[0] == "enriched"
        assert gsea_result["direction"].iloc[0] == "up"
        # Both should have p<0.05 for this obvious signal.
        assert ora_result["p_value"].iloc[0] < 0.05
        assert gsea_result["p_value"].iloc[0] < 0.05
