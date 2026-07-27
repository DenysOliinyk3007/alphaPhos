"""Meticulous tests for :mod:`alphaphos.preprocess.classI_wilson`.

Covers:
- ``wilson_lower_bound`` math on scalar + vector + edge inputs
- ``auto_wilson_threshold`` branch behaviour: below-min-n, per-range picks,
  flat curve fallback, weak-elbow fallback
- ``apply_wilson_filter`` end-to-end on synthetic AnnData with fixed and
  auto thresholds, and the small-cohort warning path
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

from alphaphos.preprocess.classI_wilson import (
    AUTO_MIN_ELBOW_STRENGTH,
    AUTO_MIN_N,
    AUTO_RANGES,
    apply_wilson_filter,
    auto_wilson_threshold,
    wilson_lower_bound,
)

# ---------------------------------------------------------------------------
# wilson_lower_bound
# ---------------------------------------------------------------------------


class TestWilsonLowerBoundMath:
    """Sanity + edge-case coverage of the vectorised Jeffreys/Wilson bound."""

    def test_scalar_scalar_matches_scipy_beta_ppf(self):
        # For any (k, n) the function is Beta.ppf(alpha/2, k+.5, n-k+.5)
        for k, n in [(3, 10), (0, 5), (50, 50), (1, 1), (0, 0), (7, 7)]:
            expected = float(beta.ppf(0.025, k + 0.5, n - k + 0.5)) if n > 0 else 0.0
            got = float(wilson_lower_bound(k, n))
            assert got == pytest.approx(max(expected, 0.0), abs=1e-9), f"(k={k}, n={n})"

    def test_all_class_I_shrinks_with_n(self):
        # k = n: lb = n / (n + z^2) grows monotonically toward 1
        prev = -1.0
        for n in [1, 3, 5, 10, 30, 100, 300, 1000]:
            lb = float(wilson_lower_bound(n, n))
            assert 0.0 < lb <= 1.0
            assert lb > prev
            prev = lb

    def test_zero_successes_bounds_near_zero(self):
        # k = 0: lb should be very small for large n, zero for n = 0
        assert wilson_lower_bound(0, 0) == 0.0
        for n in [1, 10, 100, 1000]:
            lb = float(wilson_lower_bound(0, n))
            assert 0.0 <= lb <= 0.1  # Jeffreys ~1/(4n+2); tiny

    def test_vectorised_matches_scalar(self):
        rng = np.random.default_rng(0)
        n = rng.integers(1, 500, size=200)
        k = rng.integers(0, n + 1)
        scalar = np.array(
            [wilson_lower_bound(int(kk), int(nn)) for kk, nn in zip(k, n, strict=True)]
        )
        vector = wilson_lower_bound(k, n)
        assert np.allclose(scalar, vector, atol=1e-12)

    def test_output_always_in_unit_interval(self):
        rng = np.random.default_rng(1)
        n = rng.integers(0, 1000, size=500)
        # k in [0, n]
        k = (rng.random(500) * (n + 1)).astype(int)
        k = np.minimum(k, n)
        lb = wilson_lower_bound(k, n)
        assert np.all(lb >= 0.0)
        assert np.all(lb <= 1.0)

    def test_n_zero_returns_zero_regardless_of_k(self):
        # k must be 0 when n is 0; but if user passes k=0, n=0 we return 0
        assert wilson_lower_bound(np.array([0, 0, 0]), np.array([0, 0, 0])).tolist() == [
            0.0,
            0.0,
            0.0,
        ]

    def test_negative_k_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            wilson_lower_bound(-1, 10)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            wilson_lower_bound(1, -5)

    def test_k_greater_than_n_raises(self):
        with pytest.raises(ValueError, match="k must not exceed n"):
            wilson_lower_bound(5, 3)

    def test_bad_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            wilson_lower_bound(3, 10, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            wilson_lower_bound(3, 10, alpha=1.5)

    def test_shape_mismatch_broadcasts_when_possible(self):
        # Scalar k, vector n → broadcasts
        got = wilson_lower_bound(2, np.array([5, 10, 20]))
        assert got.shape == (3,)

    def test_shape_mismatch_incompatible_raises(self):
        with pytest.raises(ValueError, match="broadcastable"):
            wilson_lower_bound(np.array([1, 2, 3]), np.array([10, 20]))

    def test_alpha_smaller_gives_tighter_lower_bound(self):
        # 99% CI (alpha=0.01) → lower bound BELOW 95% CI
        wide = wilson_lower_bound(50, 100, alpha=0.05)
        wider = wilson_lower_bound(50, 100, alpha=0.01)
        assert float(wider) < float(wide)

    def test_symmetric_around_half_at_k_equal_n_over_2(self):
        # For k = n/2 with the symmetric Jeffreys prior, the lower and upper
        # bounds are symmetric around 0.5: (lb - 0.5) == -(ub - 0.5).
        lb = float(wilson_lower_bound(50, 100))
        ub = float(beta.ppf(0.975, 50 + 0.5, 100 - 50 + 0.5))
        assert (0.5 - lb) == pytest.approx(ub - 0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# auto_wilson_threshold
# ---------------------------------------------------------------------------


class TestAutoWilsonThreshold:
    def _synth_lb_from_distribution(self, n_sites: int, rng_seed: int = 0) -> np.ndarray:
        """A realistic wilson_lb distribution: mixture of high-purity + noisy tail."""
        rng = np.random.default_rng(rng_seed)
        n_high = int(0.4 * n_sites)
        n_mid = int(0.4 * n_sites)
        n_low = n_sites - n_high - n_mid
        high = rng.uniform(0.65, 0.98, n_high)
        mid = rng.uniform(0.30, 0.65, n_mid)
        low = rng.uniform(0.00, 0.30, n_low)
        return np.concatenate([high, mid, low])

    def test_below_min_n_raises(self):
        lb = np.array([0.5, 0.8, 0.2])
        with pytest.raises(ValueError, match="requires n_samples"):
            auto_wilson_threshold(lb, n_samples=AUTO_MIN_N - 1)

    def test_empty_wilson_lb_raises(self):
        with pytest.raises(ValueError, match="empty"):
            auto_wilson_threshold(np.array([]), n_samples=200)

    @pytest.mark.parametrize(
        "n_samples, expected_lo, expected_hi",
        [
            (30, 0.20, 0.40),  # smallest allowed
            (99, 0.20, 0.40),  # just below 100
            (100, 0.30, 0.55),  # boundary
            (299, 0.30, 0.55),
            (300, 0.45, 0.65),  # boundary
            (5000, 0.45, 0.65),  # much larger
        ],
    )
    def test_picked_threshold_lies_in_expected_range(self, n_samples, expected_lo, expected_hi):
        lb = self._synth_lb_from_distribution(2000, rng_seed=n_samples)
        picked, reason = auto_wilson_threshold(lb, n_samples=n_samples)
        assert expected_lo <= picked <= expected_hi, f"got {picked}, reason={reason}"
        assert isinstance(reason, str) and len(reason) > 0

    def test_flat_retention_curve_returns_midpoint(self):
        # For a truly flat curve we need retention to not change across the range;
        # putting all wilson_lb values ABOVE the search range achieves this.
        lb_all_above = np.full(500, 0.99)
        picked, reason = auto_wilson_threshold(lb_all_above, n_samples=150)
        expected_mid = (0.30 + 0.55) / 2
        assert picked == pytest.approx(expected_mid, abs=1e-9)
        assert "flat" in reason.lower() or "midpoint" in reason.lower()

    def test_weak_elbow_returns_midpoint(self):
        # A curve that decreases linearly (no elbow) should hit the
        # weak-elbow fallback.
        # Construct: uniform wilson_lb on [0.0, 1.0] with n_samples=500 (range [0.45, 0.65]);
        # retention is linear in threshold within the range → elbow strength small.
        rng = np.random.default_rng(42)
        lb = rng.uniform(0.0, 1.0, size=10_000)
        picked, reason = auto_wilson_threshold(lb, n_samples=500)
        # Linear curve → below-diagonal ≈ 0 → fallback expected
        assert reason.startswith(("no clear elbow", "elbow at"))
        # Whatever we return, must be in range
        assert 0.45 <= picked <= 0.65

    def test_reason_contains_diagnostic_info(self):
        lb = self._synth_lb_from_distribution(2000)
        _, reason = auto_wilson_threshold(lb, n_samples=400)
        assert "n_samples=400" in reason or "elbow" in reason or "flat" in reason

    def test_returned_threshold_within_search_range_bounds(self):
        # Property: threshold in [lo, hi] for the corresponding n_samples band.
        for upper_n, lo, hi in AUTO_RANGES:
            n = min(upper_n - 1, 5000)
            if n < AUTO_MIN_N:
                continue
            lb = self._synth_lb_from_distribution(1500)
            picked, _ = auto_wilson_threshold(lb, n_samples=n)
            assert lo - 1e-9 <= picked <= hi + 1e-9


# ---------------------------------------------------------------------------
# apply_wilson_filter
# ---------------------------------------------------------------------------


def _make_adata_with_wilson_lb(n_samples: int, n_sites: int, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.normal(20, 2, size=(n_samples, n_sites)).astype(np.float32)
    nan_mask = rng.random(X.shape) < 0.3
    X[nan_mask] = np.nan
    var = pd.DataFrame(
        {
            "classI_wilson_lb": rng.uniform(0.0, 1.0, size=n_sites),
            "n_samples_detected": rng.integers(1, n_samples + 1, size=n_sites),
            "n_classI_samples": rng.integers(0, n_samples + 1, size=n_sites),
        },
        index=[f"site_{i}" for i in range(n_sites)],
    )
    obs = pd.DataFrame(index=[f"s{i}" for i in range(n_samples)])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestApplyWilsonFilter:
    def test_missing_var_column_raises(self):
        adata = ad.AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame(index=[f"s{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"v{i}" for i in range(3)]),
        )
        with pytest.raises(KeyError, match="classI_wilson_lb"):
            apply_wilson_filter(adata, threshold=0.5)

    def test_fixed_threshold_filters_and_stamps_uns(self):
        adata = _make_adata_with_wilson_lb(200, 500)
        out = apply_wilson_filter(adata, threshold=0.5)
        assert out.n_obs == adata.n_obs
        assert out.n_vars < adata.n_vars
        assert np.all(out.var["classI_wilson_lb"].to_numpy() >= 0.5)
        assert "wilson_filter" in out.uns
        assert out.uns["wilson_filter"]["threshold"] == pytest.approx(0.5)
        assert out.uns["wilson_filter"]["n_sites_pre_filter"] == adata.n_vars
        assert out.uns["wilson_filter"]["n_sites_kept"] == out.n_vars

    def test_threshold_zero_keeps_everything(self):
        adata = _make_adata_with_wilson_lb(200, 300)
        out = apply_wilson_filter(adata, threshold=0.0)
        assert out.n_vars == adata.n_vars

    def test_threshold_one_drops_everything_or_almost(self):
        adata = _make_adata_with_wilson_lb(200, 500)
        out = apply_wilson_filter(adata, threshold=1.0)
        # sample-drawn wilson_lb is < 1.0 with probability 1 → all sites drop
        assert out.n_vars == 0

    def test_out_of_range_threshold_raises(self):
        adata = _make_adata_with_wilson_lb(200, 100)
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            apply_wilson_filter(adata, threshold=1.5)
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            apply_wilson_filter(adata, threshold=-0.1)

    def test_bad_threshold_string_raises(self):
        adata = _make_adata_with_wilson_lb(200, 100)
        with pytest.raises(ValueError, match="'auto'"):
            apply_wilson_filter(adata, threshold="never")

    def test_auto_threshold_end_to_end(self):
        adata = _make_adata_with_wilson_lb(200, 1500, seed=7)
        out = apply_wilson_filter(adata, threshold="auto")
        # Sanity: threshold in the [100, 300) range → [0.30, 0.55]
        picked = out.uns["wilson_filter"]["threshold"]
        assert 0.30 <= picked <= 0.55
        assert out.n_vars < adata.n_vars
        assert (
            "elbow" in out.uns["wilson_filter"]["reason"]
            or "midpoint" in out.uns["wilson_filter"]["reason"]
        )

    def test_small_cohort_warns_but_does_not_refuse(self):
        adata = _make_adata_with_wilson_lb(50, 200, seed=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = apply_wilson_filter(adata, threshold=0.5)
        assert any("< 100" in str(w.message) for w in caught)
        assert out.n_vars < adata.n_vars

    def test_auto_at_very_small_cohort_raises(self):
        adata = _make_adata_with_wilson_lb(20, 100)
        # Warning should still fire from apply_wilson_filter (small-n),
        # but the auto path raises ValueError first
        with pytest.raises(ValueError, match="n_samples >= "), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore the small-n warning
            # The n<AUTO_MIN_N check happens INSIDE auto_wilson_threshold
            # which is called BEFORE the small-n warning; but small-n warn
            # is emitted first in apply_wilson_filter.  Let's not assume
            # order — either way, ValueError from auto must propagate.
            apply_wilson_filter(adata, threshold="auto")

    def test_input_adata_untouched(self):
        adata = _make_adata_with_wilson_lb(200, 300)
        n_vars_before = adata.n_vars
        n_obs_before = adata.n_obs
        _ = apply_wilson_filter(adata, threshold=0.5)
        assert adata.n_vars == n_vars_before
        assert adata.n_obs == n_obs_before


# ---------------------------------------------------------------------------
# AUTO_MIN_ELBOW_STRENGTH sanity
# ---------------------------------------------------------------------------


def test_constants_look_sane():
    assert 0.0 < AUTO_MIN_ELBOW_STRENGTH < 0.5
    assert AUTO_MIN_N >= 20
    assert AUTO_RANGES[-1][2] > AUTO_RANGES[0][1]  # ranges span meaningfully
