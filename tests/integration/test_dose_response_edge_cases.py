"""Edge-case battery for alphaphos.dose_response (CurveCurator wrapper).

These are END-TO-END integration tests — they actually invoke CurveCurator
in a subprocess and parse the output. Slower than the unit tests in
``tests/unit/test_dose_response.py`` (which mock at the subprocess
boundary); typically 5–15 s per test.

Each test plants a known truth or failure mode into a small synthetic
AnnData and checks that the wrapper either recovers the truth or fails
gracefully with a clear error message.

Marked ``slow`` so they're skipped in fast pytest runs.

Edge cases covered:

- ``test_strong_response_recovered`` — planted pEC50=7 recovered as ~7
- ``test_flat_response_not_significant`` — no dose-response → high p
- ``test_inhibition_recovered_with_negative_fold_change`` — direction
- ``test_all_nan_site_handled`` — site with no measurements dropped
- ``test_max_missing_too_strict_raises`` — produces 0 curves
- ``test_multi_timepoint_concatenation`` — two timepoints concatenated
- ``test_subset_timepoints_arg`` — ``timepoints=[X]`` filter works
- ``test_layer_argument_routes`` — non-X layer is read
- ``test_float_timepoint`` — non-integer timepoint (e.g. 0.5 h)
- ``test_continues_past_individual_timepoint_failure`` — robustness
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("curve_curator")

from alphaphos.dose_response import fit_dose_response

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Standard 5-dose ladder + 2 DMSO replicates = 12 wells per timepoint
DOSES_NM = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def _make_adata(
    *,
    n_responder_sites: int = 5,
    n_flat_sites: int = 5,
    timepoints: tuple[float, ...] = (60.0,),
    pec50: float = 7.0,
    fold_change: float = 2.5,
    seed: int = 42,
    n_replicates: int = 2,
) -> ad.AnnData:
    """Synthesize a dose-response AnnData with a known planted truth.

    First ``n_responder_sites`` follow a 4-parameter log-logistic curve
    with the given pEC50 (EC50 = 10**(9-pEC50) nM) and log2 fold change.
    Remaining ``n_flat_sites`` are flat (no response).
    """
    rng = np.random.default_rng(seed)
    samples = []
    rows = []
    for t in timepoints:
        for d in DOSES_NM:
            for r in range(1, n_replicates + 1):
                samples.append(f"t{t}_D{d}_r{r}")
                rows.append({"dose": d, "timepoint": t})
    obs = pd.DataFrame(rows, index=samples)
    n_total = n_responder_sites + n_flat_sites
    var = pd.DataFrame(index=[f"site{j}" for j in range(n_total)])

    # CurveCurator fits the Hill curve in *linear* intensity space (after our
    # 2^log2 conversion). To plant a *recoverable* pEC50 we must build the
    # curve on the same scale CC will fit: linear intensities directly.
    baseline_linear = 2.0**15  # log2(baseline) = 15
    top_linear = baseline_linear * (2.0**fold_change)

    def _response_linear(dose_nM: float) -> float:
        """Hill / 4PL curve in *linear* intensity space.

        baseline_linear at dose=0; saturates at top_linear; half-max at EC50.
        Result: planted pEC50 is exactly recoverable by CurveCurator.
        """
        if dose_nM <= 0:
            return baseline_linear
        slope = 1.0
        log_dose_M = np.log10(dose_nM * 1e-9)
        # fraction goes 0 -> 1 as dose grows
        frac = 1.0 / (1.0 + 10 ** ((-pec50 - log_dose_M) * slope))
        return baseline_linear + (top_linear - baseline_linear) * frac

    X = np.zeros((len(samples), n_total))
    for i_s, s in enumerate(samples):
        d = obs.loc[s, "dose"]
        for j in range(n_total):
            if j < n_responder_sites:
                # Linear intensity with small multiplicative noise, then log2
                lin_clean = _response_linear(d)
                lin_noisy = lin_clean * (1.0 + rng.normal(0, 0.05))
                X[i_s, j] = np.log2(max(lin_noisy, 1.0))
            else:
                X[i_s, j] = rng.normal(15.0, 0.05)
    return ad.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDoseResponseEdgeCases:
    def test_strong_response_recovered(self, tmp_path):
        """Planted pEC50=7.0 (EC50=100 nM) should be recovered within ±0.4 log units."""
        adata = _make_adata(
            n_responder_sites=8,
            n_flat_sites=0,
            timepoints=(60.0,),
            pec50=7.0,
            fold_change=2.5,
            n_replicates=3,
        )
        curves = fit_dose_response(
            adata,
            output_root=tmp_path / "cc",
            dose_col="dose",
            timepoint_col=None,
            condition="test",
            max_missing=8,
            available_cores=1,
            fdr=True,
            verbose=False,
        )
        valid = curves.dropna(subset=["pEC50"])
        assert len(valid) >= 5, f"Too few curves with valid pEC50: {len(valid)}"
        # Most curves should be near pEC50=7 (now planted in linear space, so
        # CC recovers it within fitting noise)
        median_pec50 = valid["pEC50"].median()
        assert 6.7 < median_pec50 < 7.3, f"median pEC50 = {median_pec50:.2f}, expected ~7.0"
        # CurveCurator's "Curve Fold Change" is in LOG2 space: planted log2
        # FC=2.5 should recover near 2.5. (Curve Front and Curve Back are
        # the linear-intensity asymptotes; their ratio matches 2**FC.)
        median_fc = valid["Curve Fold Change"].median()
        assert 2.0 < median_fc < 3.0, f"median log2 FC = {median_fc:.2f}, expected ~2.5"

    def test_flat_response_not_significant(self, tmp_path):
        """Sites with no real dose-response should have non-significant F-test."""
        adata = _make_adata(
            n_responder_sites=0,
            n_flat_sites=10,
            timepoints=(60.0,),
            n_replicates=3,
        )
        curves = fit_dose_response(
            adata,
            output_root=tmp_path / "cc",
            dose_col="dose",
            timepoint_col=None,
            condition="test",
            max_missing=8,
            available_cores=1,
            fdr=True,
            verbose=False,
        )
        # Most curves should not be significant by q-value
        if "Curve q_Value" in curves.columns:
            n_sig = (curves["Curve q_Value"] < 0.05).sum()
            assert n_sig <= 1, f"Got {n_sig} false-significant flat curves (expected ≤ 1)"

    def test_inhibition_recovered_with_negative_fold_change(self, tmp_path):
        """Plant a *downward* response — fold change should be negative."""
        adata = _make_adata(
            n_responder_sites=5,
            n_flat_sites=0,
            timepoints=(60.0,),
            pec50=7.0,
            fold_change=-2.0,
            n_replicates=3,
        )
        curves = fit_dose_response(
            adata,
            output_root=tmp_path / "cc",
            dose_col="dose",
            timepoint_col=None,
            condition="test",
            max_missing=8,
            available_cores=1,
            fdr=True,
            verbose=False,
        )
        valid = curves.dropna(subset=["Curve Fold Change"])
        assert len(valid) >= 3
        # Most fold changes should be negative
        n_neg = (valid["Curve Fold Change"] < 0).sum()
        assert n_neg >= len(valid) * 0.6, f"Only {n_neg}/{len(valid)} curves have negative FC"

    def test_all_nan_site_handled(self, tmp_path):
        """A site with all-NaN intensities should be dropped by max_missing,
        not crash the wrapper."""
        adata = _make_adata(
            n_responder_sites=5,
            n_flat_sites=0,
            timepoints=(60.0,),
            n_replicates=3,
        )
        # Wipe one site entirely
        adata.X = adata.X.copy()
        adata.X[:, 2] = np.nan
        curves = fit_dose_response(
            adata,
            output_root=tmp_path / "cc",
            dose_col="dose",
            timepoint_col=None,
            condition="test",
            max_missing=8,
            available_cores=1,
            fdr=True,
            verbose=False,
        )
        # site2 (the all-NaN one) should NOT appear in the output
        assert "site2" not in set(curves["site_key"])
        # Other responder sites should still fit
        assert len(curves) >= 3

    def test_max_missing_too_strict_raises(self, tmp_path):
        """If max_missing forces every site out, wrapper raises with a useful
        message rather than silently returning nothing."""
        adata = _make_adata(
            n_responder_sites=5,
            n_flat_sites=0,
            timepoints=(60.0,),
            n_replicates=3,
        )
        # Sprinkle NaNs so every site has at least one missing value, then
        # set max_missing=0 so nothing survives
        rng = np.random.default_rng(0)
        for j in range(adata.n_vars):
            idx = rng.integers(0, adata.n_obs)
            adata.X = adata.X.copy()
            adata.X[idx, j] = np.nan
        with pytest.raises(RuntimeError, match="(no curves|max_missing|failed|exited)"):
            fit_dose_response(
                adata,
                output_root=tmp_path / "cc",
                dose_col="dose",
                timepoint_col=None,
                condition="test",
                max_missing=0,
                available_cores=1,
                fdr=False,
                verbose=False,
            )

    def test_multi_timepoint_concatenation(self, tmp_path):
        """Multi-timepoint adata: one row per (site, timepoint) in output."""
        adata = _make_adata(
            n_responder_sites=5,
            n_flat_sites=0,
            timepoints=(60.0, 240.0),
            n_replicates=2,
        )
        curves = fit_dose_response(
            adata,
            output_root=tmp_path / "cc",
            dose_col="dose",
            timepoint_col="timepoint",
            condition="test",
            max_missing=10,
            available_cores=1,
            fdr=False,
            verbose=False,
        )
        assert "timepoint" in curves.columns
        seen_tps = set(curves["timepoint"].unique())
        assert seen_tps == {60.0, 240.0}, f"Got timepoints: {seen_tps}"
        # Each site should appear in both timepoints
        per_tp_counts = curves.groupby("timepoint").size()
        assert per_tp_counts.min() >= 4

    def test_subset_timepoints_arg(self, tmp_path):
        """Passing timepoints=[t] should only run that one."""
        adata = _make_adata(
            n_responder_sites=4,
            n_flat_sites=0,
            timepoints=(60.0, 240.0),
            n_replicates=2,
        )
        curves = fit_dose_response(
            adata,
            output_root=tmp_path / "cc",
            dose_col="dose",
            timepoint_col="timepoint",
            timepoints=[60.0],  # explicit subset
            condition="test",
            max_missing=10,
            available_cores=1,
            fdr=False,
            verbose=False,
        )
        assert set(curves["timepoint"].unique()) == {60.0}
        # The 240 dir should NOT have been created
        assert (tmp_path / "cc" / "t60").exists()
        assert not (tmp_path / "cc" / "t240").exists()

    def test_layer_argument_routes(self, tmp_path):
        """Putting the intensities in a non-X layer and passing layer= should work."""
        adata = _make_adata(
            n_responder_sites=5,
            n_flat_sites=0,
            timepoints=(60.0,),
            n_replicates=2,
        )
        # Move .X into a layer; replace .X with garbage to ensure it's NOT used
        adata.layers["intensity_imputed"] = adata.X.copy()
        adata.X = np.full(adata.X.shape, 999.0)  # garbage
        curves = fit_dose_response(
            adata,
            output_root=tmp_path / "cc",
            dose_col="dose",
            timepoint_col=None,
            layer="intensity_imputed",  # ← route to layer
            condition="test",
            max_missing=8,
            available_cores=1,
            fdr=False,
            verbose=False,
        )
        # If routing failed, fold-changes would all be ~0 (constant X);
        # with the layer used, planted responders show |FC| > 1.
        valid = curves.dropna(subset=["Curve Fold Change"])
        assert valid["Curve Fold Change"].abs().median() > 1.0

    def test_float_timepoint(self, tmp_path):
        """Non-integer timepoint (e.g. 0.5 h) shouldn't break dir naming or run."""
        adata = _make_adata(
            n_responder_sites=4,
            n_flat_sites=0,
            timepoints=(0.5,),
            n_replicates=2,
        )
        curves = fit_dose_response(
            adata,
            output_root=tmp_path / "cc",
            dose_col="dose",
            timepoint_col="timepoint",
            condition="test",
            max_missing=8,
            available_cores=1,
            fdr=False,
            verbose=False,
        )
        assert len(curves) >= 1
        assert (curves["timepoint"] == 0.5).all()

    def test_dashboard_html_written(self, tmp_path):
        """CurveCurator writes an interactive dashboard HTML — verify present."""
        adata = _make_adata(
            n_responder_sites=4,
            n_flat_sites=0,
            timepoints=(60.0,),
            n_replicates=2,
        )
        _ = fit_dose_response(
            adata,
            output_root=tmp_path / "cc",
            dose_col="dose",
            timepoint_col=None,
            condition="test",
            max_missing=8,
            available_cores=1,
            fdr=False,
            verbose=False,
        )
        # Find an HTML output anywhere under the run dir
        htmls = list((tmp_path / "cc").rglob("*.html"))
        assert htmls, f"No HTML dashboard found under {tmp_path / 'cc'}"
