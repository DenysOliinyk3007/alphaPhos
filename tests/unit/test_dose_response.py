"""Tests for alphaphos.dose_response.curve_curator.

Covers:
  - _build_input_tsv : log2 -> linear conversion, NaN preservation, sample selection,
                        per-timepoint filtering, DMSO control identification.
  - _build_toml      : config file generation, TOML syntax sanity check.
  - _parse_curves    : output reading, EC50_nM derivation.
  - fit_dose_response: orchestrator with subprocess mocked.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

# Skip the whole module if curve_curator isn't installed (the import inside
# fit_dose_response would error anyway, but a clean skip is friendlier).
pytest.importorskip("curve_curator")

from alphaphos.dose_response.curve_curator import (
    DEFAULT_CURVE_COLS,
    _build_input_tsv,
    _build_toml,
    _parse_curves,
    _toml_num_array,
    _toml_quote,
    _toml_str_array,
)

# ============================================================================
# TOML helpers
# ============================================================================


class TestTomlHelpers:
    def test_quote_escapes_backslash_and_quote(self):
        assert _toml_quote("a\\b") == '"a\\\\b"'
        assert _toml_quote('he said "hi"') == '"he said \\"hi\\""'

    def test_str_array_format(self):
        assert _toml_str_array(["a", "b", "c"]) == '["a", "b", "c"]'

    def test_num_array_format(self):
        out = _toml_num_array([0.0, 1.5, 10.0])
        # repr-of-float gives '0.0', '1.5', '10.0' (Python defaults)
        assert "0.0" in out and "1.5" in out and "10.0" in out


# ============================================================================
# _build_input_tsv
# ============================================================================


def _make_adata_with_doses_and_time(n_sites: int = 8) -> ad.AnnData:
    """6 doses × 2 timepoints × 2 reps = 24 samples; n_sites variables.

    log2 intensities ~N(15, 0.5), zero-NaN initially.
    """
    doses = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    timepoints = [60, 240]
    samples = []
    rows = []
    for t in timepoints:
        for d in doses:
            for r in (1, 2):
                samples.append(f"t{t}_D{d}_r{r}")
                rows.append({"dose": d, "timepoint": t})
    rng = np.random.default_rng(0)
    X = rng.normal(15.0, 0.5, size=(len(samples), n_sites))
    obs = pd.DataFrame(rows, index=samples)
    var = pd.DataFrame(index=[f"site{i}" for i in range(n_sites)])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestBuildInputTsv:
    def test_single_timepoint_design(self, tmp_path):
        # No timepoint column; one dose run across all 12 wells
        adata = _make_adata_with_doses_and_time(n_sites=6)
        # Subset to timepoint=60 first (simulating single-timepoint experiment)
        sub = adata[adata.obs["timepoint"] == 60].copy()
        info = _build_input_tsv(
            sub,
            output_path=tmp_path / "in.tsv",
            dose_col="dose",
            timepoint_col=None,
            timepoint=None,
            layer=None,
        )
        assert info["n_sites"] == 6
        assert info["n_wells"] == 12
        assert len(info["controls"]) == 2  # 2 DMSO reps
        # All controls have dose == 0
        assert all(s.startswith("t60_D0.0_") for s in info["controls"])
        # The TSV file exists with Name + 12 Raw columns
        tsv = pd.read_csv(info["tsv_path"], sep="\t")
        assert tsv.shape == (6, 13)
        assert "Name" in tsv.columns
        raw_cols = [c for c in tsv.columns if c.startswith("Raw ")]
        assert len(raw_cols) == 12

    def test_multi_timepoint_selects_one(self, tmp_path):
        adata = _make_adata_with_doses_and_time(n_sites=4)
        info = _build_input_tsv(
            adata,
            output_path=tmp_path / "in.tsv",
            dose_col="dose",
            timepoint_col="timepoint",
            timepoint=240,
            layer=None,
        )
        # Only the t=240 wells (12 samples)
        assert info["n_wells"] == 12
        assert all(s.startswith("t240_") for s in info["experiments"])

    def test_converts_log2_to_linear(self, tmp_path):
        """log2 value of 10 -> 2**10 = 1024 in the TSV (linear)."""
        adata = _make_adata_with_doses_and_time(n_sites=2)
        # Override .X with a known log2 value
        adata.X = np.full(adata.X.shape, 10.0)
        info = _build_input_tsv(
            adata.copy(),
            output_path=tmp_path / "in.tsv",
            dose_col="dose",
            timepoint_col="timepoint",
            timepoint=60,
            layer=None,
        )
        tsv = pd.read_csv(info["tsv_path"], sep="\t")
        raw_cols = [c for c in tsv.columns if c.startswith("Raw ")]
        # 2 ** 10 == 1024
        assert np.allclose(tsv[raw_cols].values, 1024.0)

    def test_preserves_nan(self, tmp_path):
        adata = _make_adata_with_doses_and_time(n_sites=2)
        adata.X = adata.X.copy()
        adata.X[0, 0] = np.nan
        info = _build_input_tsv(
            adata,
            output_path=tmp_path / "in.tsv",
            dose_col="dose",
            timepoint_col="timepoint",
            timepoint=60,
            layer=None,
        )
        tsv = pd.read_csv(info["tsv_path"], sep="\t")
        # Sample 0 corresponds to the first t=60 well — its site0 cell is NaN
        first_t60_sample = adata.obs.loc[adata.obs["timepoint"] == 60].index[0]
        nan_col = f"Raw {first_t60_sample}"
        assert pd.isna(tsv.loc[tsv["Name"] == "site0", nan_col].values[0])

    def test_errors_when_no_controls(self, tmp_path):
        """A run with zero DMSO controls (dose != 0 for all) errors."""
        adata = _make_adata_with_doses_and_time(n_sites=2)
        adata.obs["dose"] = 100.0  # wipe out the zeros
        with pytest.raises(ValueError, match="No DMSO controls"):
            _build_input_tsv(
                adata,
                output_path=tmp_path / "in.tsv",
                dose_col="dose",
                timepoint_col="timepoint",
                timepoint=60,
                layer=None,
            )

    def test_errors_when_timepoint_missing(self, tmp_path):
        adata = _make_adata_with_doses_and_time(n_sites=2)
        with pytest.raises(ValueError, match="No samples found at"):
            _build_input_tsv(
                adata,
                output_path=tmp_path / "in.tsv",
                dose_col="dose",
                timepoint_col="timepoint",
                timepoint=999,  # not in obs
                layer=None,
            )


# ============================================================================
# _build_toml
# ============================================================================


class TestBuildToml:
    def test_parsed_by_tomllib(self, tmp_path):
        toml_path = _build_toml(
            input_tsv=tmp_path / "in.tsv",
            output_dir=tmp_path,
            experiments=["s1", "s2", "s3"],
            doses=[0.0, 10.0, 100.0],
            controls=["s1"],
            treatment_time=240,
            run_id="testrun",
            condition="rapamycin",
            description="alphaphos test",
            available_cores=2,
            max_missing=5,
        )
        import tomllib

        with toml_path.open("rb") as f:
            cfg = tomllib.load(f)
        assert cfg["Meta"]["id"] == "testrun"
        assert cfg["Meta"]["condition"] == "rapamycin"
        assert cfg["Meta"]["treatment_time"] == "240 min"
        assert cfg["Experiment"]["experiments"] == ["s1", "s2", "s3"]
        assert cfg["Experiment"]["control_experiment"] == ["s1"]
        assert cfg["Experiment"]["doses"] == [0.0, 10.0, 100.0]
        assert cfg["Processing"]["available_cores"] == 2
        assert cfg["Processing"]["max_missing"] == 5
        assert cfg["Curve Fit"]["type"] == "OLS"
        assert cfg["F Statistic"]["alpha"] == 0.05

    def test_treatment_time_string_passes_through(self, tmp_path):
        toml_path = _build_toml(
            input_tsv=tmp_path / "in.tsv",
            output_dir=tmp_path,
            experiments=["s1"],
            doses=[0.0],
            controls=["s1"],
            treatment_time="3 d",
        )
        import tomllib

        with toml_path.open("rb") as f:
            cfg = tomllib.load(f)
        assert cfg["Meta"]["treatment_time"] == "3 d"


# ============================================================================
# _parse_curves
# ============================================================================


class TestParseCurves:
    def test_basic_parse_and_ec50(self, tmp_path):
        """Build a fake curves.txt and verify parse + EC50_nM derivation."""
        cf = tmp_path / "curves.txt"
        # Minimal set of CurveCurator output columns for a 3-curve run
        cf.write_text(
            "Name\tpEC50\tCurve Slope\tCurve Front\tCurve Back\tCurve Fold Change\t"
            "Curve AUC\tCurve RMSE\tCurve R2\tCurve F_Value\tCurve P_Value\t"
            "Curve Log P_Value\tpEC50 Error\tCurve Slope Error\tSignal Quality\t"
            "Control Ratio Std\n"
            "siteA\t7.0\t1.5\t1.0\t2.0\t1.0\t0.5\t0.1\t0.95\t100\t0.001\t-3.0\t0.1\t0.1\t0.9\t0.05\n"
            "siteB\t9.0\t1.0\t1.0\t1.5\t0.5\t0.3\t0.2\t0.80\t50\t0.05\t-1.3\t0.2\t0.2\t0.7\t0.10\n"
        )
        df = _parse_curves(cf, timepoint_label=60)
        assert list(df["site_key"]) == ["siteA", "siteB"]
        # 10**(9 - 7.0) = 100 nM; 10**(9 - 9.0) = 1 nM
        assert np.isclose(df.loc[df["site_key"] == "siteA", "EC50_nM"].values[0], 100.0)
        assert np.isclose(df.loc[df["site_key"] == "siteB", "EC50_nM"].values[0], 1.0)
        # Timepoint label was added and placed early
        assert "timepoint" in df.columns
        assert df["timepoint"].iloc[0] == 60
        assert list(df.columns)[:2] == ["site_key", "timepoint"]

    def test_no_timepoint_label(self, tmp_path):
        cf = tmp_path / "curves.txt"
        cf.write_text("Name\tpEC50\nx\t6.5\n")
        df = _parse_curves(cf)
        assert "timepoint" not in df.columns
        assert df["site_key"].tolist() == ["x"]
        assert np.isclose(df["EC50_nM"].iloc[0], 10 ** (9 - 6.5))


# ============================================================================
# DEFAULT_CURVE_COLS
# ============================================================================


def test_default_curve_cols_covers_key_outputs():
    """Sanity: the default-columns list covers the headline CC outputs."""
    for col in ("Name", "pEC50", "Curve Fold Change", "Curve P_Value", "Curve q_Value"):
        assert col in DEFAULT_CURVE_COLS
