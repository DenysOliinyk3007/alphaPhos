"""Settings validation + engine-wiring tests for :func:`alphaphos.collapse_sites`.

End-to-end behavior (var/obs/uns population, key format, localization
strategies, aggregation methods, non-phospho drop) is covered by the
spike-in integration matrix in ``tests/integration/test_collapse_spiked.py``.
This file focuses on the pure-code paths those integration tests don't
touch: settings validation, ``resolve_settings``, quantification-level
dispatch + fallback, and top-N attribution wiring.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import alphaphos as ap

# ---------------------------------------------------------------------------
# Fixture: minimal synthetic Spectronaut-style PSM DataFrame
# ---------------------------------------------------------------------------


def _make_synthetic_psm() -> pd.DataFrame:
    """Two conditions x 2 reps, 3 phospho sites + 1 non-phospho row per sample."""
    rows = []
    for sample in ["ctrl_1", "ctrl_2", "trt_1", "trt_2"]:
        # AKT1 precursor with 2 phospho positions.
        rows.append(
            {
                "R.FileName": sample,
                "EG.PrecursorId": "_S[Phospho (STY)]TVQVAVSAGKT[Phospho (STY)]YHR_.2",
                "EG.TotalQuantity (Settings)": 1000 + hash(sample) % 500,
                "PEP.PeptidePosition": "473",
                "EG.PTMAssayProbability": 0.90,
                "EG.PTMLocalizationProbabilities": (
                    "_S[Phospho (STY): 95.0%]TVQVAVSAGKT[Phospho (STY): 88.0%]YHR_"
                ),
                "PG.Genes": "AKT1",
                "PG.ProteinGroups": "P31749",
            }
        )
        if sample in ("ctrl_1", "trt_1"):
            rows.append(
                {
                    "R.FileName": sample,
                    "EG.PrecursorId": "_ASGQGS[Phospho (STY)]PGVK_.2",
                    "EG.TotalQuantity (Settings)": 2000 + hash(sample) % 500,
                    "PEP.PeptidePosition": "570",
                    "EG.PTMAssayProbability": 0.92,
                    "EG.PTMLocalizationProbabilities": (
                        "_ASGQGS[Phospho (STY): 92.0%]PGVK_"
                    ),
                    "PG.Genes": "SIK1B",
                    "PG.ProteinGroups": "A0A0B4J2F2",
                }
            )
        rows.append(
            {
                "R.FileName": sample,
                "EG.PrecursorId": "_LMNVT[Phospho (STY)]PVLK_.2",
                "EG.TotalQuantity (Settings)": 500 + hash(sample) % 200,
                "PEP.PeptidePosition": "584",
                "EG.PTMAssayProbability": 0.80,
                "EG.PTMLocalizationProbabilities": ("_LMNVT[Phospho (STY): 80.0%]PVLK_"),
                "PG.Genes": "SIK1B",
                "PG.ProteinGroups": "A0A0B4J2F2",
            }
        )
        rows.append(
            {
                "R.FileName": sample,
                "EG.PrecursorId": "_ACTIN[Carbamidomethyl (C)]K_.2",
                "EG.TotalQuantity (Settings)": 999,
                "PEP.PeptidePosition": "100",
                "EG.PTMAssayProbability": 1.0,
                "EG.PTMLocalizationProbabilities": "_ACTINCK_",
                "PG.Genes": "ACT",
                "PG.ProteinGroups": "PACT",
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["source_path"] = "synthetic.parquet"
    df.attrs["n_rows_loaded"] = len(df)
    return df


def _make_synthetic_conditions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": ["ctrl_1", "ctrl_2", "trt_1", "trt_2"],
            "condition": ["ctrl", "ctrl", "trt", "trt"],
        }
    )


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("advanced", "exc_type", "match"),
    [
        ({"agregation": "sum"}, ValueError, "Unknown keys"),
        ({"search_engine": "Fragpipe"}, NotImplementedError, "search_engine"),
        ({"search_engine": "Mascot"}, ValueError, "search_engine must be one of"),
        ({"aggregation_method": "geomean"}, ValueError, "aggregation_method"),
    ],
)
def test_bad_settings_raise(advanced, exc_type, match):
    psm_df = _make_synthetic_psm()
    cdf = _make_synthetic_conditions()
    with pytest.raises(exc_type, match=match):
        ap.collapse_sites(psm_df, condition_df=cdf, advanced=advanced)


def test_condition_strategy_requires_condition_df():
    # This one is separate: the missing arg is condition_df, not an
    # entry in `advanced`, so it doesn't parametrize cleanly with the rest.
    psm_df = _make_synthetic_psm()
    with pytest.raises(ValueError, match="requires condition_df"):
        ap.collapse_sites(psm_df, condition_df=None)


# ---------------------------------------------------------------------------
# resolve_settings direct test
# ---------------------------------------------------------------------------


class TestResolveSettings:
    def test_defaults(self):
        from alphaphos import DEFAULT_COLLAPSE_SETTINGS, resolve_settings

        s = resolve_settings(None)
        assert s == DEFAULT_COLLAPSE_SETTINGS
        assert s is not DEFAULT_COLLAPSE_SETTINGS

    def test_partial_override(self):
        from alphaphos import resolve_settings

        s = resolve_settings({"cutoff": 0.65, "aggregation_method": "median"})
        assert s["cutoff"] == 0.65
        assert s["aggregation_method"] == "median"
        assert s["localization_strategy"] == "condition"


# ---------------------------------------------------------------------------
# quantification_level + fallback + top_n_attribution wiring
# ---------------------------------------------------------------------------


class TestQuantificationLevelWiring:
    @pytest.mark.parametrize("requested_level", ["MS2", "MS1"])
    def test_fallback_to_auto_and_warns(self, requested_level):
        # Fixture only has EG.TotalQuantity (Settings): the reader falls back
        # from MS2 or MS1 to 'auto' and warns.
        psm_df = _make_synthetic_psm()
        cdf = _make_synthetic_conditions()
        with pytest.warns(UserWarning, match=f"quantification_level='{requested_level}' unavailable"):
            adata = ap.collapse_sites(
                psm_df,
                condition_df=cdf,
                advanced={
                    "quantification_level": requested_level,
                    "localization_strategy": "per_run",
                },
            )
        stats = adata.uns["alphaphos"]["stats"]
        assert stats["quantification_level_used"] == "auto"
        assert stats["quantification_column_used"] == "EG.TotalQuantity (Settings)"

    def test_auto_no_fallback_no_warning(self):
        psm_df = _make_synthetic_psm()
        cdf = _make_synthetic_conditions()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            adata = ap.collapse_sites(
                psm_df,
                condition_df=cdf,
                advanced={"quantification_level": "auto", "localization_strategy": "per_run"},
            )
        assert adata.uns["alphaphos"]["stats"]["quantification_level_used"] == "auto"

    def test_ms2_direct_hit_when_ms2_column_present(self):
        psm_df = _make_synthetic_psm()
        cdf = _make_synthetic_conditions()
        psm_df["FG.MS2Quantity"] = psm_df["EG.TotalQuantity (Settings)"] * 2.0
        adata = ap.collapse_sites(
            psm_df,
            condition_df=cdf,
            advanced={"quantification_level": "MS2", "localization_strategy": "per_run"},
        )
        stats = adata.uns["alphaphos"]["stats"]
        assert stats["quantification_column_used"] == "FG.MS2Quantity"
        assert stats["quantification_level_used"] == "MS2"

    def test_invalid_quantification_level_raises(self):
        psm_df = _make_synthetic_psm()
        cdf = _make_synthetic_conditions()
        with pytest.raises(ValueError, match="quantification_level must be one of"):
            ap.collapse_sites(psm_df, condition_df=cdf, advanced={"quantification_level": "MS3"})


class TestTopNAttributionWiring:
    def test_disable_skips_filter(self):
        psm_df = _make_synthetic_psm()
        cdf = _make_synthetic_conditions()
        adata_on = ap.collapse_sites(
            psm_df,
            condition_df=cdf,
            advanced={"localization_strategy": "per_run", "top_n_attribution": True},
        )
        adata_off = ap.collapse_sites(
            psm_df,
            condition_df=cdf,
            advanced={"localization_strategy": "per_run", "top_n_attribution": False},
        )
        n_on = adata_on.uns["alphaphos"]["stats"]["n_psms_after_top_n"]
        n_off = adata_off.uns["alphaphos"]["stats"]["n_psms_after_top_n"]
        assert n_off >= n_on

    def test_bad_top_n_value_raises(self):
        psm_df = _make_synthetic_psm()
        cdf = _make_synthetic_conditions()
        with pytest.raises(ValueError, match="top_n_attribution must be bool"):
            ap.collapse_sites(psm_df, condition_df=cdf, advanced={"top_n_attribution": "yes"})
