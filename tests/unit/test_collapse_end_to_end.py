"""End-to-end integration test for :func:`alphaphos.collapse_sites`.

Uses a minimal synthetic PSM DataFrame to exercise every stage without
depending on real Spectronaut output. Verifies:

* Return type + shape (AnnData; samples x sites).
* Metadata population (`.var` columns, `.obs` columns, `.uns` provenance).
* Settings validation (unknown keys / unknown search engine / bad values).
* Delimiter migration (keys use ``|``, gene has no underscore-to-hash leak).
* Selectivity ends up in `.obs`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alphaphos as ap


# ---------------------------------------------------------------------------
# Fixture: minimal synthetic Spectronaut-style PSM DataFrame
# ---------------------------------------------------------------------------


def _make_synthetic_psm() -> pd.DataFrame:
    """Two conditions x 2 reps x 4 phospho sites, plus 2 non-phospho rows.

    Layout::

        Site 1 (AKT1_S473_M1): all 4 samples,  loc = 0.95+
        Site 2 (AKT1_T450_M1): all 4 samples,  loc = 0.85+
        Site 3 (SIK1B_S575_M1): only 2 samples, loc = 0.90
        Site 4 (SIK1B_T588_M1): 4 samples, some with lower loc
        + non-phospho control rows (to test they get dropped)
    """
    rows = []
    for sample in ["ctrl_1", "ctrl_2", "trt_1", "trt_2"]:
        # AKT1 sites -- one precursor with 2 phospho positions.
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
        # SIK1B site 3 (only 2 samples get this)
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
        # SIK1B site 4 (all 4 samples)
        rows.append(
            {
                "R.FileName": sample,
                "EG.PrecursorId": "_LMNVT[Phospho (STY)]PVLK_.2",
                "EG.TotalQuantity (Settings)": 500 + hash(sample) % 200,
                "PEP.PeptidePosition": "584",
                "EG.PTMAssayProbability": 0.80,
                "EG.PTMLocalizationProbabilities": (
                    "_LMNVT[Phospho (STY): 80.0%]PVLK_"
                ),
                "PG.Genes": "SIK1B",
                "PG.ProteinGroups": "A0A0B4J2F2",
            }
        )
        # Non-phospho row -- should not appear in output
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
# Happy path
# ---------------------------------------------------------------------------


@pytest.fixture
def collapsed():
    psm_df = _make_synthetic_psm()
    cdf = _make_synthetic_conditions()
    # Use per_run strategy to avoid the condition-aware majority masking
    # (which needs more replicates to be interesting on our small fixture).
    return ap.collapse_sites(
        psm_df,
        condition_df=cdf,
        advanced={"localization_strategy": "per_run", "cutoff": 0.75},
    )


class TestCollapseSitesEndToEnd:
    def test_returns_anndata(self, collapsed):
        import anndata as ad
        assert isinstance(collapsed, ad.AnnData)

    def test_shape_samples_x_sites(self, collapsed):
        # 4 samples, sites: AKT1_S473_M2, AKT1_T450_M2, SIK1B_S575_M1, SIK1B_T588_M1
        # (AKT1 precursor has 2 phospho -> multiplicity 2; explodes to 2 sites)
        assert collapsed.n_obs == 4
        assert collapsed.n_vars >= 3  # exact count depends on masking

    def test_var_columns_present(self, collapsed):
        expected = {
            "short_key",
            "pg_key",
            "protein_group_id",
            "gene",
            "site_aa",
            "site_position",
            "multiplicity",
            "UPD_seq",
            "n_samples_detected",
            "mean_loc_prob",
            "max_loc_prob",
            "min_loc_prob",
            "n_classI_samples",
            "fraction_classI",
        }
        assert expected.issubset(set(collapsed.var.columns)), (
            f"Missing var columns: {expected - set(collapsed.var.columns)}"
        )

    def test_index_uses_pipe_delimiter(self, collapsed):
        for key in collapsed.var.index:
            parts = key.split("|")
            assert len(parts) == 4, f"expected 4 |-fields, got {parts} in {key}"

    def test_short_key_no_pipes_after_gene(self, collapsed):
        # short_key format is Gene|Site|Mult -- 3 pipe-separated fields.
        for key in collapsed.var["short_key"]:
            base = key.split("#")[0]  # strip any collision suffix
            parts = base.split("|")
            assert len(parts) == 3, f"expected 3 |-fields in short_key, got {parts}"

    def test_obs_carries_conditions(self, collapsed):
        assert "condition" in collapsed.obs.columns
        assert set(collapsed.obs["condition"].dropna()) == {"ctrl", "trt"}

    def test_selectivity_in_obs(self, collapsed):
        assert "phospho_selectivity_pct" in collapsed.obs.columns
        # Fixture has 3 phospho precursor patterns per sample + 1 non-phospho
        # -> selectivity ~ 75% (varies slightly with dedupe / hashing).
        assert collapsed.obs["phospho_selectivity_pct"].between(50, 100).all()

    def test_uns_provenance(self, collapsed):
        assert "alphaphos" in collapsed.uns
        params = collapsed.uns["alphaphos"]["pipeline_params"]
        assert params["localization_strategy"] == "per_run"
        assert params["cutoff"] == 0.75
        assert collapsed.uns["alphaphos"]["version"] == "0.0.0"
        assert "stats" in collapsed.uns["alphaphos"]

    def test_source_attrs_forwarded(self, collapsed):
        assert collapsed.uns["source_attrs"]["source_path"] == "synthetic.parquet"

    def test_localization_layer(self, collapsed):
        assert "localization" in collapsed.layers
        assert collapsed.layers["localization"].shape == collapsed.X.shape

    def test_non_phospho_dropped(self, collapsed):
        # ACT_?_? shouldn't appear anywhere
        assert not (collapsed.var["gene"] == "ACT").any()


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


class TestSettingsValidation:
    def test_unknown_advanced_key_raises(self):
        psm_df = _make_synthetic_psm()
        cdf = _make_synthetic_conditions()
        with pytest.raises(ValueError, match="Unknown keys"):
            ap.collapse_sites(psm_df, condition_df=cdf, advanced={"agregation": "sum"})

    def test_unknown_search_engine_raises(self):
        psm_df = _make_synthetic_psm()
        cdf = _make_synthetic_conditions()
        with pytest.raises(NotImplementedError, match="search_engine"):
            ap.collapse_sites(
                psm_df, condition_df=cdf,
                advanced={"search_engine": "Diann"},
            )

    def test_invalid_search_engine_value_raises(self):
        psm_df = _make_synthetic_psm()
        cdf = _make_synthetic_conditions()
        with pytest.raises(ValueError, match="search_engine must be one of"):
            ap.collapse_sites(
                psm_df, condition_df=cdf,
                advanced={"search_engine": "Mascot"},
            )

    def test_invalid_aggregation_raises(self):
        psm_df = _make_synthetic_psm()
        cdf = _make_synthetic_conditions()
        with pytest.raises(ValueError, match="aggregation_method"):
            ap.collapse_sites(
                psm_df, condition_df=cdf,
                advanced={"aggregation_method": "geomean"},
            )

    def test_condition_strategy_requires_condition_df(self):
        psm_df = _make_synthetic_psm()
        with pytest.raises(ValueError, match="requires condition_df"):
            ap.collapse_sites(psm_df, condition_df=None)  # default strategy = 'condition'


# ---------------------------------------------------------------------------
# resolve_settings direct test
# ---------------------------------------------------------------------------


class TestResolveSettings:
    def test_defaults(self):
        from alphaphos import resolve_settings, DEFAULT_COLLAPSE_SETTINGS
        s = resolve_settings(None)
        assert s == DEFAULT_COLLAPSE_SETTINGS
        # Returned copy, not the same object
        assert s is not DEFAULT_COLLAPSE_SETTINGS

    def test_partial_override(self):
        from alphaphos import resolve_settings
        s = resolve_settings({"cutoff": 0.65, "aggregation_method": "median"})
        assert s["cutoff"] == 0.65
        assert s["aggregation_method"] == "median"
        assert s["localization_strategy"] == "condition"  # default preserved

    def test_type_error_on_non_dict(self):
        from alphaphos import resolve_settings
        with pytest.raises(TypeError):
            resolve_settings("not a dict")
