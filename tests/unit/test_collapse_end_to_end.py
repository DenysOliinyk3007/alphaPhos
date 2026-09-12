"""Settings validation + engine-wiring tests for :func:`alphaphos.collapse_sites`.

End-to-end behavior (var/obs/uns population, key format, localization
strategies, aggregation methods, non-phospho drop) is covered by the
spike-in integration matrix in ``tests/integration/test_collapse_spiked.py``.
This file focuses on the pure-code paths those integration tests don't
touch: settings validation, ``resolve_settings``, quantification-level
dispatch + fallback, and top-N attribution wiring.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
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
                    "EG.PTMLocalizationProbabilities": ("_ASGQGS[Phospho (STY): 92.0%]PGVK_"),
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
        with pytest.warns(
            UserWarning, match=f"quantification_level='{requested_level}' unavailable"
        ):
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

    def test_default_none_resolves_to_ms2_for_spectronaut(self):
        psm_df = _make_synthetic_psm()
        cdf = _make_synthetic_conditions()
        # Fixture has no MS2 column -> the engine default "MS2" is what warns.
        with pytest.warns(UserWarning, match="quantification_level='MS2' unavailable"):
            adata = ap.collapse_sites(
                psm_df, condition_df=cdf, advanced={"localization_strategy": "per_run"}
            )
        stats = adata.uns["alphaphos"]["stats"]
        assert stats["quantification_level_requested"] is None
        assert stats["quantification_level_used"] == "auto"

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
        with pytest.raises(ValueError, match="top_n_attribution must be bool or 'auto'"):
            ap.collapse_sites(psm_df, condition_df=cdf, advanced={"top_n_attribution": "yes"})


# ---------------------------------------------------------------------------
# Contaminant provenance in the collapse output
# ---------------------------------------------------------------------------


def _make_psm_with_contam_variants(
    sample_pg: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Two-sample PSM where each site's protein group can be overridden.

    ``sample_pg`` maps ``"gene"`` -> ``(protein_groups_string, gene_symbol)``.
    Same phospho precursor per site for simplicity.
    """
    rows = []
    for sample in ("s1", "s2"):
        for i, (_gene, (pg, gene_symbol)) in enumerate(sample_pg.items()):
            rows.append(
                {
                    "R.FileName": sample,
                    "EG.PrecursorId": f"_S[Phospho (STY)]{'A' * (i + 3)}_.2",
                    "EG.TotalQuantity (Settings)": 1000 + i * 100,
                    "PEP.PeptidePosition": str(100 + i * 10),
                    "EG.PTMAssayProbability": 0.95,
                    "EG.PTMLocalizationProbabilities": (
                        f"_S[Phospho (STY): 95.0%]{'A' * (i + 3)}_"
                    ),
                    "PG.Genes": gene_symbol,
                    "PG.ProteinGroups": pg,
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["source_path"] = "synthetic.parquet"
    df.attrs["n_rows_loaded"] = len(df)
    return df


def _cdf():
    return pd.DataFrame({"sample": ["s1", "s2"], "condition": ["a", "b"]})


class TestContaminantSurfacing:
    """Sites with any Cont_-tagged protein must announce that in the site key
    and in ``adata.var['is_contaminant_match']``, regardless of Spectronaut's
    within-PG ordering.
    """

    def test_clean_pg_has_no_contam_flag(self):
        psm = _make_psm_with_contam_variants(
            {"CLEAN": ("P12345", "AKT1")},
        )
        adata = ap.collapse_sites(psm, condition_df=_cdf())
        assert "is_contaminant_match" in adata.var.columns
        assert not adata.var["is_contaminant_match"].any()
        # Key uses the untouched accession
        assert any(k.startswith("P12345|AKT1") for k in adata.var_names)

    def test_contam_first_key_shows_prefix_and_flag_set(self):
        # Spectronaut wrote the contaminant tag first (cardio-observed order).
        psm = _make_psm_with_contam_variants(
            {"KRTLIKE": ("Cont_P05783;P05783", "KRT18")},
        )
        adata = ap.collapse_sites(psm, condition_df=_cdf())
        # Key preserves the Cont_ marker
        assert any(k.startswith("Cont_P05783|KRT18") for k in adata.var_names)
        # Flag is set
        assert bool(adata.var["is_contaminant_match"].all())

    def test_contam_second_still_surfaces_prefix_in_key(self):
        # Spectronaut wrote the untagged accession first -- WITHOUT the fix
        # the site key would silently drop the Cont_ marker.  Our fix picks
        # the contaminant-tagged variant regardless of order.
        psm = _make_psm_with_contam_variants(
            {"AMBIG": ("P05783;Cont_P05783", "KRT18")},
        )
        adata = ap.collapse_sites(psm, condition_df=_cdf())
        # The Cont_ marker must still appear in the key
        assert any(k.startswith("Cont_P05783|KRT18") for k in adata.var_names), (
            f"Cont_ marker lost from site key; got: {list(adata.var_names)}"
        )
        assert bool(adata.var["is_contaminant_match"].all())

    def test_flag_targets_only_contam_rows(self):
        # Two sites: one clean, one ambiguous.  Only the ambiguous one is flagged.
        psm = _make_psm_with_contam_variants(
            {
                "CLEAN": ("P12345", "AKT1"),
                "AMBIG": ("P05783;Cont_P05783", "KRT18"),
            }
        )
        adata = ap.collapse_sites(psm, condition_df=_cdf())
        flagged_keys = adata.var_names[adata.var["is_contaminant_match"].fillna(False)]
        assert len(flagged_keys) >= 1
        for k in flagged_keys:
            assert k.startswith("Cont_"), f"flagged key {k!r} missing Cont_ marker"
        unflagged_keys = adata.var_names[~adata.var["is_contaminant_match"].fillna(False)]
        for k in unflagged_keys:
            assert not k.startswith(("Cont_", "CON__", "contam_")), (
                f"unflagged key {k!r} unexpectedly contains contam prefix"
            )


# ---------------------------------------------------------------------------
# Regressions from the preprocess code review -- none of these had a test.
# ---------------------------------------------------------------------------


class TestReviewRegressions:
    def test_short_key_collisions_are_h5ad_serialisable(self, tmp_path):
        psm = _make_synthetic_psm()
        # A second peptide (missed cleavage, extra K) covering the same AKT1
        # residues under a second protein group -> two full keys share one
        # short key -> a collision entry in uns.  (Merely changing the PG of
        # the same precursor would not do: the pivot is keyed by precursor.)
        dup = psm[psm["PG.Genes"] == "AKT1"].copy()
        dup["PG.ProteinGroups"] = "P31749-2"
        dup["EG.PrecursorId"] = "_S[Phospho (STY)]TVQVAVSAGKT[Phospho (STY)]YHRK_.2"
        dup["EG.PTMLocalizationProbabilities"] = (
            "_S[Phospho (STY): 95.0%]TVQVAVSAGKT[Phospho (STY): 88.0%]YHRK_"
        )
        adata = ap.collapse_sites(
            pd.concat([psm, dup], ignore_index=True),
            condition_df=_make_synthetic_conditions(),
            advanced={"localization_strategy": "per_run"},
        )
        coll = adata.uns["alphaphos"]["short_key_collisions"]
        assert coll and isinstance(coll, dict)
        assert all(isinstance(v, list) for v in coll.values())
        adata.write_h5ad(tmp_path / "coll.h5ad")  # list-of-tuples used to raise here

    def test_gene_underscore_survives_into_keys(self):
        psm = _make_synthetic_psm()
        psm["PG.Genes"] = psm["PG.Genes"].str.replace("AKT1", "AKT_1", regex=False)
        adata = ap.collapse_sites(
            psm,
            condition_df=_make_synthetic_conditions(),
            advanced={"localization_strategy": "per_run"},
        )
        assert any(k.startswith("P31749|AKT_1|") for k in adata.var_names)
        assert not adata.var_names.str.contains("#", regex=False).any()
        assert not adata.var["short_key"].str.contains("#", regex=False).any()

    @pytest.mark.parametrize("strategy", ["per_run", "condition"])
    def test_drop_all_nan_false_keeps_matrices_aligned(self, strategy):
        adata = ap.collapse_sites(
            _make_synthetic_psm(),
            condition_df=_make_synthetic_conditions(),
            advanced={"drop_all_nan": False, "localization_strategy": strategy},
        )
        assert adata.n_vars >= 3
        assert adata.var_names.is_unique

    def test_duplicated_condition_row_does_not_break_obs(self):
        cdf = _make_synthetic_conditions()
        cdf_dup = pd.concat([cdf, cdf.iloc[[0]]], ignore_index=True)
        adata = ap.collapse_sites(_make_synthetic_psm(), condition_df=cdf_dup)
        assert adata.n_obs == 4
        assert adata.obs.loc["ctrl_1", "condition"] == "ctrl"

    def test_integer_sample_ids_in_condition_df_still_join(self):
        psm = _make_synthetic_psm()
        psm["R.FileName"] = psm["R.FileName"].map(
            {"ctrl_1": "1", "ctrl_2": "2", "trt_1": "3", "trt_2": "4"}
        )
        cdf = pd.DataFrame({"sample": [1, 2, 3, 4], "condition": ["ctrl", "ctrl", "trt", "trt"]})
        adata = ap.collapse_sites(psm, condition_df=cdf)
        assert adata.obs["condition"].notna().all()

    def test_unlabelled_sample_gets_strict_per_run_mask(self, caplog):
        psm = _make_synthetic_psm()
        # Make the LMNVT site poorly localised in trt_2 only (40% < 0.75).
        lmnvt_trt2 = (psm["R.FileName"] == "trt_2") & psm["EG.PrecursorId"].str.contains("LMNVT")
        psm.loc[lmnvt_trt2, "EG.PTMLocalizationProbabilities"] = "_LMNVT[Phospho (STY): 40.0%]PVLK_"
        cdf = _make_synthetic_conditions()

        # Labelled: trt_1 is Class-I, so the condition rule keeps trt_2's cell.
        labelled = ap.collapse_sites(psm, condition_df=cdf)
        site = [k for k in labelled.var_names if "|T588|" in k][0]
        assert np.isfinite(labelled["trt_2", site].X[0, 0])

        # Unlabelled: trt_2 must fall back to the strict per-run rule -> masked.
        with caplog.at_level(logging.WARNING):
            unlabelled = ap.collapse_sites(psm, condition_df=cdf[cdf["sample"] != "trt_2"])
        assert any("no entry in condition_df" in r.message for r in caplog.records)
        assert np.isnan(unlabelled["trt_2", site].X[0, 0])

    def test_logger_level_is_not_clobbered(self):
        lg = logging.getLogger("alphaphos.preprocess.collapse")
        old = lg.level
        try:
            lg.setLevel(logging.DEBUG)
            ap.collapse_sites(_make_synthetic_psm(), condition_df=_make_synthetic_conditions())
            assert lg.level == logging.DEBUG
        finally:
            lg.setLevel(old)

    def test_collapse_level_P_is_removed(self):
        with pytest.raises(ValueError, match="collapse_level must be 'PG'"):
            ap.collapse_sites(
                _make_synthetic_psm(),
                condition_df=_make_synthetic_conditions(),
                advanced={"collapse_level": "P"},
            )

    @pytest.mark.parametrize(
        "advanced",
        [
            {"cutoff": "0.5"},
            {"classI_cutoff": True},
            {"condition_threshold": 0},
            {"wilson_threshold": True},
            {"wilson_threshold": [0.5]},
        ],
    )
    def test_non_numeric_or_bool_thresholds_are_rejected(self, advanced):
        with pytest.raises(ValueError):
            ap.resolve_settings(advanced)

    def test_source_attrs_captured_before_frame_is_rewritten(self):
        adata = ap.collapse_sites(_make_synthetic_psm(), condition_df=_make_synthetic_conditions())
        assert adata.uns["source_attrs"]["source_path"] == "synthetic.parquet"


class TestDuplicateRunHeuristic:
    """The duplicate-run warning must key on quantities, not just precursor ids
    (deep DIA runs of the same sample type share their alphabetically-first ids)."""

    def _collapse(self, psm, caplog):
        with caplog.at_level(logging.WARNING):
            ap.collapse_sites(psm, condition_df=_make_synthetic_conditions())
        return [r.message for r in caplog.records if "duplicated run files" in r.message]

    def test_same_ids_different_quant_no_warning(self, caplog):
        # ctrl_1 and ctrl_2 carry identical precursor ids but different intensities.
        assert self._collapse(_make_synthetic_psm(), caplog) == []

    def test_identical_quant_triggers_warning(self, caplog):
        psm = _make_synthetic_psm()
        src = psm[psm["R.FileName"] == "ctrl_1"].copy()
        src["R.FileName"] = "ctrl_1_copy"
        psm = pd.concat([psm[psm["R.FileName"] != "ctrl_2"], src], ignore_index=True)
        cdf = _make_synthetic_conditions().replace({"ctrl_2": "ctrl_1_copy"})
        with caplog.at_level(logging.WARNING):
            ap.collapse_sites(psm, condition_df=cdf)
        msgs = [r.message for r in caplog.records if "duplicated run files" in r.message]
        assert msgs and "ctrl_1_copy" in msgs[0]


# ---------------------------------------------------------------------------
# precursor_loc_gate: Spectronaut-style per-precursor Class-I gate
# ---------------------------------------------------------------------------


class TestPrecursorLocGate:
    """A precursor whose phospho is NOT confidently localized to a site in a run
    must not add its intensity to that site when a Class-I precursor exists;
    when no precursor is Class-I, everything is summed and the site-level mask
    decides (keeps the ``condition`` strategy's recovery path)."""

    def _psm_with_ambiguous_second_precursor(self):
        psm = _make_synthetic_psm()
        # Missed-cleavage variant of the AKT1 peptide (extra K) covering the
        # same S473 / T484 sites, but with S473 only 30% localized.
        extra = {
            "R.FileName": "ctrl_1",
            "EG.PrecursorId": "_S[Phospho (STY)]TVQVAVSAGKT[Phospho (STY)]YHRK_.2",
            "EG.TotalQuantity (Settings)": 5000.0,
            "PEP.PeptidePosition": "473",
            "EG.PTMAssayProbability": 0.70,
            "EG.PTMLocalizationProbabilities": (
                "_S[Phospho (STY): 30.0%]TVQVAVSAGKT[Phospho (STY): 70.0%]YHRK_"
            ),
            "PG.Genes": "AKT1",
            "PG.ProteinGroups": "P31749",
        }
        qa = float(
            psm.loc[
                (psm["R.FileName"] == "ctrl_1") & psm["EG.PrecursorId"].str.endswith("YHR_.2"),
                "EG.TotalQuantity (Settings)",
            ].iloc[0]
        )
        return pd.concat([psm, pd.DataFrame([extra])], ignore_index=True), qa

    def _cell(self, adata, site_substr, sample):
        site = [k for k in adata.var_names if site_substr in k][0]
        return float(adata[sample, site].X[0, 0])

    def test_gate_on_excludes_ambiguous_precursor(self):
        psm, qa = self._psm_with_ambiguous_second_precursor()
        adata = ap.collapse_sites(
            psm,
            condition_df=_make_synthetic_conditions(),
            advanced={"localization_strategy": "per_run"},  # gate ON by default
        )
        # S473: only the 95%-localized precursor contributes.
        assert self._cell(adata, "|S473|M2", "ctrl_1") == pytest.approx(np.log2(qa), abs=1e-4)
        assert adata.uns["alphaphos"]["stats"]["n_precursor_cells_gated"] >= 1
        assert adata.uns["alphaphos"]["pipeline_params"]["precursor_loc_gate"] is True

    def test_gate_off_sums_everything(self):
        psm, qa = self._psm_with_ambiguous_second_precursor()
        adata = ap.collapse_sites(
            psm,
            condition_df=_make_synthetic_conditions(),
            advanced={"localization_strategy": "per_run", "precursor_loc_gate": False},
        )
        assert self._cell(adata, "|S473|M2", "ctrl_1") == pytest.approx(
            np.log2(qa + 5000.0), abs=1e-4
        )
        assert adata.uns["alphaphos"]["stats"]["n_precursor_cells_gated"] == 0

    def test_gate_falls_back_to_all_when_no_precursor_is_class_i(self):
        from alphaphos.preprocess._collapse.site_pipeline import aggregate_precursors_to_sites

        idx = pd.MultiIndex.from_tuples(
            [("P1", 1), ("P2", 1)], names=["PTM_group", "PTM_0_pos_val"]
        )
        quant = pd.DataFrame({"s1": [100.0, 300.0], "s2": [100.0, 300.0]}, index=idx)
        # s1: P1 is Class-I, P2 is not -> only P1.   s2: neither -> sum both.
        loc = pd.DataFrame({"s1": [0.90, 0.30], "s2": [0.40, 0.20]}, index=idx)
        meta = pd.DataFrame({"full_key": ["K", "K"], "gene": ["G", "G"]}, index=idx)
        q, site_loc, _ = aggregate_precursors_to_sites(
            quant, loc, meta, aggregation_method="sum", precursor_loc_gate=0.75
        )
        assert q.loc["K", "s1"] == 100.0
        assert q.loc["K", "s2"] == 400.0
        assert site_loc.loc["K", "s1"] == 0.90  # loc matrix is still the max over precursors
        assert q.attrs["n_precursor_cells_gated"] == 1

    def test_gate_must_be_bool(self):
        with pytest.raises(ValueError, match="precursor_loc_gate must be bool"):
            ap.resolve_settings({"precursor_loc_gate": "yes"})


class TestTopNAuto:
    def test_auto_applies_for_spectronaut(self):
        adata = ap.collapse_sites(_make_synthetic_psm(), condition_df=_make_synthetic_conditions())
        assert adata.uns["alphaphos"]["stats"]["top_n_attribution_applied"] is True
        assert adata.uns["alphaphos"]["pipeline_params"]["top_n_attribution"] == "auto"

    def test_explicit_false_overrides_auto(self):
        adata = ap.collapse_sites(
            _make_synthetic_psm(),
            condition_df=_make_synthetic_conditions(),
            advanced={"top_n_attribution": False},
        )
        assert adata.uns["alphaphos"]["stats"]["top_n_attribution_applied"] is False
