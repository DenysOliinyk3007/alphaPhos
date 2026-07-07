"""Tests for :func:`alphaphos.collapse_precursors` and
:func:`alphaphos.precursor_to_site_view`.

Precursor-level collapse is a sibling of :func:`collapse_sites` -- it
skips residue attribution and localization masking, keeping one row per
unique ``(protein, gene, peptide, charge, mods)`` precursor. These tests
lock the contract: the .var index format, the .obs / .uns contents, the
phospho-only filter, the noise-floor filter, the aggregation modes, the
localization annotation, and the bridge back to alphaPhos site keys.

Uses the same synthetic Spectronaut-style PSM DataFrame pattern as
``test_collapse_end_to_end.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alphaphos as ap
from alphaphos.preprocess.collapse_precursors import (
    _build_keys,
    _parse_precursors,
    aggregate_to_site_level,
    resolve_precursor_settings,
)

# ---------------------------------------------------------------------------
# Fixture: minimal synthetic Spectronaut-style PSM DataFrame
# ---------------------------------------------------------------------------


def _make_synthetic_psm() -> pd.DataFrame:
    """4 samples x 3 precursors, plus 1 non-phospho + 1 dup row for coverage.

    Uses real Spectronaut-style precursor IDs pulled from egf_mini.tsv-style
    examples so parsing is exercised end-to-end.
    """
    rows: list[dict] = []
    for sample in ("ctrl_1", "ctrl_2", "trt_1", "trt_2"):
        # AKT1 precursor with 2 phospho positions
        rows.append(
            {
                "R.FileName": sample,
                "EG.PrecursorId": "_S[Phospho (STY)]TVQVAVSAGKT[Phospho (STY)]YHR_.2",
                "EG.TotalQuantity (Settings)": 1000.0 + (hash(sample) % 500),
                "PEP.PeptidePosition": "473",
                "EG.PTMLocalizationProbabilities": (
                    "_S[Phospho (STY): 95.0%]TVQVAVSAGKT[Phospho (STY): 88.0%]YHR_"
                ),
                "PG.Genes": "AKT1",
                "PG.ProteinGroups": "P31749",
            }
        )
        # SIK1B present only in ctrl samples -> sparse coverage
        if sample in ("ctrl_1", "ctrl_2"):
            rows.append(
                {
                    "R.FileName": sample,
                    "EG.PrecursorId": "_ASGQGS[Phospho (STY)]PGVK_.2",
                    "EG.TotalQuantity (Settings)": 2000.0 + (hash(sample) % 500),
                    "PEP.PeptidePosition": "570",
                    "EG.PTMLocalizationProbabilities": ("_ASGQGS[Phospho (STY): 92.0%]PGVK_"),
                    "PG.Genes": "SIK1B",
                    "PG.ProteinGroups": "A0A0B4J2F2",
                }
            )
        # Non-phospho row -- must be dropped when phospho_only=True (default)
        rows.append(
            {
                "R.FileName": sample,
                "EG.PrecursorId": "_LMNVTPVLK_.2",
                "EG.TotalQuantity (Settings)": 500.0,
                "PEP.PeptidePosition": "42",
                "EG.PTMLocalizationProbabilities": "",
                "PG.Genes": "GENEX",
                "PG.ProteinGroups": "Q00000",
            }
        )
        # Duplicate row for AKT1 to test aggregation -- same precursor, same
        # sample, different intensity.  Sum-aggregation should combine them.
        rows.append(
            {
                "R.FileName": sample,
                "EG.PrecursorId": "_S[Phospho (STY)]TVQVAVSAGKT[Phospho (STY)]YHR_.2",
                "EG.TotalQuantity (Settings)": 300.0,
                "PEP.PeptidePosition": "473",
                "EG.PTMLocalizationProbabilities": (
                    "_S[Phospho (STY): 95.0%]TVQVAVSAGKT[Phospho (STY): 88.0%]YHR_"
                ),
                "PG.Genes": "AKT1",
                "PG.ProteinGroups": "P31749",
            }
        )
    return pd.DataFrame(rows)


def _make_conditions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": ["ctrl_1", "ctrl_2", "trt_1", "trt_2"],
            "condition": ["ctrl", "ctrl", "trt", "trt"],
        }
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestClassIGate:
    """Class I gate: drop precursors whose PEAK localization probability
    across all PSM rows was below ``classI_cutoff``."""

    def _make_psm_with_variable_loc(self) -> pd.DataFrame:
        """Two phospho precursors: one with high loc (99%), one with low (40%)."""
        rows = []
        for sample in ("s1", "s2", "s3", "s4"):
            # HIGH: 99% loc every run
            rows.append(
                {
                    "R.FileName": sample,
                    "EG.PrecursorId": "_S[Phospho (STY)]TDNAFENPFFK_.2",
                    "EG.TotalQuantity (Settings)": 1000.0,
                    "PEP.PeptidePosition": "100",
                    "EG.PTMLocalizationProbabilities": ("_S[Phospho (STY): 99.0%]TDNAFENPFFK_"),
                    "PG.Genes": "HIGH_GENE",
                    "PG.ProteinGroups": "P_HIGH",
                }
            )
            # LOW: 40% loc every run -- would drop at default cutoff 0.75
            rows.append(
                {
                    "R.FileName": sample,
                    "EG.PrecursorId": "_LMNVT[Phospho (STY)]PVLK_.2",
                    "EG.TotalQuantity (Settings)": 2000.0,
                    "PEP.PeptidePosition": "50",
                    "EG.PTMLocalizationProbabilities": ("_LMNVT[Phospho (STY): 40.0%]PVLK_"),
                    "PG.Genes": "LOW_GENE",
                    "PG.ProteinGroups": "P_LOW",
                }
            )
        return pd.DataFrame(rows)

    def test_default_cutoff_drops_low_loc_precursor(self):
        psm = self._make_psm_with_variable_loc()
        adata = ap.collapse_precursors(psm)
        # Default cutoff 0.75: LOW (40%) dropped, HIGH (99%) kept
        genes = adata.var["gene"].tolist()
        assert "HIGH_GENE" in genes
        assert "LOW_GENE" not in genes
        assert adata.uns["alphaphos"]["stats"]["n_dropped_classI"] == 1

    def test_none_disables_gate(self):
        psm = self._make_psm_with_variable_loc()
        adata = ap.collapse_precursors(psm, advanced={"classI_cutoff": None})
        genes = adata.var["gene"].tolist()
        assert "HIGH_GENE" in genes
        assert "LOW_GENE" in genes
        assert adata.uns["alphaphos"]["stats"]["n_dropped_classI"] == 0

    def test_lower_cutoff_keeps_more(self):
        psm = self._make_psm_with_variable_loc()
        adata = ap.collapse_precursors(psm, advanced={"classI_cutoff": 0.3})
        genes = adata.var["gene"].tolist()
        assert "HIGH_GENE" in genes
        assert "LOW_GENE" in genes

    def test_cutoff_boundary_inclusive(self):
        """A precursor at exactly the cutoff must be kept (>= comparison)."""
        psm = self._make_psm_with_variable_loc()
        # LOW is at 40%; cutoff of exactly 0.40 must keep it
        adata = ap.collapse_precursors(psm, advanced={"classI_cutoff": 0.40})
        assert "LOW_GENE" in adata.var["gene"].tolist()

    def test_non_phospho_bypasses_gate(self):
        """Non-phospho precursors have no meaningful localization -- they
        must pass through the gate untouched when ``phospho_only=False``."""
        psm = self._make_psm_with_variable_loc()
        # Append a non-phospho precursor row per sample
        for sample in ("s1", "s2", "s3", "s4"):
            psm = pd.concat(
                [
                    psm,
                    pd.DataFrame(
                        [
                            {
                                "R.FileName": sample,
                                "EG.PrecursorId": "_LMNVTPVLK_.2",
                                "EG.TotalQuantity (Settings)": 500.0,
                                "PEP.PeptidePosition": "42",
                                "EG.PTMLocalizationProbabilities": "",
                                "PG.Genes": "NONPHOS_GENE",
                                "PG.ProteinGroups": "P_NONPHOS",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        adata = ap.collapse_precursors(psm, advanced={"phospho_only": False, "classI_cutoff": 0.75})
        # Non-phospho precursor kept (loc gate doesn't apply);
        # low-loc phospho precursor dropped by the gate.
        genes = adata.var["gene"].tolist()
        assert "NONPHOS_GENE" in genes
        assert "HIGH_GENE" in genes
        assert "LOW_GENE" not in genes

    def test_all_dropped_raises(self):
        """If EVERY precursor gets gated out, we raise rather than return
        an empty AnnData -- that's almost always a config problem."""
        psm = self._make_psm_with_variable_loc()
        with pytest.raises(ValueError, match="No precursors survived"):
            ap.collapse_precursors(psm, advanced={"classI_cutoff": 0.9999})

    def test_cutoff_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            resolve_precursor_settings({"classI_cutoff": 1.5})

    def test_cutoff_bool_type_raises(self):
        # `True` is technically a valid float in [0, 1] but almost certainly a
        # user error (they meant to enable/disable the gate).
        with pytest.raises(ValueError, match="classI_cutoff must be a float"):
            resolve_precursor_settings({"classI_cutoff": True})

    def test_cutoff_without_annotate_localization_raises(self):
        with pytest.raises(ValueError, match="requires annotate_localization"):
            resolve_precursor_settings({"classI_cutoff": 0.75, "annotate_localization": False})


class TestResolveSettings:
    def test_defaults_returned_when_none(self):
        out = resolve_precursor_settings(None)
        assert out["search_engine"] == "SN"
        assert out["quantification_level"] == "MS2"
        assert out["phospho_only"] is True
        assert out["annotate_localization"] is True

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown advanced keys"):
            resolve_precursor_settings({"nonsense_flag": True})

    def test_bad_engine_raises(self):
        with pytest.raises(ValueError, match="search_engine must be"):
            resolve_precursor_settings({"search_engine": "MaxQuant"})

    def test_bad_level_raises(self):
        with pytest.raises(ValueError, match="quantification_level must be"):
            resolve_precursor_settings({"quantification_level": "MS3"})

    def test_bad_agg_method_raises(self):
        with pytest.raises(ValueError, match="aggregation_method must be"):
            resolve_precursor_settings({"aggregation_method": "product"})

    def test_bool_type_enforced(self):
        with pytest.raises(ValueError, match="phospho_only must be a bool"):
            resolve_precursor_settings({"phospho_only": "true"})


# ---------------------------------------------------------------------------
# Precursor parsing + key construction
# ---------------------------------------------------------------------------


class TestParsePrecursors:
    def test_extracts_peptide_charge_mods(self):
        ids = pd.Series(
            [
                "_S[Phospho (STY)]TDNAFENPFFK_.2",
                "_S[Phospho (STY)]SPNPFVGS[Phospho (STY)]PPK_.2",
                "_PRPVS[Phospho (STY)]PSSLLDTAISEEAR_.3",
            ]
        )
        parsed = _parse_precursors(ids)
        assert list(parsed["peptide_sequence"]) == [
            "STDNAFENPFFK",
            "SSPNPFVGSPPK",
            "PRPVSPSSLLDTAISEEAR",
        ]
        assert list(parsed["charge"]) == [2, 2, 3]
        assert list(parsed["n_phospho"]) == [1, 2, 1]
        # Peptide-local 1-indexed positions
        assert parsed.iloc[1]["phospho_positions_peptide"] == (1, 9)

    def test_mods_field_lists_all_bracket_content(self):
        ids = pd.Series(["_S[Phospho (STY)]TDNAFENPFFK_.2"])
        parsed = _parse_precursors(ids)
        assert parsed.iloc[0]["mods"] == "Phospho (STY)"

    def test_non_phospho_returns_zero_count(self):
        parsed = _parse_precursors(pd.Series(["_LMNVTPVLK_.2"]))
        assert parsed.iloc[0]["n_phospho"] == 0

    def test_key_has_alphaphos_format(self):
        proteins = pd.Series(["P00533"])
        genes = pd.Series(["EGFR"])
        peptides = pd.Series(["STDNAFENPFFK"])
        charges = pd.Series([2])
        mods = pd.Series(["Phospho (STY)"])
        key = _build_keys(proteins, genes, peptides, charges, mods).iloc[0]
        assert key == "P00533|EGFR|STDNAFENPFFK|2|Phospho (STY)"

    def test_empty_mods_slot_rendered_as_none(self):
        key = _build_keys(
            pd.Series(["P1"]),
            pd.Series(["G1"]),
            pd.Series(["ABCDE"]),
            pd.Series([2]),
            pd.Series([""]),
        ).iloc[0]
        assert key == "P1|G1|ABCDE|2|none"


# ---------------------------------------------------------------------------
# End-to-end collapse_precursors
# ---------------------------------------------------------------------------


class TestCollapsePrecursorsEndToEnd:
    def test_returns_anndata_with_expected_shape(self):
        psm = _make_synthetic_psm()
        conditions = _make_conditions()
        adata = ap.collapse_precursors(psm, condition_df=conditions)
        # 2 phospho precursors (AKT1 + SIK1B), 4 samples
        assert adata.n_obs == 4
        assert adata.n_vars == 2
        assert adata.layers["intensity_log2"].shape == adata.X.shape

    def test_non_phospho_dropped_by_default(self):
        psm = _make_synthetic_psm()
        adata = ap.collapse_precursors(psm)
        assert "GENEX" not in adata.var["gene"].tolist()

    def test_phospho_only_false_keeps_non_phospho(self):
        psm = _make_synthetic_psm()
        adata = ap.collapse_precursors(psm, advanced={"phospho_only": False})
        assert "GENEX" in adata.var["gene"].tolist()

    def test_var_index_is_alphaphos_precursor_key(self):
        psm = _make_synthetic_psm()
        adata = ap.collapse_precursors(psm)
        for key in adata.var.index:
            # "Protein|Gene|Peptide|Charge|Mods"
            parts = key.split("|")
            assert len(parts) == 5
        assert adata.var.index.name == "precursor_key"

    def test_var_columns_populated(self):
        psm = _make_synthetic_psm()
        adata = ap.collapse_precursors(psm)
        for col in (
            "protein_group_id",
            "gene",
            "peptide_sequence",
            "charge",
            "mods",
            "n_phospho",
            "peptide_start",
            "best_localization_prob",
            "best_localization_pos_peptide",
            "best_localization_pos_protein",
        ):
            assert col in adata.var.columns, f"{col} missing from var"

    def test_aggregation_sum_combines_duplicate_rows(self):
        # Our fixture has TWO rows per (AKT1 precursor, sample): 1000+drift and 300.
        # Sum agg should give ~1300 * hash-noise before log2.
        psm = _make_synthetic_psm()
        adata_sum = ap.collapse_precursors(psm, advanced={"aggregation_method": "sum"})
        akt = adata_sum.var["gene"] == "AKT1"
        # In sum mode, AKT1's linear intensity (before log2) is ~1300, so
        # log2 ~ 10.34.
        akt_log2 = adata_sum.layers["intensity_log2"][:, akt].flatten()
        assert (akt_log2 > 10.0).all()
        assert (akt_log2 < 11.5).all()

    def test_aggregation_mean_gives_lower_value(self):
        psm = _make_synthetic_psm()
        adata_sum = ap.collapse_precursors(psm, advanced={"aggregation_method": "sum"})
        adata_mean = ap.collapse_precursors(psm, advanced={"aggregation_method": "mean"})
        akt_s = adata_sum.var["gene"] == "AKT1"
        akt_m = adata_mean.var["gene"] == "AKT1"
        # mean(a, b) < a + b always (both positive)
        assert (
            adata_mean.layers["intensity_log2"][:, akt_m].mean()
            < adata_sum.layers["intensity_log2"][:, akt_s].mean()
        )

    def test_obs_carries_condition_and_selectivity(self):
        psm = _make_synthetic_psm()
        conditions = _make_conditions()
        adata = ap.collapse_precursors(psm, condition_df=conditions)
        assert "condition" in adata.obs.columns
        assert "phospho_selectivity_pct" in adata.obs.columns
        # 3 phospho / 4 rows per sample per fixture (AKT dup + SIK1B ctrl-only + non-phospho)
        # selectivity is >= 50 for every sample
        assert (adata.obs["phospho_selectivity_pct"] > 40).all()

    def test_uns_provenance(self):
        psm = _make_synthetic_psm()
        adata = ap.collapse_precursors(psm)
        prov = adata.uns["alphaphos"]
        assert "version" in prov
        assert "pipeline_params" in prov
        assert prov["pipeline_params"]["search_engine"] == "SN"
        assert prov["stats"]["n_precursors"] == 2
        assert prov["stats"]["n_samples"] == 4

    def test_localization_annotated_but_never_masking(self):
        psm = _make_synthetic_psm()
        adata = ap.collapse_precursors(psm)
        # AKT1 precursor has loc 95% at pep-pos 1 and 88% at pep-pos 12; peak is 95.
        akt_row = adata.var[adata.var["gene"] == "AKT1"].iloc[0]
        assert akt_row["best_localization_prob"] == pytest.approx(0.95, abs=1e-6)
        assert akt_row["best_localization_pos_peptide"] == 1
        # Absolute protein position = peptide_start (473) + pep_pos (1) - 1 = 473
        assert akt_row["best_localization_pos_protein"] == 473
        # Localization layer is populated (not NaN)
        assert not np.isnan(adata.layers["localization"]).all()

    def test_annotate_localization_false_skips(self):
        psm = _make_synthetic_psm()
        # annotate_localization=False also disables the classI gate (no loc
        # info to gate on -- the resolver enforces this pairing).
        adata = ap.collapse_precursors(
            psm,
            advanced={"annotate_localization": False, "classI_cutoff": None},
        )
        akt_row = adata.var[adata.var["gene"] == "AKT1"].iloc[0]
        assert pd.isna(akt_row["best_localization_prob"])

    def test_noise_floor_filter_drops_zero_log2_cells(self):
        psm = _make_synthetic_psm()
        # Inject two AKT1 rows summing to 1.0 for ctrl_1 -> log2(1) = 0 -> in
        # the {0, 1} noise floor set, so the cell must land as NaN.
        akt_mask = psm["EG.PrecursorId"].str.contains(r"TVQVAVSAGKT", regex=True)
        psm.loc[akt_mask & (psm["R.FileName"] == "ctrl_1"), "EG.TotalQuantity (Settings)"] = 0.5
        adata = ap.collapse_precursors(psm, advanced={"noise_floor_filter": True})
        akt_idx = adata.var.index[adata.var["gene"] == "AKT1"][0]
        ctrl1_idx = list(adata.obs.index).index("ctrl_1")
        val = adata.layers["intensity_log2"][ctrl1_idx][list(adata.var.index).index(akt_idx)]
        assert np.isnan(val)

    def test_source_attrs_preserved(self):
        psm = _make_synthetic_psm()
        psm.attrs["source_path"] = "synthetic.parquet"
        psm.attrs["engine"] = "SN"
        adata = ap.collapse_precursors(psm)
        assert adata.uns["source_attrs"]["engine"] == "SN"

    def test_missing_precursor_col_raises(self):
        psm = _make_synthetic_psm().drop(columns=["EG.PrecursorId"])
        with pytest.raises(KeyError, match="EG.PrecursorId"):
            ap.collapse_precursors(psm)


# ---------------------------------------------------------------------------
# Bridge: precursor -> best-guess site key
# ---------------------------------------------------------------------------


class TestPrecursorToSiteView:
    def test_returns_site_key_for_confident_localization(self):
        psm = _make_synthetic_psm()
        adata = ap.collapse_precursors(psm)
        bridge = ap.precursor_to_site_view(adata)
        # AKT1 precursor: loc 95% at peptide pos 1, peptide_start=473
        # peptide "STVQVAVSAGKTYHR" position 1 is "S" -> S473 with M2 (2 phospho)
        akt_row = bridge[bridge.index.str.contains("AKT1")]
        assert not akt_row.empty
        row = akt_row.iloc[0]
        assert bool(row["passes_localization"])
        assert row["site_residue"] == "S"
        assert int(row["site_position_protein"]) == 473
        assert row["site_key"] == "P31749|AKT1|S473|M2"

    def test_below_threshold_returns_nan_site_key(self):
        psm = _make_synthetic_psm()
        adata = ap.collapse_precursors(psm)
        bridge = ap.precursor_to_site_view(adata, require_localization=0.99)
        # All test precursors have loc <= 95% -> all NaN site keys
        assert bridge["site_key"].isna().all()

    def test_require_localization_none_skips_threshold(self):
        psm = _make_synthetic_psm()
        adata = ap.collapse_precursors(psm)
        bridge_gated = ap.precursor_to_site_view(adata, require_localization=0.75)
        bridge_open = ap.precursor_to_site_view(adata, require_localization=None)
        # Opening the gate should never remove keys
        assert bridge_open["site_key"].notna().sum() >= bridge_gated["site_key"].notna().sum()

    def test_raises_when_var_missing_annotation_columns(self):
        psm = _make_synthetic_psm()
        adata = ap.collapse_precursors(
            psm,
            advanced={"annotate_localization": False, "classI_cutoff": None},
        )
        adata.var.drop(columns=["best_localization_pos_peptide"], inplace=True)
        with pytest.raises(ValueError, match="best_localization_pos_peptide"):
            ap.precursor_to_site_view(adata)


# ---------------------------------------------------------------------------
# TRKA-style motivating case: high loc + high detection but sparse per-run
# ---------------------------------------------------------------------------


class TestTRKAStyleRescue:
    """Reproduces the motivating rationale in the CHO TAB2 doc: a precursor
    with 99.9% localization on the residue of interest, detected in most
    runs -- but with per-run loc noise that would trip site-level masking.
    Precursor-level must keep it."""

    def _make_trka_style_psm(self, n_runs: int = 10, n_missing_runs: int = 2) -> pd.DataFrame:
        """A single high-loc phospho precursor with a few missing runs."""
        rows = []
        for i in range(n_runs):
            if i < n_missing_runs:
                continue  # simulate detection failure in a few runs
            rows.append(
                {
                    "R.FileName": f"run_{i}",
                    "EG.PrecursorId": "_DIYS[Phospho (STY)]TDYYR_.2",
                    "EG.TotalQuantity (Settings)": 5000.0 + i * 10,
                    "PEP.PeptidePosition": "678",
                    "EG.PTMLocalizationProbabilities": ("_DIYS[Phospho (STY): 99.9%]TDYYR_"),
                    "PG.Genes": "NTRK1",
                    "PG.ProteinGroups": "P04629",
                }
            )
        return pd.DataFrame(rows)

    def test_high_loc_precursor_survives(self):
        psm = self._make_trka_style_psm(n_runs=10, n_missing_runs=2)
        adata = ap.collapse_precursors(psm)
        # One precursor row, 8 samples (10 runs minus 2 missing)
        assert adata.n_vars == 1
        assert adata.n_obs == 8
        row = adata.var.iloc[0]
        assert row["gene"] == "NTRK1"
        assert row["best_localization_prob"] == pytest.approx(0.999, abs=1e-3)

    def test_bridge_recovers_the_target_site(self):
        psm = self._make_trka_style_psm(n_runs=10, n_missing_runs=2)
        adata = ap.collapse_precursors(psm)
        bridge = ap.precursor_to_site_view(adata, require_localization=0.75)
        row = bridge.iloc[0]
        # Peptide "DIYSTDYYR" position 4 is "S" -- absolute pos =
        # peptide_start (678) + 4 - 1 = 681
        assert row["site_residue"] == "S"
        assert row["site_position_protein"] == 681
        assert row["site_key"] == "P04629|NTRK1|S681|M1"


# ---------------------------------------------------------------------------
# aggregate_to_site_level: precursor-indexed diff-exp -> site-indexed
# ---------------------------------------------------------------------------


class TestAggregateToSiteLevel:
    """Multiple precursors covering the same site must aggregate cleanly so
    downstream KSEA / kinase_activity sees a duplicate-free site index."""

    def _make_precursor_result_and_view(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Three precursors on two sites (two share a site) + one no-site."""
        precursor_result = pd.DataFrame(
            {
                "log2fc": [3.0, 1.0, -2.0, 5.0],
                "fdr": [0.001, 0.05, 0.002, 0.20],
            },
            index=[
                "P1|GA|PEPA|2|Phospho (STY)",  # site A
                "P1|GA|PEPB|2|Phospho (STY)",  # site A (same site, different peptide)
                "P2|GB|PEPC|2|Phospho (STY)",  # site B
                "P3|GC|PEPD|2|Phospho (STY)",  # no site (below loc threshold)
            ],
        )
        site_view = pd.DataFrame(
            {
                "site_key": ["P1|GA|S10|M1", "P1|GA|S10|M1", "P2|GB|S20|M1", None],
                "passes_localization": [True, True, True, False],
            },
            index=[
                "P1|GA|PEPA|2|Phospho (STY)",
                "P1|GA|PEPB|2|Phospho (STY)",
                "P2|GB|PEPC|2|Phospho (STY)",
                "P3|GC|PEPD|2|Phospho (STY)",
            ],
        )
        return precursor_result, site_view

    def test_max_abs_keeps_strongest(self):
        result, view = self._make_precursor_result_and_view()
        out = aggregate_to_site_level(result, view, stat_col="log2fc", stat_agg="max_abs")
        # Two unique sites, third dropped (no site_key)
        assert len(out) == 2
        # Site A had two precursors: 3.0 and 1.0 -> max_abs keeps 3.0
        assert out.loc["P1|GA|S10|M1", "log2fc"] == pytest.approx(3.0)
        # Site B single precursor
        assert out.loc["P2|GB|S20|M1", "log2fc"] == pytest.approx(-2.0)

    def test_mean_averages_across_precursors(self):
        result, view = self._make_precursor_result_and_view()
        out = aggregate_to_site_level(result, view, stat_col="log2fc", stat_agg="mean")
        # Site A: mean(3.0, 1.0) = 2.0
        assert out.loc["P1|GA|S10|M1", "log2fc"] == pytest.approx(2.0)

    def test_first_agg(self):
        result, view = self._make_precursor_result_and_view()
        out = aggregate_to_site_level(result, view, stat_col="log2fc", stat_agg="first")
        assert out.loc["P1|GA|S10|M1", "log2fc"] == pytest.approx(3.0)

    def test_fdr_min_aggregation(self):
        result, view = self._make_precursor_result_and_view()
        out = aggregate_to_site_level(result, view, stat_col="log2fc", fdr_agg="min")
        # Site A had FDRs 0.001 and 0.05 -> min = 0.001
        assert out.loc["P1|GA|S10|M1", "fdr"] == pytest.approx(0.001)

    def test_fdr_mean_aggregation(self):
        result, view = self._make_precursor_result_and_view()
        out = aggregate_to_site_level(result, view, stat_col="log2fc", fdr_agg="mean")
        assert out.loc["P1|GA|S10|M1", "fdr"] == pytest.approx(0.0255)

    def test_fdr_col_none_skips_fdr(self):
        result, view = self._make_precursor_result_and_view()
        out = aggregate_to_site_level(result, view, stat_col="log2fc", fdr_col=None)
        assert "fdr" not in out.columns

    def test_n_precursors_column(self):
        result, view = self._make_precursor_result_and_view()
        out = aggregate_to_site_level(result, view, stat_col="log2fc")
        assert out.loc["P1|GA|S10|M1", "n_precursors"] == 2
        assert out.loc["P2|GB|S20|M1", "n_precursors"] == 1

    def test_site_key_index_is_unique(self):
        """The whole point of the helper: no duplicate site keys in the output."""
        result, view = self._make_precursor_result_and_view()
        out = aggregate_to_site_level(result, view, stat_col="log2fc")
        assert out.index.is_unique

    def test_precursors_without_site_key_dropped(self):
        result, view = self._make_precursor_result_and_view()
        out = aggregate_to_site_level(result, view, stat_col="log2fc")
        # "P3|GC|PEPD|..." had no site_key -> dropped
        assert not (out.index == "P3|GC").any()

    def test_bad_stat_agg_raises(self):
        result, view = self._make_precursor_result_and_view()
        with pytest.raises(ValueError, match="stat_agg must be"):
            aggregate_to_site_level(result, view, stat_agg="median")

    def test_bad_fdr_agg_raises(self):
        result, view = self._make_precursor_result_and_view()
        with pytest.raises(ValueError, match="fdr_agg must be"):
            aggregate_to_site_level(result, view, fdr_agg="max")

    def test_missing_site_key_column_raises(self):
        result, view = self._make_precursor_result_and_view()
        with pytest.raises(ValueError, match="site_view must have a 'site_key' column"):
            aggregate_to_site_level(result, view.drop(columns=["site_key"]))

    def test_missing_stat_col_raises(self):
        result, view = self._make_precursor_result_and_view()
        with pytest.raises(KeyError, match="stat_col"):
            aggregate_to_site_level(result, view, stat_col="does_not_exist")

    def test_all_dropped_raises(self):
        result, view = self._make_precursor_result_and_view()
        view["site_key"] = None
        with pytest.raises(ValueError, match="No precursors have a resolvable site_key"):
            aggregate_to_site_level(result, view)


# ---------------------------------------------------------------------------
# Gene extraction from precursor keys (in pathway modules)
# ---------------------------------------------------------------------------


class TestPathwayGeneExtractionAcceptsBothKeyFormats:
    """After the fix, ``pathway_enrichment`` and ``pathway_gsea`` extract
    genes from BOTH site keys (``Protein|Gene|Site|Mult``) AND precursor
    keys (``Protein|Gene|Peptide|Charge|Mods``)."""

    def test_pathway_enrichment_gene_extraction_works_on_precursor_keys(self):
        from alphaphos.enrichment.pathway.enrichment import _keys_to_genes

        keys = [
            "P00533|EGFR|SPMK|2|Phospho (STY)",  # precursor key
            "P00533|EGFR|S1046|M1",  # site key
            "not_a_key",  # gibberish -- dropped
            "P|G",  # too short but has gene
        ]
        out = _keys_to_genes(keys)
        # Both key formats resolved to EGFR
        assert out["P00533|EGFR|SPMK|2|Phospho (STY)"] == "EGFR"
        assert out["P00533|EGFR|S1046|M1"] == "EGFR"
        # Gibberish (no pipe) dropped
        assert "not_a_key" not in out
        # Two-field key with gene still works
        assert out["P|G"] == "G"

    def test_pathway_gsea_gene_extraction_works_on_precursor_keys(self):
        from alphaphos.enrichment.pathway_gsea.gsea import _keys_to_genes

        keys = [
            "P00533|EGFR|SPMK|2|Phospho (STY)",
            "P00533|EGFR|S1046|M1",
            "not_a_key",
        ]
        out = _keys_to_genes(keys)
        assert out["P00533|EGFR|SPMK|2|Phospho (STY)"] == "EGFR"
        assert out["P00533|EGFR|S1046|M1"] == "EGFR"
        assert "not_a_key" not in out
