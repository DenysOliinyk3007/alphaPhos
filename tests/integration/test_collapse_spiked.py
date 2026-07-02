"""Spike synthetic phospho PSMs into the real EGF mini-fixture and verify
that ``collapse_sites`` emits the expected site rows in ``adata.var``.

Every synthetic protein uses ID prefix ``TEST_`` so the spike-in sites are
easy to isolate from the ~110 real sites in the base fixture.

Test matrix (proof-of-concept -- expand from here):

* M1 unambiguous phospho at a clean STY residue -- baseline.
* M1 ambiguous (two candidate STYs, 55/45 split) -- top-N=1 attribution
  should keep only the winning position.
* M2 unambiguous (two placed phosphos at ~100% each) -- multiplicity=2
  site key emerges.
* Phospho at the N-terminal residue of the peptide -- position math
  boundary check (peptide_start == absolute position).
* Same site reached via two charge states -- collapse should merge the
  two rows into one site key with summed intensity.
* Same site reached via a peptide and its N-terminally-extended
  missed-cleavage variant -- again should merge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alphaphos as ap
from tests.fixtures.synthetic_psms import Spike, spike_into, uniform_intensities

EGF_MINI = Path(__file__).parent.parent / "data" / "egf_mini.tsv"


# --------------------------------------------------------------------------
# Fixtures shared across the whole spike-in module.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_psm_df() -> pd.DataFrame:
    """Real reader output on the mini EGF fixture."""
    if not EGF_MINI.exists():
        pytest.skip(f"mini fixture not present: {EGF_MINI}")
    return ap.read_spectronaut(EGF_MINI)


@pytest.fixture(scope="module")
def runs(base_psm_df) -> list[str]:
    """The 6 real R.FileName values in the base fixture."""
    return sorted(base_psm_df["R.FileName"].unique())


@pytest.fixture(scope="module")
def condition_df(runs) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": runs,
            "condition": ["EGF" if "withEGF" in r else "ctrl" for r in runs],
        }
    )


def _collapse_with(
    base_psm_df,
    condition_df,
    spikes: list[Spike],
    advanced: dict | None = None,
):
    combined = spike_into(base_psm_df, spikes)
    return ap.collapse_sites(combined, condition_df=condition_df, advanced=advanced)


def _synthetic_sites(adata) -> list[str]:
    """Site keys that came from a spike (protein starts with TEST_)."""
    return [k for k in adata.var_names if k.startswith("TEST_")]


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


class TestM1Unambiguous:
    """One clear phospho at loc-prob 99% -- the happy path."""

    def test_site_key_shape_and_position(self, base_psm_df, condition_df, runs):
        # Peptide: AAASAAASAAAK, phospho on the FIRST S (position 3), peptide starts at 100.
        # -> site should be S103, multiplicity M1.
        spike = Spike(
            protein_id="TEST_M1U",
            gene="M1UGENE",
            base_sequence="AAASAAASAAAK",
            peptide_start=100,
            phospho_positions=[3],
            loc_probs={3: 99.0, 7: 1.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        synthetic = _synthetic_sites(adata)
        assert synthetic == ["TEST_M1U|M1UGENE|S103|M1"]
        row = adata.var.loc[synthetic[0]]
        assert row["site_aa"] == "S"
        assert int(row["site_position"]) == 103
        assert int(row["multiplicity"]) == 1

    def test_intensity_reconstructs(self, base_psm_df, condition_df, runs):
        spike = Spike(
            protein_id="TEST_M1U2",
            gene="M1U2GENE",
            base_sequence="AAASAAAK",
            peptide_start=200,
            phospho_positions=[3],
            loc_probs={3: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        key = "TEST_M1U2|M1U2GENE|S203|M1"
        assert key in adata.var_names
        expected_log2 = float(np.log2(1e6))
        j = list(adata.var_names).index(key)
        actual = np.asarray(adata.layers["intensity_log2"])[:, j]
        # Every sample got the same 1e6 -> log2 20 (within float tolerance).
        np.testing.assert_allclose(actual, expected_log2, rtol=1e-4)


class TestM1AmbiguousLocalization:
    """Two candidate STYs; how the Class-I mask handles the ambiguity."""

    def test_ambiguous_below_cutoff_masked_out(self, base_psm_df, condition_df, runs):
        # Both candidates below the default classI_cutoff (0.75). The default
        # condition-aware mask should reject the site entirely -- ambiguous
        # localization is not called with the "condition" strategy.
        spike = Spike(
            protein_id="TEST_M1AM",
            gene="M1AMGENE",
            base_sequence="AASAAASAAAK",  # S at 2, S at 6
            peptide_start=300,
            phospho_positions=[2],
            loc_probs={2: 55.0, 6: 45.0},
            intensities_per_run=uniform_intensities(runs, 5e5),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        assert _synthetic_sites(adata) == []

    def test_moderate_confidence_winner_passes_cutoff(self, base_psm_df, condition_df, runs):
        # Winner at 80% > classI_cutoff 75%: the placed position emerges.
        # Loser at 20% is well below the cutoff and does not produce a
        # competing site (Spectronaut would only over-export the 80% variant).
        spike = Spike(
            protein_id="TEST_M1P",
            gene="M1PGENE",
            base_sequence="AASAAASAAAK",
            peptide_start=310,
            phospho_positions=[2],
            loc_probs={2: 80.0, 6: 20.0},
            intensities_per_run=uniform_intensities(runs, 5e5),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        assert _synthetic_sites(adata) == ["TEST_M1P|M1PGENE|S312|M1"]


class TestM2Unambiguous:
    """Two placed phosphos at high loc probs -> M2 site."""

    def test_multiplicity_2(self, base_psm_df, condition_df, runs):
        # Peptide: two placed phosphos on S at index 3 and S at index 7.
        # Loc probs sum to 200% (100+100), which is the M=2 convention.
        spike = Spike(
            protein_id="TEST_M2U",
            gene="M2UGENE",
            base_sequence="AAASAAASAAAK",
            peptide_start=400,
            phospho_positions=[3, 7],
            loc_probs={3: 99.5, 7: 99.5},
            intensities_per_run=uniform_intensities(runs, 2e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        synthetic = _synthetic_sites(adata)
        # M2 emits TWO site rows -- one per placed phospho -- both at
        # multiplicity=2. Site positions are 403 and 407.
        expected = {
            "TEST_M2U|M2UGENE|S403|M2",
            "TEST_M2U|M2UGENE|S407|M2",
        }
        assert set(synthetic) == expected
        for k in expected:
            row = adata.var.loc[k]
            assert int(row["multiplicity"]) == 2


class TestNTerminalPhospho:
    """Phospho on the peptide's first residue.  Off-by-one guard on
    ``site_position = peptide_start + intra - 1``."""

    def test_position_math(self, base_psm_df, condition_df, runs):
        # Peptide starts at residue 500 in the protein; phospho on the S at
        # intra-peptide position 0 -> absolute S500.
        spike = Spike(
            protein_id="TEST_NT",
            gene="NTGENE",
            base_sequence="SAAAAAAAAAR",
            peptide_start=500,
            phospho_positions=[0],
            loc_probs={0: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e5),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        synthetic = _synthetic_sites(adata)
        assert synthetic == ["TEST_NT|NTGENE|S500|M1"]
        assert int(adata.var.loc[synthetic[0], "site_position"]) == 500


class TestSameSiteMultipleCharges:
    """Same site reached via +2 and +3 charge states.  Collapse
    aggregates precursors into one site."""

    def test_two_charges_yield_one_site(self, base_psm_df, condition_df, runs):
        spikes = [
            Spike(
                protein_id="TEST_CHZ",
                gene="CHZGENE",
                base_sequence="AAASAAAK",
                peptide_start=600,
                phospho_positions=[3],
                loc_probs={3: 99.0},
                intensities_per_run=uniform_intensities(runs, 1e6),
                charge=2,
            ),
            Spike(
                protein_id="TEST_CHZ",
                gene="CHZGENE",
                base_sequence="AAASAAAK",
                peptide_start=600,
                phospho_positions=[3],
                loc_probs={3: 99.0},
                intensities_per_run=uniform_intensities(runs, 2e6),
                charge=3,
            ),
        ]
        adata = _collapse_with(base_psm_df, condition_df, spikes)
        synthetic = _synthetic_sites(adata)
        assert synthetic == ["TEST_CHZ|CHZGENE|S603|M1"]
        # Default aggregation ('sum') gives 1e6 + 2e6 = 3e6 per run.
        expected_log2 = float(np.log2(3e6))
        j = list(adata.var_names).index(synthetic[0])
        actual = np.asarray(adata.layers["intensity_log2"])[:, j]
        np.testing.assert_allclose(actual, expected_log2, rtol=1e-4)


class TestSameSiteViaMissedCleavage:
    """Two peptides differing only in a K/R miscleavage that both cover
    the same phosphosite must collapse to one site."""

    def test_missed_cleavage_variant_merges(self, base_psm_df, condition_df, runs):
        # Base peptide starting at residue 700: "AAASAAAK"; phospho on S at
        # intra 3 -> S703. Missed-cleavage variant starts one residue
        # earlier ("KAAASAAAK") at 699; phospho on S at intra 4 -> also S703.
        spikes = [
            Spike(
                protein_id="TEST_MC",
                gene="MCGENE",
                base_sequence="AAASAAAK",
                peptide_start=700,
                phospho_positions=[3],
                loc_probs={3: 99.0},
                intensities_per_run=uniform_intensities(runs, 8e5),
            ),
            Spike(
                protein_id="TEST_MC",
                gene="MCGENE",
                base_sequence="KAAASAAAK",
                peptide_start=699,
                phospho_positions=[4],
                loc_probs={4: 99.0},
                intensities_per_run=uniform_intensities(runs, 2e5),
            ),
        ]
        adata = _collapse_with(base_psm_df, condition_df, spikes)
        synthetic = _synthetic_sites(adata)
        assert synthetic == ["TEST_MC|MCGENE|S703|M1"]
        expected_log2 = float(np.log2(8e5 + 2e5))
        j = list(adata.var_names).index(synthetic[0])
        actual = np.asarray(adata.layers["intensity_log2"])[:, j]
        np.testing.assert_allclose(actual, expected_log2, rtol=1e-4)


class TestFixtureIsolation:
    """Sanity: real sites from the base fixture keep working alongside
    the spike-ins, and the spike-ins don't collide with real proteins."""

    def test_real_sites_survive_spike(self, base_psm_df, condition_df, runs):
        spike = Spike(
            protein_id="TEST_ISO",
            gene="ISOGENE",
            base_sequence="AASAAK",
            peptide_start=42,
            phospho_positions=[2],
            loc_probs={2: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata_spiked = _collapse_with(base_psm_df, condition_df, [spike])
        adata_bare = ap.collapse_sites(base_psm_df, condition_df=condition_df)

        real_sites_spiked = set(adata_spiked.var_names) - set(_synthetic_sites(adata_spiked))
        real_sites_bare = set(adata_bare.var_names)
        # Adding a synthetic protein must not remove any real site.
        assert real_sites_bare == real_sites_spiked

    def test_synthetic_prefix_present(self, base_psm_df, condition_df, runs):
        spike = Spike(
            protein_id="TEST_ISO2",
            gene="ISO2GENE",
            base_sequence="AASAAK",
            peptide_start=42,
            phospho_positions=[2],
            loc_probs={2: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        # Every spike-in var_name starts with TEST_
        for k in _synthetic_sites(adata):
            assert k.startswith("TEST_")


# ==========================================================================
# A. Multi-phospho depth
# ==========================================================================


class TestMultiplicityDepth:
    def test_M3_three_placed_phosphos(self, base_psm_df, condition_df, runs):
        # AASATSAYSAK -- STY at 2, 4, 5, 7, 8. Place three at 2, 5, 8 (all S).
        spike = Spike(
            protein_id="TEST_M3",
            gene="M3GENE",
            base_sequence="AASATSAYSAK",
            peptide_start=1000,
            phospho_positions=[2, 5, 8],
            loc_probs={2: 99.0, 4: 0.5, 5: 99.0, 7: 0.5, 8: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        synthetic = sorted(_synthetic_sites(adata))
        expected = sorted(
            [
                "TEST_M3|M3GENE|S1002|M3",
                "TEST_M3|M3GENE|S1005|M3",
                "TEST_M3|M3GENE|S1008|M3",
            ]
        )
        assert synthetic == expected
        for k in synthetic:
            assert int(adata.var.loc[k, "multiplicity"]) == 3

    def test_M4_clamps_to_multiplicity_3(self, base_psm_df, condition_df, runs):
        # SATSAYSSK -- STY at 0, 2, 3, 5, 6, 7. Place four at 0, 3, 5, 6.
        # alphaphos clamps multiplicity to 3 (M1/M2/M3+ convention).
        spike = Spike(
            protein_id="TEST_M4",
            gene="M4GENE",
            base_sequence="SATSAYSSK",
            peptide_start=200,
            phospho_positions=[0, 3, 5, 6],
            loc_probs={0: 99.0, 2: 0.5, 3: 99.0, 5: 99.0, 6: 99.0, 7: 0.5},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        synthetic = _synthetic_sites(adata)
        # Four placed phosphos -> four site rows, all with multiplicity=3 in the key
        assert len(synthetic) == 4
        assert all("|M3" in k for k in synthetic), synthetic
        for k in synthetic:
            assert int(adata.var.loc[k, "multiplicity"]) == 3


# ==========================================================================
# B. Mixed-modification peptides
# ==========================================================================


class TestMixedModifications:
    def test_carbamidomethyl_C_does_not_leak_into_site_key(self, base_psm_df, condition_df, runs):
        # Peptide with a C at position 4 (Cam-C) and a phospho on S at 7.
        # Site key must reflect ONLY the phospho -- Cam-C is a fixed mod, no
        # site row for it.
        spike = Spike(
            protein_id="TEST_CAM",
            gene="CAMGENE",
            base_sequence="AAAACAASAAK",
            peptide_start=100,
            phospho_positions=[7],
            loc_probs={7: 99.0},
            other_mods={4: "Carbamidomethyl (C)"},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        synthetic = _synthetic_sites(adata)
        assert synthetic == ["TEST_CAM|CAMGENE|S107|M1"]

    def test_oxidized_M_variant_merges_with_unoxidized(self, base_psm_df, condition_df, runs):
        # Same peptide, two variants: one with Oxidation-M at position 3,
        # one without. Same phospho site (S at intra 6). The site key is
        # (Protein|Gene|Site|Mult) -- neither depends on Oxidation-M -> the two
        # precursors collapse to one site.
        spikes = [
            Spike(
                protein_id="TEST_OX",
                gene="OXGENE",
                base_sequence="AAAMAASAAAK",
                peptide_start=300,
                phospho_positions=[6],
                loc_probs={6: 99.0},
                other_mods={3: "Oxidation (M)"},
                intensities_per_run=uniform_intensities(runs, 6e5),
                charge=3,
            ),
            Spike(
                protein_id="TEST_OX",
                gene="OXGENE",
                base_sequence="AAAMAASAAAK",
                peptide_start=300,
                phospho_positions=[6],
                loc_probs={6: 99.0},
                intensities_per_run=uniform_intensities(runs, 4e5),
                charge=3,
            ),
        ]
        adata = _collapse_with(base_psm_df, condition_df, spikes)
        synthetic = _synthetic_sites(adata)
        assert synthetic == ["TEST_OX|OXGENE|S306|M1"]
        j = list(adata.var_names).index(synthetic[0])
        actual = np.asarray(adata.layers["intensity_log2"])[:, j]
        np.testing.assert_allclose(actual, float(np.log2(6e5 + 4e5)), rtol=1e-4)

    def test_phospho_with_multiple_comods(self, base_psm_df, condition_df, runs):
        # Cam-C at 4 AND Ox-M at 6, phospho on S at 8. Site key still just
        # reflects the phospho.
        spike = Spike(
            protein_id="TEST_MULTI",
            gene="MULTIGENE",
            base_sequence="AAAACAMASAAK",
            peptide_start=100,
            phospho_positions=[8],
            loc_probs={8: 99.0},
            other_mods={4: "Carbamidomethyl (C)", 6: "Oxidation (M)"},
            intensities_per_run=uniform_intensities(runs, 5e5),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        assert _synthetic_sites(adata) == ["TEST_MULTI|MULTIGENE|S108|M1"]


# ==========================================================================
# C. Localization strategies
# ==========================================================================


def _split_runs(runs: list[str]) -> tuple[list[str], list[str]]:
    egf = [r for r in runs if "withEGF" in r]
    ctrl = [r for r in runs if "woEGF" in r]
    return egf, ctrl


class TestLocalizationStrategies:
    """The same PSMs -- high loc probs in EGF, low in ctrl -- under the
    three localization strategies produce different masking behaviors."""

    def _spikes(self, runs):
        # Same placed position (2) in all runs -- top-N=1 attribution doesn't
        # drop anything.  What varies is the CONFIDENCE of that top position:
        # 99% in EGF (Class-I) vs 65% in ctrl (below the 75% cutoff).
        egf, ctrl = _split_runs(runs)
        return [
            Spike(
                protein_id="TEST_LOC",
                gene="LOCGENE",
                base_sequence="AASAASAAAK",
                peptide_start=800,
                phospho_positions=[2],
                loc_probs={2: 99.0, 5: 1.0},
                intensities_per_run=uniform_intensities(egf, 1e6),
            ),
            Spike(
                protein_id="TEST_LOC",
                gene="LOCGENE",
                base_sequence="AASAASAAAK",
                peptide_start=800,
                phospho_positions=[2],
                loc_probs={2: 65.0, 5: 35.0},
                intensities_per_run=uniform_intensities(ctrl, 1e6),
            ),
        ]

    def test_condition_strategy_ctrl_masked(self, base_psm_df, condition_df, runs):
        # 3/3 EGF Class-I -> EGF cells kept; 0/3 ctrl -> ctrl cells masked to NaN.
        adata = _collapse_with(
            base_psm_df,
            condition_df,
            self._spikes(runs),
            advanced={"localization_strategy": "condition"},
        )
        key = "TEST_LOC|LOCGENE|S802|M1"
        assert key in adata.var_names
        j = list(adata.var_names).index(key)
        X = np.asarray(adata.layers["intensity_log2"])[:, j]
        egf_mask = adata.obs["condition"].to_numpy() == "EGF"
        ctrl_mask = ~egf_mask
        assert not np.isnan(X[egf_mask]).any(), "EGF cells should be kept"
        assert np.isnan(X[ctrl_mask]).all(), "ctrl cells should be masked"

    def test_per_run_strategy_masks_cell_by_cell(self, base_psm_df, condition_df, runs):
        # Per-run: each cell independently gates on its own loc prob.
        adata = _collapse_with(
            base_psm_df,
            condition_df,
            self._spikes(runs),
            advanced={"localization_strategy": "per_run"},
        )
        key = "TEST_LOC|LOCGENE|S802|M1"
        assert key in adata.var_names
        j = list(adata.var_names).index(key)
        X = np.asarray(adata.layers["intensity_log2"])[:, j]
        egf_mask = adata.obs["condition"].to_numpy() == "EGF"
        # EGF cells (99% loc) kept; ctrl cells (40% loc) masked.
        assert not np.isnan(X[egf_mask]).any()
        assert np.isnan(X[~egf_mask]).all()

    def test_global_max_strategy_keeps_all_cells(self, base_psm_df, condition_df, runs):
        # global_max: any single sample above cutoff -> keep the whole site.
        adata = _collapse_with(
            base_psm_df,
            condition_df,
            self._spikes(runs),
            advanced={"localization_strategy": "global_max"},
        )
        key = "TEST_LOC|LOCGENE|S802|M1"
        assert key in adata.var_names
        j = list(adata.var_names).index(key)
        X = np.asarray(adata.layers["intensity_log2"])[:, j]
        # All 6 cells kept (the peptide identity is trusted after the global check).
        assert not np.isnan(X).any()


# ==========================================================================
# D. Aggregation methods
# ==========================================================================


class TestAggregationMethods:
    """Same site reached by 3 charge states with intensities 1e6, 2e6, 3e6.
    Each aggregation method should produce a distinct expected value."""

    def _spikes(self, runs):
        return [
            Spike(
                protein_id="TEST_AGG",
                gene="AGGGENE",
                base_sequence="AAASAAAK",
                peptide_start=900,
                phospho_positions=[3],
                loc_probs={3: 99.0},
                intensities_per_run=uniform_intensities(runs, i),
                charge=c,
            )
            for c, i in [(2, 1e6), (3, 2e6), (4, 3e6)]
        ]

    def _site_log2(self, adata, key: str):
        j = list(adata.var_names).index(key)
        return np.asarray(adata.layers["intensity_log2"])[:, j]

    def test_sum(self, base_psm_df, condition_df, runs):
        adata = _collapse_with(
            base_psm_df,
            condition_df,
            self._spikes(runs),
            advanced={"aggregation_method": "sum"},
        )
        expected = float(np.log2(1e6 + 2e6 + 3e6))
        vals = self._site_log2(adata, "TEST_AGG|AGGGENE|S903|M1")
        np.testing.assert_allclose(vals, expected, rtol=1e-4)

    def test_median(self, base_psm_df, condition_df, runs):
        adata = _collapse_with(
            base_psm_df,
            condition_df,
            self._spikes(runs),
            advanced={"aggregation_method": "median"},
        )
        expected = float(np.log2(2e6))  # median of {1,2,3}e6
        vals = self._site_log2(adata, "TEST_AGG|AGGGENE|S903|M1")
        np.testing.assert_allclose(vals, expected, rtol=1e-4)

    def test_mean(self, base_psm_df, condition_df, runs):
        adata = _collapse_with(
            base_psm_df,
            condition_df,
            self._spikes(runs),
            advanced={"aggregation_method": "mean"},
        )
        expected = float(np.log2(2e6))  # mean of {1,2,3}e6
        vals = self._site_log2(adata, "TEST_AGG|AGGGENE|S903|M1")
        np.testing.assert_allclose(vals, expected, rtol=1e-4)

    def test_consolidate_produces_a_finite_value(self, base_psm_df, condition_df, runs):
        # "consolidate" is Hogrebe's ratio-based aggregation. Exact expected
        # value depends on internals; assert it stays in a plausible range
        # bounded by the input intensities.
        adata = _collapse_with(
            base_psm_df,
            condition_df,
            self._spikes(runs),
            advanced={"aggregation_method": "consolidate"},
        )
        vals = self._site_log2(adata, "TEST_AGG|AGGGENE|S903|M1")
        assert not np.isnan(vals).any()
        # log2(1e6) ~ 19.93; log2(6e6) ~ 22.52. Any reasonable aggregation
        # of the three sits inside this range.
        assert (vals >= float(np.log2(1e6)) - 0.1).all()
        assert (vals <= float(np.log2(6e6)) + 0.1).all()


# ==========================================================================
# E. Multi-protein groups
# ==========================================================================


class TestMultiProteinGroup:
    def test_PG_level_picks_first_protein(self, base_psm_df, condition_df, runs):
        # Peptide maps to two proteins; default collapse_level='PG' picks
        # the first entry of the semicolon-joined PG.ProteinGroups.
        spike = Spike(
            protein_id="TEST_PG1",
            gene="PG1GENE",
            base_sequence="AASAAAK",
            peptide_start=50,
            phospho_positions=[2],
            loc_probs={2: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
            protein_ids_semicolon="TEST_PG1;TEST_PG2",
            genes_semicolon="PG1GENE;PG2GENE",
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        synthetic = _synthetic_sites(adata)
        # 'PG' collapse level keeps only the first protein in the group.
        assert synthetic == ["TEST_PG1|PG1GENE|S52|M1"]
        assert adata.var.loc[synthetic[0], "protein_group_id"] == "TEST_PG1"
        assert adata.var.loc[synthetic[0], "gene"] == "PG1GENE"

    def test_P_level_keeps_full_group_string(self, base_psm_df, condition_df, runs):
        # collapse_level='P' keeps the semicolon-joined string in
        # protein_group_id (behavior of the current implementation --
        # downstream may explode it further, but the collapse output does
        # not).
        spike = Spike(
            protein_id="TEST_P1",
            gene="P1GENE",
            base_sequence="AASAAAK",
            peptide_start=60,
            phospho_positions=[2],
            loc_probs={2: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
            protein_ids_semicolon="TEST_P1;TEST_P2",
            genes_semicolon="P1GENE;P2GENE",
        )
        adata = _collapse_with(
            base_psm_df,
            condition_df,
            [spike],
            advanced={"collapse_level": "P"},
        )
        synthetic = _synthetic_sites(adata)
        assert len(synthetic) == 1
        assert synthetic[0].startswith("TEST_P1;TEST_P2|"), synthetic
        assert adata.var.loc[synthetic[0], "protein_group_id"] == "TEST_P1;TEST_P2"


# ==========================================================================
# F. Weird biology
# ==========================================================================


class TestWeirdBiology:
    def test_phospho_at_protein_position_1(self, base_psm_df, condition_df, runs):
        # Peptide covers the protein N-terminus (peptide_start=1); phospho on
        # the very first residue.  Absolute position must equal 1 -- not 0.
        spike = Spike(
            protein_id="TEST_NT1",
            gene="NT1GENE",
            base_sequence="SAAAAAAR",
            peptide_start=1,
            phospho_positions=[0],
            loc_probs={0: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        assert _synthetic_sites(adata) == ["TEST_NT1|NT1GENE|S1|M1"]
        assert int(adata.var.loc["TEST_NT1|NT1GENE|S1|M1", "site_position"]) == 1

    def test_tyrosine_phospho(self, base_psm_df, condition_df, runs):
        # Same as the baseline M1 test but with Y instead of S.
        spike = Spike(
            protein_id="TEST_Y",
            gene="YGENE",
            base_sequence="AAAYAAAK",
            peptide_start=400,
            phospho_positions=[3],
            loc_probs={3: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        assert _synthetic_sites(adata) == ["TEST_Y|YGENE|Y403|M1"]
        assert adata.var.loc["TEST_Y|YGENE|Y403|M1", "site_aa"] == "Y"

    def test_clustered_STY_M3(self, base_psm_df, condition_df, runs):
        # Adjacent STY residues (SSTTY).  M3 placed at 2, 4, 6 (S, T, Y).
        # No off-by-one crosstalk between adjacent sites.
        spike = Spike(
            protein_id="TEST_CLST",
            gene="CLSTGENE",
            base_sequence="AASSTTYAK",  # S=2, S=3, T=4, T=5, Y=6
            peptide_start=500,
            phospho_positions=[2, 4, 6],
            loc_probs={2: 99.0, 3: 0.5, 4: 99.0, 5: 0.5, 6: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        synthetic = sorted(_synthetic_sites(adata))
        expected = sorted(
            [
                "TEST_CLST|CLSTGENE|S502|M3",
                "TEST_CLST|CLSTGENE|T504|M3",
                "TEST_CLST|CLSTGENE|Y506|M3",
            ]
        )
        assert synthetic == expected


# ==========================================================================
# G. Bad inputs
# ==========================================================================


class TestBadInputs:
    def test_factory_rejects_non_STY_phospho(self):
        # The factory guards against phospho on non-STY at construction time.
        from tests.fixtures.synthetic_psms import make_phospho_psm

        with pytest.raises(ValueError, match="not STY"):
            make_phospho_psm(
                protein_id="TEST_BAD",
                gene="BADGENE",
                base_sequence="AAAAAAAK",
                peptide_start=100,
                phospho_positions=[3],  # A at position 3, not STY
                loc_probs={3: 99.0},
                intensities_per_run={"r1": 1e6},
            )

    def test_collapse_drops_non_STY_phospho_from_raw_row(self, base_psm_df, condition_df, runs):
        # Build a MALFORMED raw row that bypasses the factory guard: phospho
        # tag on an A residue (not STY). compute_site_metadata should drop it.
        from tests.fixtures.synthetic_psms import (
            SPECTRONAUT_COLUMNS,
            make_phospho_psm,
        )

        # Start from a valid PSM, then corrupt EG.PrecursorId to place the
        # phospho on the "A" at position 0.
        good = make_phospho_psm(
            protein_id="TEST_NSTY",
            gene="NSTYGENE",
            base_sequence="ASAAAAAK",  # S at 1
            peptide_start=100,
            phospho_positions=[1],
            loc_probs={1: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        good = good.copy()
        good["EG.PrecursorId"] = "_A[Phospho (STY)]SAAAAAK_.3"
        good["EG.PTMLocalizationProbabilities"] = "_A[Phospho (STY): 99%]SAAAAAK_"
        good["EG.ModifiedSequence"] = "_A[Phospho (STY)]SAAAAAK_"

        # Manually concat -- spike_into() would re-encode via the factory.
        keep_cols = [c for c in SPECTRONAUT_COLUMNS if c in base_psm_df.columns]
        combined = pd.concat([base_psm_df[keep_cols], good[keep_cols]], ignore_index=True)
        combined.attrs = dict(base_psm_df.attrs)
        adata = ap.collapse_sites(combined, condition_df=condition_df)
        # The malformed row must not produce a site.
        assert "TEST_NSTY|NSTYGENE|A100|M1" not in adata.var_names
        assert not any(k.startswith("TEST_NSTY|") for k in adata.var_names)

    def test_peptide_start_zero_documents_current_behavior(self, base_psm_df, condition_df, runs):
        # peptide_start=0 with intra=1 (1-indexed inside the pipeline) gives
        # absolute_position = 0 + 1 - 1 = 0.  Not a valid protein position,
        # but the current pipeline does NOT reject it -- this test just
        # characterizes the observed behavior so a future stricter guard
        # will surface as a change here.
        spike = Spike(
            protein_id="TEST_ZERO",
            gene="ZEROGENE",
            base_sequence="SAAAAAAK",
            peptide_start=0,
            phospho_positions=[0],
            loc_probs={0: 99.0},
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        # Current behavior: the site emerges with position 0 (peptide_start=0
        # + intra_1indexed=1 - 1 = 0). Not a valid protein position but the
        # pipeline does not reject it -- this test locks that behavior so a
        # future stricter guard surfaces here.
        synthetic = _synthetic_sites(adata)
        assert synthetic == ["TEST_ZERO|ZEROGENE|S0|M1"]

    def test_low_assay_probability_still_emits(self, base_psm_df, condition_df, runs):
        # EG.PTMAssayProbability is the overall assay-level confidence,
        # NOT the site-level localization probability. The current
        # pipeline does not gate on it -- documenting that here so a
        # future gate would surface as a change.
        spike = Spike(
            protein_id="TEST_LOW",
            gene="LOWGENE",
            base_sequence="AASAAAK",
            peptide_start=700,
            phospho_positions=[2],
            loc_probs={2: 99.0},
            ptm_assay_probability=0.01,  # extremely low
            intensities_per_run=uniform_intensities(runs, 1e6),
        )
        adata = _collapse_with(base_psm_df, condition_df, [spike])
        # Site emerges despite the low assay probability.
        assert "TEST_LOW|LOWGENE|S702|M1" in adata.var_names
