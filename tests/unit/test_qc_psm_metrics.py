"""Tests for :func:`alphaphos.qc.compute_retention_time_drift`.

Covers:
- Synthetic drift injection (planted signal in a small toy PSM df)
- Missing-column error path
- Empty/degenerate inputs
- Queue integration (with a fully-matched queue, and a mismatched one)
- Real cardio integration (loose sanity checks)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaphos.qc.psm_metrics import (
    DEFAULT_INTENSITY_COLUMN,
    DEFAULT_PRECURSOR_COLUMN,
    DEFAULT_PROTEIN_COLUMN,
    DEFAULT_RT_COLUMN,
    DEFAULT_SAMPLE_COLUMN,
    compute_contaminant_fraction_per_sample,
    compute_psm_counts_per_sample,
    compute_retention_time_drift,
)
from alphaphos.qc.queue_io import load_acquisition_queue

REPO = Path(__file__).resolve().parents[2]
CARDIO_QUEUE = REPO / "test_data" / "qc" / "queue_ElKr_phosphoDVP.csv"
CARDIO_PSM = REPO / "test_data" / "proteome_path" / "cardiomyocytes_dvp_phospho_raw.parquet"


def _make_synthetic_psm_df(
    n_samples: int = 5,
    n_precursors: int = 30,
    drift_by_sample: dict[int, float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthesise a small PSM df with a known-planted RT drift signal.

    Each precursor has a base RT drawn from U[10, 90] min.  Each sample's
    observed RT = base + sample_drift + Gaussian noise (0.02 min).
    """
    rng = np.random.default_rng(seed)
    drift_by_sample = drift_by_sample or {}
    rows: list[dict] = []
    for p in range(n_precursors):
        base_rt = float(rng.uniform(10.0, 90.0))
        for s in range(n_samples):
            sample_drift = drift_by_sample.get(s, 0.0)
            rt = base_rt + sample_drift + rng.normal(0, 0.02)
            rows.append(
                {
                    DEFAULT_SAMPLE_COLUMN: f"sample_{s:02d}",
                    DEFAULT_PRECURSOR_COLUMN: f"precursor_{p:03d}",
                    DEFAULT_RT_COLUMN: rt,
                }
            )
    return pd.DataFrame(rows)


class TestBasicShape:
    def test_returns_one_row_per_sample(self):
        psm = _make_synthetic_psm_df(n_samples=5, n_precursors=20)
        drift = compute_retention_time_drift(psm)
        assert drift.shape[0] == 5
        assert set(drift.columns) == {
            "n_shared_precursors",
            "median_rt",
            "median_offset",
            "iqr_offset",
            "gradient_slope",
            "is_flagged",
        }

    def test_missing_rt_column_raises(self):
        psm = _make_synthetic_psm_df(n_samples=4, n_precursors=10)
        psm = psm.drop(columns=[DEFAULT_RT_COLUMN])
        with pytest.raises(KeyError, match="missing required columns"):
            compute_retention_time_drift(psm)


class TestDriftDetection:
    def test_no_drift_gives_near_zero_offsets(self):
        # All samples share the same RT distribution -> offsets ≈ 0
        psm = _make_synthetic_psm_df(n_samples=6, n_precursors=30)
        drift = compute_retention_time_drift(psm)
        assert (drift["median_offset"].abs() < 0.1).all()
        assert not drift["is_flagged"].any()

    def test_planted_offset_recovered(self):
        # sample_02 shifted by +2 min uniformly -> its median_offset ≈ +2
        psm = _make_synthetic_psm_df(
            n_samples=6,
            n_precursors=30,
            drift_by_sample={2: 2.0},
            seed=42,
        )
        drift = compute_retention_time_drift(psm, offset_flag_threshold=1.0)
        assert drift.loc["sample_02", "is_flagged"]
        assert abs(drift.loc["sample_02", "median_offset"] - 2.0) < 0.5
        # Others are not flagged
        others = drift.drop("sample_02")
        assert not others["is_flagged"].any()

    def test_gradient_distortion_signal(self):
        # Inject a gradient distortion: sample_03 has RT offset that
        # scales with the base RT (later peptides drift more).
        rng = np.random.default_rng(0)
        rows = []
        for p in range(50):
            base_rt = float(rng.uniform(10.0, 90.0))
            for s in range(5):
                if s == 3:
                    # Non-linear distortion: offset grows with base_rt
                    offset = 0.02 * base_rt
                else:
                    offset = 0.0
                rows.append(
                    {
                        DEFAULT_SAMPLE_COLUMN: f"sample_{s}",
                        DEFAULT_PRECURSOR_COLUMN: f"prec_{p}",
                        DEFAULT_RT_COLUMN: base_rt + offset + rng.normal(0, 0.01),
                    }
                )
        drift = compute_retention_time_drift(pd.DataFrame(rows))
        # sample_3 should have a positive gradient_slope; others near zero
        assert drift.loc["sample_3", "gradient_slope"] > 0.01
        others = drift.drop("sample_3")
        assert (others["gradient_slope"].abs() < 0.01).all()


class TestQueueIntegration:
    def test_queue_attached_adds_expected_columns(self):
        psm = _make_synthetic_psm_df(n_samples=4, n_precursors=20)
        queue = pd.DataFrame(
            {
                "sample": [f"sample_{i:02d}" for i in range(4)],
                "injection_order": [1, 2, 3, 4],
            }
        )
        drift = compute_retention_time_drift(psm, acquisition_queue=queue)
        for extra in ("injection_order", "offset_slope_vs_order", "is_early_batch"):
            assert extra in drift.columns
        assert drift.attrs["provenance"]["n_matched_queue"] == 4
        assert drift.attrs["provenance"]["n_unmatched_psm_samples"] == 0

    def test_queue_partial_mismatch_recorded_in_provenance(self):
        psm = _make_synthetic_psm_df(n_samples=4, n_precursors=20)
        # Queue misses sample_02
        queue = pd.DataFrame(
            {
                "sample": ["sample_00", "sample_01", "sample_03"],
                "injection_order": [1, 2, 3],
            }
        )
        drift = compute_retention_time_drift(psm, acquisition_queue=queue)
        prov = drift.attrs["provenance"]
        assert prov["n_matched_queue"] == 3
        assert prov["n_unmatched_psm_samples"] == 1
        assert "sample_02" in prov["unmatched_psm_samples"]
        assert pd.isna(drift.loc["sample_02", "injection_order"])

    def test_queue_missing_required_columns_raises(self):
        psm = _make_synthetic_psm_df(n_samples=3, n_precursors=10)
        bad_queue = pd.DataFrame({"sample": ["sample_00"]})  # no injection_order
        with pytest.raises(ValueError, match="missing required columns"):
            compute_retention_time_drift(psm, acquisition_queue=bad_queue)


# ---------------------------------------------------------------------------
# compute_psm_counts_per_sample
# ---------------------------------------------------------------------------


def _make_synthetic_counts_df(
    n_per_sample: dict[str, int],
    n_precursors_pool: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per PSM.  Each sample gets `n_per_sample[sample]` rows drawn
    from a pool of `n_precursors_pool` distinct precursor IDs."""
    rng = np.random.default_rng(seed)
    rows = []
    precursor_pool = [f"p_{i:03d}" for i in range(n_precursors_pool)]
    protein_pool = [f"PROT_{i // 5:03d}" for i in range(n_precursors_pool)]
    for sample, n_psms in n_per_sample.items():
        idx = rng.integers(0, n_precursors_pool, size=n_psms)
        for i in idx:
            rows.append(
                {
                    DEFAULT_SAMPLE_COLUMN: sample,
                    DEFAULT_PRECURSOR_COLUMN: precursor_pool[i],
                    DEFAULT_PROTEIN_COLUMN: protein_pool[i],
                }
            )
    return pd.DataFrame(rows)


class TestPsmCountsShape:
    def test_returns_expected_columns(self):
        psm = _make_synthetic_counts_df({"s01": 100, "s02": 120})
        counts = compute_psm_counts_per_sample(psm)
        assert set(counts.columns) == {
            "n_psms",
            "n_unique_precursors",
            "n_unique_proteins",
            "depth_z_score",
            "is_flagged",
        }
        assert counts.shape[0] == 2

    def test_missing_column_raises(self):
        psm = _make_synthetic_counts_df({"s01": 50, "s02": 60})
        psm = psm.drop(columns=[DEFAULT_PROTEIN_COLUMN])
        with pytest.raises(KeyError, match="missing required columns"):
            compute_psm_counts_per_sample(psm)


class TestPsmCountsFlagging:
    def test_uniform_depth_no_flags(self):
        # 6 samples all with ~100 rows -> z-scores near 0, no flags
        psm = _make_synthetic_counts_df(
            {f"s{i:02d}": 100 for i in range(6)},
            seed=1,
        )
        counts = compute_psm_counts_per_sample(psm)
        assert not counts["is_flagged"].any()
        assert (counts["depth_z_score"].abs() < 3.0).all()

    def test_one_deep_outlier_gets_flagged(self):
        # 8 samples at ~500 rows, one at ~20 rows -> heavy z-score,
        # gets flagged.
        depths = {f"s{i:02d}": 500 for i in range(8)}
        depths["s08_broken"] = 20
        psm = _make_synthetic_counts_df(depths, n_precursors_pool=800, seed=2)
        counts = compute_psm_counts_per_sample(psm)
        assert bool(counts.loc["s08_broken", "is_flagged"])
        assert counts.loc["s08_broken", "depth_z_score"] < -3.0
        # Others not flagged
        others = counts.drop("s08_broken")
        assert not others["is_flagged"].any()

    def test_mad_is_robust_to_outliers(self):
        # With one crash sample dragging the mean but not the median,
        # the MAD-based z-score of the OTHER samples should stay small.
        depths = {f"s{i:02d}": 1000 for i in range(10)}
        depths["s09_broken"] = 10
        psm = _make_synthetic_counts_df(depths, n_precursors_pool=1500, seed=3)
        counts = compute_psm_counts_per_sample(psm)
        # Non-broken samples should have |z| < 3
        others = counts.drop("s09_broken")
        assert (others["depth_z_score"].abs() < 3.0).all()


class TestPsmCountsQueueIntegration:
    def test_queue_adds_injection_order_column(self):
        psm = _make_synthetic_counts_df({"s01": 50, "s02": 60, "s03": 55})
        queue = pd.DataFrame({"sample": ["s01", "s02", "s03"], "injection_order": [1, 2, 3]})
        counts = compute_psm_counts_per_sample(psm, acquisition_queue=queue)
        assert "injection_order" in counts.columns
        assert counts.attrs["provenance"]["n_matched_queue"] == 3


@pytest.mark.skipif(not CARDIO_PSM.exists(), reason="cardio parquet not shipped")
class TestPsmCountsCardio:
    def test_cardio_end_to_end(self):
        psm = pd.read_parquet(CARDIO_PSM)
        queue = load_acquisition_queue(CARDIO_QUEUE)
        counts = compute_psm_counts_per_sample(psm, acquisition_queue=queue)
        # 76 samples
        assert counts.shape[0] == 76
        # At least 3 low-depth samples flagged (empirically ~5 on this data)
        assert int(counts["is_flagged"].sum()) >= 3
        # C8 is one of them -- the RT-drift outlier is also depth-flagged
        c8 = counts.index[counts.index.str.endswith("_C8")]
        assert len(c8) == 1
        assert bool(counts.loc[c8[0], "is_flagged"])
        # Cohort median should be reasonable for a DVP phospho batch
        median = counts.attrs["provenance"]["cohort_median_unique_precursors"]
        assert 10_000 < median < 50_000


# ---------------------------------------------------------------------------
# Contaminant fraction
# ---------------------------------------------------------------------------


def _make_synthetic_contam_df(
    n_samples: int = 6,
    n_real_precursors: int = 200,
    n_contaminant_precursors_per_sample: int | dict[str, int] = 0,
    contaminant_intensity_scale: float = 1e5,
    real_intensity_scale: float = 1e6,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per (sample, precursor) with a real/contaminant flag.

    Contaminant precursors get IDs prefixed with a real-world
    ``CON__`` MaxQuant tag so the prefix rule catches them.  The
    ``PG_ProteinGroups`` value is set to a ``CON__`` accession for
    those rows and a plain UniProt-looking accession for real ones.
    """
    rng = np.random.default_rng(seed)
    if isinstance(n_contaminant_precursors_per_sample, int):
        contam_map = {f"s{i:02d}": n_contaminant_precursors_per_sample for i in range(n_samples)}
    else:
        contam_map = n_contaminant_precursors_per_sample

    rows: list[dict] = []
    for i in range(n_samples):
        sample = f"s{i:02d}"
        # Real precursors: shared IDs across samples for realism
        for p in range(n_real_precursors):
            rows.append(
                {
                    DEFAULT_SAMPLE_COLUMN: sample,
                    DEFAULT_PRECURSOR_COLUMN: f"_p{p:04d}",
                    DEFAULT_PROTEIN_COLUMN: f"P{p % 500:05d}",
                    DEFAULT_INTENSITY_COLUMN: rng.gamma(shape=2.0, scale=real_intensity_scale),
                }
            )
        # Contaminants: distinct IDs per sample
        n_contam = int(contam_map.get(sample, 0))
        for c in range(n_contam):
            rows.append(
                {
                    DEFAULT_SAMPLE_COLUMN: sample,
                    DEFAULT_PRECURSOR_COLUMN: f"_c{c:04d}_{sample}",
                    DEFAULT_PROTEIN_COLUMN: f"CON__KRT{c % 20:02d}",
                    DEFAULT_INTENSITY_COLUMN: rng.gamma(
                        shape=2.0, scale=contaminant_intensity_scale
                    ),
                }
            )
    return pd.DataFrame(rows)


class TestContaminantFractionShape:
    def test_returns_expected_columns(self):
        psm = _make_synthetic_contam_df(n_samples=3, n_contaminant_precursors_per_sample=5)
        contam = compute_contaminant_fraction_per_sample(psm)
        assert set(contam.columns) == {
            "contaminant_intensity",
            "total_intensity",
            "contaminant_fraction",
            "n_contaminant_precursors",
            "n_total_precursors",
            "is_flagged",
        }
        assert contam.shape[0] == 3
        assert (contam["contaminant_fraction"] >= 0).all()
        assert (contam["contaminant_fraction"] <= 1).all()

    def test_missing_column_raises(self):
        psm = _make_synthetic_contam_df(n_samples=2)
        psm = psm.drop(columns=[DEFAULT_INTENSITY_COLUMN])
        with pytest.raises(KeyError, match="missing required columns"):
            compute_contaminant_fraction_per_sample(psm)


class TestContaminantFractionFlagging:
    def test_clean_prep_below_threshold(self):
        # Zero contaminants -> fraction = 0, no flags.
        psm = _make_synthetic_contam_df(n_samples=5, n_contaminant_precursors_per_sample=0)
        contam = compute_contaminant_fraction_per_sample(psm)
        assert (contam["contaminant_fraction"] == 0).all()
        assert not contam["is_flagged"].any()

    def test_heavy_contamination_gets_flagged(self):
        # One sample with a big contaminant load; others clean.
        contam_map = {f"s{i:02d}": 0 for i in range(5)}
        contam_map["s02"] = 400  # ~400 contam precursors vs 200 real
        psm = _make_synthetic_contam_df(
            n_samples=5,
            n_contaminant_precursors_per_sample=contam_map,
            # bias intensity so contaminants dominate the signal
            contaminant_intensity_scale=5e6,
            real_intensity_scale=1e6,
            seed=42,
        )
        contam = compute_contaminant_fraction_per_sample(psm, flag_fraction_threshold=0.10)
        assert bool(contam.loc["s02", "is_flagged"])
        assert contam.loc["s02", "contaminant_fraction"] > 0.5
        # Others clean
        others = contam.drop("s02")
        assert not others["is_flagged"].any()

    def test_threshold_is_respected(self):
        # A moderate contamination should flag under a strict threshold and
        # not flag under a lenient one.
        contam_map = {f"s{i:02d}": 0 for i in range(4)}
        contam_map["s02"] = 30
        psm = _make_synthetic_contam_df(
            n_samples=4,
            n_contaminant_precursors_per_sample=contam_map,
            contaminant_intensity_scale=1e6,
            real_intensity_scale=1e6,
            seed=1,
        )
        strict = compute_contaminant_fraction_per_sample(psm, flag_fraction_threshold=0.05)
        lenient = compute_contaminant_fraction_per_sample(psm, flag_fraction_threshold=0.50)
        assert bool(strict.loc["s02", "is_flagged"])
        assert not bool(lenient.loc["s02", "is_flagged"])


class TestContaminantFractionDetectionMechanisms:
    def test_fasta_lookup_catches_non_prefixed_contaminants(self):
        # Build a df with contaminants whose PG has NO prefix -- just the
        # bare UniProt accession of BSA (P02769, present in the bundled fasta).
        psm = pd.DataFrame(
            {
                DEFAULT_SAMPLE_COLUMN: ["s01"] * 4 + ["s02"] * 4,
                DEFAULT_PRECURSOR_COLUMN: [f"p{i}" for i in range(8)],
                DEFAULT_PROTEIN_COLUMN: [
                    "P00001",  # real
                    "P02769",  # BSA -- in MaxQuant contam fasta
                    "P00002",  # real
                    "P02769",  # BSA
                    "P00003",  # real
                    "P00004",  # real
                    "P00005",  # real
                    "P00006",  # real
                ],
                DEFAULT_INTENSITY_COLUMN: [1e6] * 8,
            }
        )
        contam = compute_contaminant_fraction_per_sample(psm)
        # s01: 2/4 precursors are BSA; s02: 0/4.
        assert contam.loc["s01", "n_contaminant_precursors"] == 2
        assert contam.loc["s02", "n_contaminant_precursors"] == 0
        assert contam.loc["s01", "contaminant_fraction"] == pytest.approx(0.5)

    def test_disable_fasta_falls_back_to_prefix_only(self):
        # Same BSA case but with fasta disabled -> not flagged, because
        # P02769 has no CON__ prefix.
        psm = pd.DataFrame(
            {
                DEFAULT_SAMPLE_COLUMN: ["s01", "s01"],
                DEFAULT_PRECURSOR_COLUMN: ["p1", "p2"],
                DEFAULT_PROTEIN_COLUMN: ["P02769", "P00001"],
                DEFAULT_INTENSITY_COLUMN: [1e6, 1e6],
            }
        )
        contam = compute_contaminant_fraction_per_sample(psm, contaminants_fasta="")
        assert contam.loc["s01", "n_contaminant_precursors"] == 0

    def test_ambiguous_protein_group_kept_as_real_by_default(self):
        # A precursor mapped to "CON__X;P00001" -- ambiguous.  With the
        # default require_all_contaminant=True it's NOT a contaminant.
        psm = pd.DataFrame(
            {
                DEFAULT_SAMPLE_COLUMN: ["s01", "s01"],
                DEFAULT_PRECURSOR_COLUMN: ["p1", "p2"],
                DEFAULT_PROTEIN_COLUMN: ["CON__KRT01;P00001", "P00001"],
                DEFAULT_INTENSITY_COLUMN: [1e6, 1e6],
            }
        )
        conservative = compute_contaminant_fraction_per_sample(psm)
        assert conservative.loc["s01", "n_contaminant_precursors"] == 0

        aggressive = compute_contaminant_fraction_per_sample(psm, require_all_contaminant=False)
        assert aggressive.loc["s01", "n_contaminant_precursors"] == 1


class TestContaminantFractionQueueIntegration:
    def test_queue_adds_injection_order_column(self):
        psm = _make_synthetic_contam_df(n_samples=3, n_contaminant_precursors_per_sample=2)
        queue = pd.DataFrame({"sample": ["s00", "s01", "s02"], "injection_order": [1, 2, 3]})
        contam = compute_contaminant_fraction_per_sample(psm, acquisition_queue=queue)
        assert "injection_order" in contam.columns
        assert contam.attrs["provenance"]["n_matched_queue"] == 3


@pytest.mark.skipif(not CARDIO_PSM.exists(), reason="cardio parquet not shipped")
class TestContaminantFractionCardio:
    def test_cardio_end_to_end(self):
        psm = pd.read_parquet(CARDIO_PSM)
        contam = compute_contaminant_fraction_per_sample(psm)
        # 76 samples
        assert contam.shape[0] == 76
        # Provenance carries the config
        prov = contam.attrs["provenance"]
        assert prov["n_fasta_accessions"] > 100
        assert prov["require_all_contaminant"] is True
        # Cohort median is very low -- IMAC-enriched phospho depletes
        # contaminants strongly on real biological samples.
        assert prov["cohort_median_fraction"] < 0.02
        # The 4 blank runs (G8/G9/G10/G11 -- BLANK_BLANK_BLANK conditions)
        # dominate contaminant intensity because they have almost no peptide
        # signal; each is flagged.
        blank_positions = ["G8", "G9", "G10", "G11"]
        for pos in blank_positions:
            match = contam.index[contam.index.str.endswith(f"_{pos}")]
            assert len(match) == 1, f"expected exactly one sample ending _{pos}"
            assert bool(contam.loc[match[0], "is_flagged"]), (
                f"blank {pos} should be flagged for high contaminant fraction"
            )
            assert contam.loc[match[0], "contaminant_fraction"] > 0.30


# ---------------------------------------------------------------------------
# RT drift (below) + cardio (moved back)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CARDIO_PSM.exists(), reason="cardio parquet not shipped")
class TestCardioIntegration:
    def test_cardio_end_to_end(self):
        psm = pd.read_parquet(CARDIO_PSM)
        queue = load_acquisition_queue(CARDIO_QUEUE)
        drift = compute_retention_time_drift(psm, acquisition_queue=queue)
        # 76 samples in the PSM
        assert drift.shape[0] == 76
        # 75 matched via exact join (1 has the Spectronaut timestamp prefix)
        assert drift.attrs["provenance"]["n_matched_queue"] == 75
        # Spearman ρ vs order should be significant (real drift signal)
        assert drift.attrs["provenance"]["spearman_p_value"] < 0.01
        # C8 sample is a real failed injection: |offset| > 1 min
        c8 = drift.index[drift.index.str.contains("_C8$")]
        assert len(c8) == 1
        assert bool(drift.loc[c8[0], "is_flagged"])
