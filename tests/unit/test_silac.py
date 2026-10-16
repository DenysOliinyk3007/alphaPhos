"""Multi-channel (SILAC) support: channel assignment, guards, run-grid Class-I.

The behaviours that matter and that nothing else covers:

* ``FG.Id`` is the only carrier of channel identity, so a labelled report
  exported without it must fail loudly rather than sum its channels.
* A channel is a SAMPLE, so collapse must emit ``runs x channels`` samples.
* Localisation is assigned once per RUN and shared by that run's channels, so
  the Wilson Class-I bound must be computed on the run grid. Counting channels
  as independent observations is anti-conservative and is the error this
  module exists to prevent.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import alphaphos as ap
from alphaphos.io.silac import (
    DEFAULT_LABEL_CHANNELS,
    derive_channel,
    detect_multichannel,
    validate_label_channels,
)
from alphaphos.io.spectronaut import _assign_channels
from alphaphos.preprocess.classI_wilson import wilson_lower_bound

RUNS = ["runA", "runB", "runC"]
CHANNELS = ["light", "medium", "heavy"]
# Spectronaut writes the label as a residue tag inside FG.Id; light carries none.
TAG = {"light": "", "medium": "[Arg6]", "heavy": "[Arg10]"}


def _silac_psm(loc_prob_by_run: dict[str, float] | None = None) -> pd.DataFrame:
    """Two sites x 3 runs x 3 channels, already carrying a derived Channel column.

    ``loc_prob_by_run`` overrides the localisation probability per run so a
    test can make a site Class-I in some runs only. The probability is a
    property of the run, not the channel, which is what the real data shows.
    """
    loc_prob_by_run = loc_prob_by_run or dict.fromkeys(RUNS, 0.95)
    rows = []
    for run in RUNS:
        prob = loc_prob_by_run[run]
        pct = f"{prob * 100:.1f}"
        for channel in CHANNELS:
            for seq, pos, gene, acc, quant in (
                ("_ASGQGS[Phospho (STY)]PGVK_", "570", "SIK1B", "A0A0B4J2F2", 2000.0),
                ("_LLS[Phospho (STY)]PGVTR_", "120", "AKT1", "P31749", 800.0),
            ):
                rows.append(
                    {
                        "R.FileName": run,
                        "EG.PrecursorId": f"{seq}.2",
                        "FG.Id": f"{seq[:-1]}{TAG[channel]}_.2",
                        "Channel": channel,
                        "EG.TotalQuantity (Settings)": quant,
                        "PEP.PeptidePosition": pos,
                        "EG.PTMAssayProbability": prob,
                        "EG.PTMLocalizationProbabilities": seq.replace(
                            "[Phospho (STY)]", f"[Phospho (STY): {pct}%]"
                        ),
                        "PG.Genes": gene,
                        "PG.ProteinGroups": acc,
                    }
                )
    return pd.DataFrame(rows)


def _collapse(df: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ap.collapse_sites(df, advanced={"localization_strategy": "global_max"})


# ---------------------------------------------------------------------------
# Channel assignment
# ---------------------------------------------------------------------------


def test_derive_channel_reads_the_label_tag_not_the_precursor_id():
    fg_ids = pd.Series(
        [
            "_AAAAS[Phospho (STY)]PGPAQPGPR_.2",
            "_AAAAS[Phospho (STY)]PGPAQPGPR[Arg6]_.2",
            "_AAAAS[Phospho (STY)]PGPAQPGPR[Arg10]_.2",
            "_TENLLLDGNM[Oxidation (M)]DIK[Lys8]_.2",
            "_EDQEATGPR[Arg6]LS[Phospho (STY)]K[Lys4]_.2",
        ]
    )
    assert list(derive_channel(fg_ids)) == ["light", "medium", "heavy", "heavy", "medium"]


def test_derive_channel_rejects_a_precursor_spanning_two_channels():
    # Arg6 is medium and Lys8 is heavy: one precursor cannot be both, and
    # silently picking one would corrupt every ratio it contributes to.
    fg_ids = pd.Series(["_SAMPLER[Arg6]PEPTIDEK[Lys8]_.2"])
    with pytest.raises(ValueError, match="more than one"):
        derive_channel(fg_ids)


def test_detect_multichannel_is_false_for_a_label_free_report():
    df = pd.DataFrame({"FG.Id": ["_PEPTIDEK_.2", "_SAMPLER_.3"]})
    assert detect_multichannel(df) == set()


def test_validate_label_channels_rejects_mapping_a_tag_to_light():
    # Light is defined by the ABSENCE of a tag; an explicit entry would make
    # a labelled precursor indistinguishable from an unlabelled one.
    with pytest.raises(ValueError, match="ABSENCE"):
        validate_label_channels({"Arg0": "light"})


@pytest.mark.parametrize("bad", [{}, None, {"Arg6": ""}, {"": "heavy"}])
def test_validate_label_channels_rejects_malformed_mappings(bad):
    with pytest.raises(ValueError):
        validate_label_channels(bad)


def test_custom_label_channels_support_a_two_plex():
    fg_ids = pd.Series(["_PEPTIDEK_.2", "_PEPTIDEK[Lys8]_.2"])
    out = derive_channel(fg_ids, {"Lys8": "heavy"})
    assert list(out) == ["light", "heavy"]


# ---------------------------------------------------------------------------
# The guard: labelled report exported without FG.Id
# ---------------------------------------------------------------------------


def test_duplicate_precursors_without_fg_id_raise_instead_of_merging_silently():
    # This is exactly what a SILAC report looks like when FG.Id is left out of
    # the export: several real rows per run and precursor, nothing to tell them
    # apart. Summing them would average the channels together.
    df = _silac_psm().drop(columns=["FG.Id", "Channel"])
    with pytest.raises(ValueError, match="FG.Id"):
        _assign_channels(df, None)


def test_single_channel_report_without_fg_id_is_left_alone():
    df = _silac_psm()
    df = df[df["Channel"] == "light"].drop(columns=["FG.Id", "Channel"])
    assert _assign_channels(df, None) is None


def test_assign_channels_reports_per_channel_counts():
    df = _silac_psm().drop(columns=["Channel"])
    counts = _assign_channels(df, None)
    assert counts == {"light": 6, "medium": 6, "heavy": 6}
    assert list(df["Channel"].unique()) == ["light", "medium", "heavy"]


# ---------------------------------------------------------------------------
# Collapse: a channel is a sample
# ---------------------------------------------------------------------------


def test_collapse_emits_one_sample_per_run_and_channel():
    adata = _collapse(_silac_psm())
    assert adata.n_obs == len(RUNS) * len(CHANNELS)
    assert set(adata.obs["run"]) == set(RUNS)
    assert set(adata.obs["channel"]) == set(CHANNELS)
    assert all(ap.constants.SAMPLE_CHANNEL_SEP in s for s in adata.obs_names)


def test_collapse_keeps_channel_intensities_separate():
    df = _silac_psm()
    # Make heavy four-fold the light intensity for every row.
    heavy = df["Channel"] == "heavy"
    df.loc[heavy, "EG.TotalQuantity (Settings)"] *= 4
    adata = _collapse(df)
    by_channel = (
        pd.DataFrame(adata.X, index=adata.obs["channel"].to_numpy()).groupby(level=0).mean()
    )
    # X is log2, so a 4x ratio is +2.
    assert (by_channel.loc["heavy"] - by_channel.loc["light"]).round(6).eq(2.0).all()


def test_condition_df_naming_runs_instead_of_channels_raises():
    df = _silac_psm()
    condition_df = pd.DataFrame({"sample": RUNS, "condition": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="belongs to the channel"):
        ap.collapse_sites(df, condition_df=condition_df)


# ---------------------------------------------------------------------------
# The reason this module exists: run-grid Class-I evidence
# ---------------------------------------------------------------------------


def test_wilson_bound_uses_the_run_grid_not_the_channel_grid():
    adata = _collapse(_silac_psm())

    assert (adata.var["n_samples_detected"] == 9).all()  # 3 runs x 3 channels
    assert (adata.var["n_runs_detected"] == 3).all()  # the real evidence count
    assert (adata.var["n_classI_runs"] == 3).all()

    # The bound must match 3/3, NOT the inflated 9/9 a channel-grid count gives.
    expected = wilson_lower_bound([3], [3])[0]
    inflated = wilson_lower_bound([9], [9])[0]
    assert adata.var["classI_wilson_lb"].round(6).eq(round(expected, 6)).all()
    assert expected < inflated  # the error being prevented is real, not cosmetic


def test_partial_class_i_evidence_is_counted_once_per_run():
    # Class-I in one run of three. Channels must not turn that into 3 of 9.
    adata = _collapse(_silac_psm(loc_prob_by_run={"runA": 0.95, "runB": 0.40, "runC": 0.40}))
    assert (adata.var["n_classI_runs"] == 1).all()
    assert (adata.var["n_runs_detected"] == 3).all()
    assert (
        adata.var["classI_wilson_lb"].round(6).eq(round(wilson_lower_bound([1], [3])[0], 6)).all()
    )


def test_single_channel_collapse_has_no_run_grid_columns():
    # Without channels the sample grid IS the run grid; adding the columns
    # would imply a distinction that does not exist.
    df = _silac_psm()
    df = df[df["Channel"] == "light"].drop(columns=["FG.Id", "Channel"])
    adata = _collapse(df)
    assert "n_runs_detected" not in adata.var
    assert "run" not in adata.obs
    assert adata.n_obs == len(RUNS)
