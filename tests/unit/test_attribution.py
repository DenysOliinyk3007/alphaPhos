"""Synthetic + property tests for alphaphos.preprocess.attribution.

These tests pin the *exact* semantics of the top-N attribution fix so any
future refactor will fail loudly if it changes behavior on known inputs.

Categories:
  - parse_loc_dict       : edge cases of the loc-probability string parser
  - parse_precid_phospho_positions : edge cases of the PrecId parser
  - top_n_positions      : ranking + tie-breaking
  - filter_to_top_n_positions : end-to-end synthetic scenarios + invariants
"""

from __future__ import annotations

import pandas as pd
import pytest

from alphaphos.preprocess.attribution import (
    filter_to_top_n_positions,
    parse_loc_dict,
    parse_precid_phospho_positions,
    top_n_positions,
)


# ============================================================================
# parse_loc_dict
# ============================================================================

class TestParseLocDict:
    def test_single_phospho(self):
        s = "_PEPS[Phospho (STY): 99.5%]TIDE_"
        assert parse_loc_dict(s) == {4: 0.995}

    def test_two_phosphos(self):
        s = "_PVS[Phospho (STY): 92.3%]PS[Phospho (STY): 7.6%]K_"
        d = parse_loc_dict(s)
        assert set(d.keys()) == {3, 5}
        assert d[3] == pytest.approx(0.923)
        assert d[5] == pytest.approx(0.076)

    def test_three_phosphos_mixed_probs(self):
        s = "_S[Phospho (STY): 50.0%]TS[Phospho (STY): 30.0%]VS[Phospho (STY): 20.0%]K_"
        d = parse_loc_dict(s)
        assert d == pytest.approx({1: 0.5, 3: 0.3, 5: 0.2})

    def test_ignores_other_modifications(self):
        s = "_C[Carbamidomethyl (C)]EPS[Phospho (STY): 80.0%]TIDE_"
        # Carbamidomethyl bracket has no leading position-as-amino-acid issue;
        # it occupies position 1 (C). Phospho is at position 4 (S).
        assert parse_loc_dict(s) == {4: 0.8}

    def test_empty_string_returns_empty(self):
        assert parse_loc_dict("") == {}

    def test_none_returns_empty(self):
        assert parse_loc_dict(None) == {}

    def test_nan_returns_empty(self):
        assert parse_loc_dict(float("nan")) == {}

    def test_no_phospho_returns_empty(self):
        assert parse_loc_dict("_PEPTIDE_") == {}

    def test_only_other_ptms_returns_empty(self):
        assert parse_loc_dict("_C[Carbamidomethyl (C)]PEPTIDE_") == {}

    def test_malformed_bracket_no_close_handled(self):
        # Unclosed bracket — function should not raise
        out = parse_loc_dict("_PEPS[Phospho (STY): 50.0_")
        assert isinstance(out, dict)

    def test_percentages_above_99_clamp_to_units(self):
        # 99.99% rounds to ~1.0
        s = "_S[Phospho (STY): 99.99%]K_"
        d = parse_loc_dict(s)
        assert d[1] == pytest.approx(0.9999)

    def test_low_probability(self):
        s = "_S[Phospho (STY): 0.1%]K_"
        assert parse_loc_dict(s) == {1: 0.001}


# ============================================================================
# parse_precid_phospho_positions
# ============================================================================

class TestParsePrecidPhosphoPositions:
    def test_single_phospho(self):
        assert parse_precid_phospho_positions("_PEPS[Phospho (STY)]TIDE_.2") == (4,)

    def test_two_phosphos(self):
        assert parse_precid_phospho_positions(
            "_S[Phospho (STY)]TS[Phospho (STY)]K_.3"
        ) == (1, 3)

    def test_three_phosphos(self):
        assert parse_precid_phospho_positions(
            "_S[Phospho (STY)]TS[Phospho (STY)]VS[Phospho (STY)]K_.3"
        ) == (1, 3, 5)

    def test_phospho_with_other_modifications(self):
        # Carbamidomethyl on C at position 2, phospho on S at position 4
        assert parse_precid_phospho_positions(
            "_AC[Carbamidomethyl (C)]ES[Phospho (STY)]TIDE_.2"
        ) == (4,)

    def test_phospho_at_start(self):
        assert parse_precid_phospho_positions("_S[Phospho (STY)]K_.2") == (1,)

    def test_phospho_at_end_before_terminus(self):
        assert parse_precid_phospho_positions("_PEPS[Phospho (STY)]_.2") == (4,)

    def test_no_phospho(self):
        assert parse_precid_phospho_positions("_PEPTIDE_.2") == ()

    def test_only_other_modification(self):
        assert parse_precid_phospho_positions(
            "_C[Carbamidomethyl (C)]PEPTIDE_.2"
        ) == ()

    def test_none_returns_empty(self):
        assert parse_precid_phospho_positions(None) == ()

    def test_nan_returns_empty(self):
        assert parse_precid_phospho_positions(float("nan")) == ()

    def test_positions_returned_sorted(self):
        # Even if encoded out of order, we sort
        # (in practice they're always in order, but the contract guarantees sort)
        result = parse_precid_phospho_positions(
            "_S[Phospho (STY)]TS[Phospho (STY)]K_.3"
        )
        assert list(result) == sorted(result)


# ============================================================================
# top_n_positions
# ============================================================================

class TestTopNPositions:
    def test_top_1_clear_winner(self):
        assert top_n_positions({3: 0.9, 8: 0.1}, n=1) == (3,)

    def test_top_2_distinct_probs(self):
        assert top_n_positions({3: 0.5, 8: 0.3, 12: 0.2}, n=2) == (3, 8)

    def test_top_3_keeps_all(self):
        assert top_n_positions({3: 0.5, 8: 0.3, 12: 0.2}, n=3) == (3, 8, 12)

    def test_ties_break_by_lower_position(self):
        # R's rank(-prob, ties.method='first') with equal probs gives lower
        # position first; we then sort positions, so the LOWER-position site
        # wins the tie.
        assert top_n_positions({3: 0.5, 8: 0.5, 12: 0.2}, n=1) == (3,)
        assert top_n_positions({3: 0.5, 8: 0.5, 12: 0.5}, n=2) == (3, 8)

    def test_n_zero_returns_empty(self):
        assert top_n_positions({3: 0.5}, n=0) == ()

    def test_n_negative_returns_empty(self):
        assert top_n_positions({3: 0.5}, n=-1) == ()

    def test_empty_dict_returns_empty(self):
        assert top_n_positions({}, n=2) == ()

    def test_n_larger_than_dict_returns_all(self):
        assert top_n_positions({3: 0.5, 8: 0.3}, n=5) == (3, 8)

    def test_result_always_sorted_ascending(self):
        result = top_n_positions({3: 0.5, 8: 0.7, 12: 0.6}, n=3)
        assert list(result) == sorted(result)


# ============================================================================
# filter_to_top_n_positions — synthetic end-to-end cases
# ============================================================================

def _row(
    precid: str,
    loc_string: str,
    *,
    qty: float = 1000.0,
    sample: str = "s1",
    charge: int = 2,
    pep: str | None = None,
    pep_pos: str = "1",
    genes: str = "GENE",
    prots: str = "P12345",
) -> dict:
    """Helper to build a synthetic precursor row dict."""
    return {
        "R.FileName": sample,
        "EG.PrecursorId": precid,
        "EG.PTMLocalizationProbabilities": loc_string,
        "EG.PTMAssayProbability": 0.99,
        "EG.TotalQuantity (Settings)": qty,
        "PEP.PeptidePosition": pep_pos,
        "PEP.StrippedSequence": pep or precid.strip("_*. ").translate(
            str.maketrans("", "", "[]")
        ).replace("Phospho (STY)", ""),
        "PG.Genes": genes,
        "PG.ProteinGroups": prots,
        "FG.Charge": charge,
    }


class TestFilterEndToEnd:
    def test_single_unambiguous_phospho_kept(self):
        # One row, one phospho with 99% loc — must pass
        df = pd.DataFrame([_row(
            "_PEPS[Phospho (STY)]TIDE_.2",
            "_PEPS[Phospho (STY): 99.0%]TIDE_",
        )])
        out = filter_to_top_n_positions(df)
        assert len(out) == 1

    def test_ambiguous_top_match_kept_others_dropped(self):
        # Same peptide exported 3 times with phospho at 3 different positions,
        # loc string says S322 is the top-1. Only the row with PrecId at S322 survives.
        loc = "_PRS[Phospho (STY): 1.0%]PS[Phospho (STY): 98.0%]K[Phospho (STY): 1.0%]_"
        rows = [
            _row("_PRS[Phospho (STY)]PSK_.2", loc),         # phospho at pos 3 -- non-top
            _row("_PRSPS[Phospho (STY)]K_.2", loc),         # phospho at pos 5 -- TOP
            _row("_PRSPSK[Phospho (STY)]_.2", loc),         # phospho at pos 6 -- non-top
        ]
        df = pd.DataFrame(rows)
        out = filter_to_top_n_positions(df)
        assert len(out) == 1
        assert out["EG.PrecursorId"].iloc[0] == "_PRSPS[Phospho (STY)]PSK_.2".replace(
            "PS[Phospho (STY)]PS", "PS[Phospho (STY)]K_.2".replace("K_.2", "K")
        ) or out["EG.PrecursorId"].iloc[0] == "_PRSPS[Phospho (STY)]K_.2"
        # Simpler assertion
        assert out["EG.PrecursorId"].iloc[0] == "_PRSPS[Phospho (STY)]K_.2"

    def test_multi_phospho_M2_top_2_match_kept(self):
        # M2 peptide. Loc string says positions 1 and 3 are the top 2.
        loc = ("_S[Phospho (STY): 99.0%]TS[Phospho (STY): 99.0%]"
               "VS[Phospho (STY): 1.0%]K_")
        rows = [
            # PrecId puts phospho at positions 1 and 3 — TOP
            _row("_S[Phospho (STY)]TS[Phospho (STY)]VSK_.3", loc),
            # PrecId puts phospho at positions 1 and 5 — wrong second site
            _row("_S[Phospho (STY)]TSVS[Phospho (STY)]K_.3", loc),
            # PrecId puts phospho at positions 3 and 5 — wrong both
            _row("_STS[Phospho (STY)]VS[Phospho (STY)]K_.3", loc),
        ]
        df = pd.DataFrame(rows)
        out = filter_to_top_n_positions(df)
        assert len(out) == 1
        assert "_S[Phospho (STY)]TS[Phospho (STY)]VSK_.3" in out["EG.PrecursorId"].values

    def test_multi_phospho_M3_top_3_match_kept(self):
        # M3 peptide — all three positions confident; PrecId must match exactly.
        loc = ("_S[Phospho (STY): 99.0%]TS[Phospho (STY): 99.0%]"
               "VS[Phospho (STY): 99.0%]K_")
        rows = [
            _row("_S[Phospho (STY)]TS[Phospho (STY)]VS[Phospho (STY)]K_.3", loc),
        ]
        df = pd.DataFrame(rows)
        out = filter_to_top_n_positions(df)
        assert len(out) == 1

    def test_non_phospho_row_passthrough(self):
        # A row carrying only Carbamidomethyl — no phospho. Should pass through.
        rows = [
            _row("_C[Carbamidomethyl (C)]PEPTIDE_.2", "_PEPTIDE_"),
        ]
        df = pd.DataFrame(rows)
        out = filter_to_top_n_positions(df)
        assert len(out) == 1

    def test_phospho_row_missing_loc_string_dropped(self):
        # Phospho row but loc string is None / empty — we can't validate, drop it
        rows = [
            _row("_PEPS[Phospho (STY)]TIDE_.2", ""),
        ]
        df = pd.DataFrame(rows)
        out = filter_to_top_n_positions(df)
        assert len(out) == 0

    def test_phospho_row_loc_string_nan_dropped(self):
        rows = [_row("_PEPS[Phospho (STY)]TIDE_.2", "")]
        df = pd.DataFrame(rows)
        df.loc[0, "EG.PTMLocalizationProbabilities"] = pd.NA
        out = filter_to_top_n_positions(df)
        assert len(out) == 0

    def test_charge_states_evaluated_independently(self):
        # Same peptide, two charges, both with top-1 = position 3.
        # Both should be kept (they're separate observations).
        loc = "_PEP[Phospho (STY): 1.0%]S[Phospho (STY): 98.0%]K[Phospho (STY): 1.0%]_"
        rows = [
            _row("_PEPS[Phospho (STY)]K_.2", loc, charge=2),
            _row("_PEPS[Phospho (STY)]K_.3", loc, charge=3),
        ]
        df = pd.DataFrame(rows)
        out = filter_to_top_n_positions(df)
        # Wait — both rows have phospho at position 4 (S), and top-1 is also position 4.
        # So both pass.
        assert len(out) == 2

    def test_per_sample_loc_independence(self):
        # Same peptide in 2 samples with DIFFERENT per-sample loc strings.
        # Sample s1 has phospho at top position 3, sample s2 has it at top position 5.
        # Both rows reflect the correct top position for THEIR sample, so both survive.
        loc_s1 = "_PRS[Phospho (STY): 95.0%]PS[Phospho (STY): 5.0%]K_"
        loc_s2 = "_PRS[Phospho (STY): 5.0%]PS[Phospho (STY): 95.0%]K_"
        rows = [
            _row("_PRS[Phospho (STY)]PSK_.2", loc_s1, sample="s1"),  # top1=3 ✓
            _row("_PRSPS[Phospho (STY)]K_.2", loc_s2, sample="s2"),  # top1=5 ✓
        ]
        df = pd.DataFrame(rows)
        out = filter_to_top_n_positions(df)
        assert len(out) == 2

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=[
            "EG.PrecursorId", "EG.PTMLocalizationProbabilities"
        ])
        out = filter_to_top_n_positions(df)
        assert len(out) == 0

    def test_missing_required_column_raises(self):
        df = pd.DataFrame([{"EG.PrecursorId": "_S[Phospho (STY)]_.2"}])
        # Missing EG.PTMLocalizationProbabilities
        with pytest.raises(KeyError, match="EG.PTMLocalizationProbabilities"):
            filter_to_top_n_positions(df)

    def test_missing_precid_column_raises(self):
        df = pd.DataFrame([{"EG.PTMLocalizationProbabilities": "_S_"}])
        with pytest.raises(KeyError, match="EG.PrecursorId"):
            filter_to_top_n_positions(df)


# ============================================================================
# Invariants / properties (manual property-style assertions, no hypothesis lib)
# ============================================================================

class TestFilterInvariants:
    def _sample_dataset(self) -> pd.DataFrame:
        """A mid-sized synthetic dataset mixing all the scenarios."""
        loc_clear = "_PEPS[Phospho (STY): 99.0%]TIDE_"
        loc_ambig = "_PRS[Phospho (STY): 50.0%]PS[Phospho (STY): 50.0%]K_"
        loc_M2 = ("_S[Phospho (STY): 99.0%]TS[Phospho (STY): 99.0%]"
                  "VSK_")
        rows = []
        # 10 unambiguous phospho rows (should all survive)
        for i in range(10):
            rows.append(_row("_PEPS[Phospho (STY)]TIDE_.2", loc_clear,
                             qty=1000 + i, sample=f"s{i % 3 + 1}"))
        # 6 ambiguous (3 candidate positions x 2 samples) - only 2 should survive
        for sample in ["s1", "s2"]:
            for pos in [3, 5]:
                if pos == 3:
                    precid = "_PRS[Phospho (STY)]PSK_.2"
                else:
                    precid = "_PRSPS[Phospho (STY)]K_.2"
                rows.append(_row(precid, loc_ambig, sample=sample))
            # also add the wrong-tie-breaker row (loc 5) - both have prob 0.5
            # with our tie-break, position 3 wins; so the pos=5 row drops
        # 4 M2 rows: 2 correct, 2 wrong
        for sample in ["s1", "s2"]:
            rows.append(_row(
                "_S[Phospho (STY)]TS[Phospho (STY)]VSK_.3", loc_M2, sample=sample
            ))  # top-2 = (1, 3) — kept
            rows.append(_row(
                "_S[Phospho (STY)]TSVS[Phospho (STY)]K_.3", loc_M2, sample=sample
            ))  # top-2 wrong — dropped
        # 3 non-phospho passthrough
        for i in range(3):
            rows.append(_row(
                "_C[Carbamidomethyl (C)]PEPTIDE_.2", "_PEPTIDE_",
                sample=f"s{i + 1}"
            ))
        return pd.DataFrame(rows)

    def test_idempotent(self):
        df = self._sample_dataset()
        once = filter_to_top_n_positions(df)
        twice = filter_to_top_n_positions(once)
        pd.testing.assert_frame_equal(once, twice)

    def test_deterministic(self):
        df = self._sample_dataset()
        out1 = filter_to_top_n_positions(df)
        out2 = filter_to_top_n_positions(df)
        pd.testing.assert_frame_equal(out1, out2)

    def test_filter_never_adds_rows(self):
        df = self._sample_dataset()
        out = filter_to_top_n_positions(df)
        assert len(out) <= len(df)

    def test_filter_preserves_column_set(self):
        df = self._sample_dataset()
        out = filter_to_top_n_positions(df)
        assert set(out.columns) == set(df.columns)

    def test_filter_preserves_dtypes_for_retained_columns(self):
        df = self._sample_dataset()
        out = filter_to_top_n_positions(df)
        for col in out.columns:
            assert out[col].dtype == df[col].dtype, (
                f"{col}: out dtype {out[col].dtype} != input dtype {df[col].dtype}"
            )

    def test_ambiguous_phospho_collapses_to_one_per_pep_charge_sample(self):
        # For AMBIGUOUS phospho peptides (multiple candidate-position rows for
        # the same chromatographic measurement), after filtering exactly ONE
        # candidate-position row should survive per (peptide, charge, sample).
        loc_ambig = ("_PRS[Phospho (STY): 1.0%]PS[Phospho (STY): 98.0%]"
                     "K[Phospho (STY): 1.0%]_")
        rows = []
        # 3 candidate-position rows x 2 samples = 6 input rows.
        # Only the position-5 (top-1) row per sample should survive.
        for sample in ["s1", "s2"]:
            for precid in [
                "_PRS[Phospho (STY)]PSK_.2",   # phospho at position 3 — non-top
                "_PRSPS[Phospho (STY)]K_.2",   # phospho at position 5 — TOP
                "_PRSPSK[Phospho (STY)]_.2",   # phospho at position 6 — non-top
            ]:
                rows.append(_row(precid, loc_ambig, sample=sample))
        df = pd.DataFrame(rows)
        out = filter_to_top_n_positions(df)
        # Group only the ambiguous phospho survivors
        grp = out.groupby(
            ["PEP.StrippedSequence", "FG.Charge", "R.FileName"]
        ).size()
        assert (grp == 1).all(), (
            f"each (pep, charge, sample) ambiguous group should yield exactly 1 row; got {grp.to_dict()}"
        )
        assert len(out) == 2  # 2 samples x 1 survivor each

    def test_all_phospho_rows_have_valid_top_n_match(self):
        # After filtering, every phospho row's PrecId positions should equal
        # the top-N from its loc string.
        df = self._sample_dataset()
        out = filter_to_top_n_positions(df)
        for _, row in out.iterrows():
            precid_pos = parse_precid_phospho_positions(row["EG.PrecursorId"])
            if not precid_pos:
                continue  # non-phospho passthrough
            loc = parse_loc_dict(row["EG.PTMLocalizationProbabilities"])
            expected = top_n_positions(loc, len(precid_pos))
            assert precid_pos == expected, (
                f"Row violated top-N: precid={precid_pos}, expected={expected}, "
                f"prec={row['EG.PrecursorId']!r}, loc={row['EG.PTMLocalizationProbabilities']!r}"
            )

    def test_non_phospho_rows_always_pass_through(self):
        df = self._sample_dataset()
        out = filter_to_top_n_positions(df)
        # Count non-phospho rows in input and output
        n_in = (~df["EG.PrecursorId"].str.contains(r"\[Phospho \(STY\)\]")).sum()
        n_out = (~out["EG.PrecursorId"].str.contains(r"\[Phospho \(STY\)\]")).sum()
        assert n_in == n_out, "Filter dropped non-phospho rows"

    def test_index_is_reset(self):
        df = self._sample_dataset()
        out = filter_to_top_n_positions(df)
        assert list(out.index) == list(range(len(out)))

    def test_does_not_mutate_input(self):
        df = self._sample_dataset()
        df_before = df.copy(deep=True)
        _ = filter_to_top_n_positions(df)
        pd.testing.assert_frame_equal(df, df_before)
