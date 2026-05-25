"""Top-N peptide-to-site attribution.

Spectronaut exports the same phospho-peptide measurement as multiple precursor
rows — one per candidate localization position — all with similar (or identical)
intensities for the same underlying chromatographic peak. Spectronaut's native
PTM consolidation (the Hogrebe R script embedded in `PluginPeptideCollapse.dll`)
dedups these by:

    1. Reading per-row, per-position localization probabilities from
       `EG.PTMLocalizationProbabilities`,
    2. Selecting the **top-N positions by probability** (N = number of phospho
       groups on the peptide, parsed from `EG.PrecursorId`),
    3. Treating the peptide as if it carried phospho **only** at those top-N
       positions, regardless of which `[Phospho (STY)]` bracket positions the
       PrecursorId string happens to encode.

`PeptideCollapse_v4` (the canonical legacy Python pipeline) instead derives
positions from `EG.PrecursorId` directly — so for the typical N=20 candidate-
position rows of an ambiguously localized peptide it produces 20 separate site
attributions, each receiving the full peptide quant. Downstream this inflates
quants at all the non-top sites by 5-10x or more.

`filter_to_top_n_positions(df)` reproduces Spectronaut's dedup behavior in
Python by keeping only the rows whose PrecId-encoded positions exactly match
the top-N set from the per-row loc probabilities. This is a single linear pass
over the dataframe; the filter retains ~40-60% of rows on typical Spectronaut
exports.

Validated against Spectronaut's native PTM site report on `nanoPhos_dilser_noEGF_1000ng`:
Pearson r = 0.98 (log2) / 0.99 (linear) per cell, r = 0.989 per-site median;
mean per-cell log2 difference reduced from +0.36 (baseline) to +0.056 (fixed).
See `docs/design/compare_to_spectronaut_native.py` and
`docs/design/prototype_top_n_attribution.py`.
"""

from __future__ import annotations

import re

import pandas as pd


def parse_loc_dict(s: str | float) -> dict[int, float]:
    """Parse `EG.PTMLocalizationProbabilities` -> {peptide_position_1indexed: prob}.

    Returns an empty dict for non-string / missing / unparseable input. Only
    `[Phospho (STY)]` annotations are extracted; other PTM types are ignored.
    Positions are 1-indexed amino-acid offsets into the peptide. Probabilities
    are returned in [0, 1] (Spectronaut writes percentages, we divide by 100).

    Example
    -------
    >>> parse_loc_dict("_PVS[Phospho (STY): 92.3%]PS[Phospho (STY): 7.6%]_")
    {3: 0.923, 5: 0.076}
    """
    if not isinstance(s, str):
        return {}
    s = s.strip("_.* ")
    out: dict[int, float] = {}
    pos = 0
    i = 0
    while i < len(s):
        if s[i] == "[":
            try:
                end = s.index("]", i)
            except ValueError:
                break
            bracket = s[i + 1 : end]
            if bracket.startswith("Phospho (STY)"):
                m = re.search(r":\s*([\d.]+)%", bracket)
                if m:
                    out[pos] = float(m.group(1)) / 100.0
            i = end + 1
        elif s[i].isalpha():
            pos += 1
            i += 1
        else:
            i += 1
    return out


def parse_precid_phospho_positions(precid: str | float) -> tuple[int, ...]:
    """Return the sorted tuple of phospho positions encoded in `EG.PrecursorId`.

    `EG.PrecursorId` looks like ``_SOME[Carbamidomethyl (C)]PEP[Phospho (STY)]TIDE_.2``.
    We strip non-phospho brackets, the wrapper underscores, and the trailing
    ``.charge``, then locate every `[Phospho (STY)]` marker and report its
    1-indexed amino-acid position.

    Returns an empty tuple for missing / non-phospho input.

    Example
    -------
    >>> parse_precid_phospho_positions("_S[Phospho (STY)]TS[Phospho (STY)]K_.3")
    (1, 3)
    """
    if not isinstance(precid, str):
        return ()
    only_phos = re.sub(r"\[(?!Phospho \(STY\))[^\]]*\]", "", precid)
    only_phos = only_phos.strip("_*. ")
    only_phos = re.sub(r"\.\d+$", "", only_phos)  # trailing .charge
    parts = only_phos.split("[Phospho (STY)]")
    if len(parts) < 2:
        return ()
    positions: list[int] = []
    cur = 0
    for seg in parts[:-1]:
        cur += len(seg)
        positions.append(cur)
    return tuple(sorted(positions))


def top_n_positions(loc_dict: dict[int, float], n: int) -> tuple[int, ...]:
    """Return the top-N positions by descending probability, ties broken by
    ascending position. Matches R's ``rank(-prob, ties.method='first')`` then
    take the first N.

    Returns an empty tuple if N <= 0 or loc_dict is empty.

    Example
    -------
    >>> top_n_positions({3: 0.5, 8: 0.5, 12: 0.2}, n=2)
    (3, 8)
    >>> top_n_positions({3: 0.9, 8: 0.1}, n=1)
    (3,)
    """
    if n <= 0 or not loc_dict:
        return ()
    ranked = sorted(loc_dict.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return tuple(sorted(p for p, _ in ranked))


def filter_to_top_n_positions(
    df: pd.DataFrame,
    *,
    precid_column: str = "EG.PrecursorId",
    loc_string_column: str = "EG.PTMLocalizationProbabilities",
) -> pd.DataFrame:
    """Drop precursor rows whose `EG.PrecursorId` positions don't equal the
    top-N positions from the per-row localization probability string.

    Rows are kept when:
      - They carry no phospho (passthrough — they're not consolidated anyway), or
      - Their PrecId-encoded phospho positions exactly equal the top-N (where
        N = number of phospho groups on the peptide) selected from
        `EG.PTMLocalizationProbabilities`.

    Returns a new DataFrame with the same columns; the index is reset.

    Notes
    -----
    The filter is **idempotent** — applying it twice gives the same result as
    applying it once — and **deterministic** (no random state).

    On a typical Spectronaut export this retains ~40-60% of rows. Performance
    is linear in row count; ~1-6 seconds for 270k-1.3M rows on a modern laptop.
    """
    if precid_column not in df.columns:
        raise KeyError(f"Required column missing: {precid_column!r}")
    if loc_string_column not in df.columns:
        raise KeyError(f"Required column missing: {loc_string_column!r}")

    loc_dicts = df[loc_string_column].map(parse_loc_dict)
    precid_positions = df[precid_column].map(parse_precid_phospho_positions)

    def _is_top_n(loc: dict[int, float], precid_pos: tuple[int, ...]) -> bool:
        n = len(precid_pos)
        if n == 0:
            # Non-phospho row: passthrough (we don't touch it)
            return True
        if not loc:
            # Phospho row with no parseable loc string: drop (can't validate)
            return False
        return top_n_positions(loc, n) == precid_pos

    keep = [_is_top_n(loc, pp) for loc, pp in zip(loc_dicts, precid_positions)]
    return df.loc[keep].reset_index(drop=True)
