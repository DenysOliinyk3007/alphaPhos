"""Parse Spectronaut PSM string columns into structured Python data.

Spectronaut encodes phospho information across two column families:

* ``EG.PrecursorId`` -- the modified sequence in "bracket notation", e.g.
  ``_S[Phospho (STY)]TS[Phospho (STY)]K_.3``. The underscores wrap the peptide,
  ``[Phospho (STY)]`` markers sit immediately AFTER the modified residue, and
  a trailing ``.<charge>`` tag identifies the precursor charge state. Other
  brackets (``[Carbamidomethyl (C)]``, ``[Acetyl (Protein N-term)]``, ...) may
  interleave freely.

* ``EG.PTMLocalizationProbabilities`` -- the same sequence but the phospho
  bracket carries a per-position probability, e.g.
  ``_PVS[Phospho (STY): 92.3%]PS[Phospho (STY): 7.6%]_``. Probabilities sum
  across candidate positions on the peptide to 100%. This column is used to
  derive per-(site, run) localization scores rather than the joint peptide-
  level ``EG.PTMAssayProbability``.

Everything here is **pure**: no side effects, no state, no logging. Each
function takes a string (or a NaN) and returns Python primitives.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# EG.PrecursorId parsing
# ---------------------------------------------------------------------------


def extract_sequence_modifications(sequence: str) -> dict[str, Any]:
    """Parse a Spectronaut ``EG.PrecursorId`` string.

    The returned dict has the following keys::

        clean_sequence   -- the amino-acid sequence with ALL brackets removed,
                            e.g. "STSK" from "_S[Phospho (STY)]TS[Phospho (STY)]K_.3"
        phospho_positions -- 1-indexed positions of phospho residues within
                             the clean sequence: [1, 3] in the example above.
        phospho_count     -- number of ``[Phospho (STY)]`` markers.
        all_modifications -- every bracket content in order of appearance
                             (used later to build peptide-collapse keys).
        phospho_sequence  -- the sequence with ONLY phospho brackets preserved,
                             other mods stripped. Used as the input to
                             :func:`calculate_phospho_positions` when re-deriving
                             positions from a stripped view.
        base_sequence     -- the sequence with wrapper underscores and the
                             trailing charge suffix removed, but ALL brackets
                             preserved. Used for regex-based mod extraction.

    Parameters
    ----------
    sequence : str
        A Spectronaut precursor id string. Common shapes:

        * ``"_PEPTIDE_.2"`` -- unmodified, charge 2.
        * ``"_S[Phospho (STY)]TSK_.3"`` -- single phospho.
        * ``"*.PS[Phospho (STY)]T.*"`` -- alternative wrapper convention.

    Returns
    -------
    dict[str, Any]
        Always populated; empty lists / 0 counts for non-phospho / unparseable
        input. Never raises.

    Notes
    -----
    Wrapper stripping handles four historical Spectronaut conventions
    (``_..._``, ``*..*``, ``_..._.charge``, ``*...*.charge``); if none match,
    the input is used as-is.
    """
    base_sequence = sequence

    # Strip wrapper underscores/stars and any trailing .charge tag.
    if len(sequence) >= 4:
        if sequence.startswith("*.") or sequence.startswith("_."):
            base_sequence = base_sequence[2:]
        elif sequence.startswith("*") or sequence.startswith("_"):
            base_sequence = base_sequence[1:]

        if ".*" in base_sequence:
            base_sequence = base_sequence.split(".*")[0]
        elif "._" in base_sequence:
            base_sequence = base_sequence.split("._")[0]
        elif base_sequence.endswith("*") or base_sequence.endswith("_"):
            base_sequence = base_sequence[:-1]

        # Drop trailing ``.<charge>`` (an integer suffix after a dot).
        parts = base_sequence.split(".")
        if len(parts) > 1 and parts[-1].isdigit():
            base_sequence = ".".join(parts[:-1])

    # Delete non-phospho brackets so we can re-derive per-residue positions
    # against a sequence that only carries the phospho markers.
    non_phospho_bracket = r"\[(?!Phospho \(STY\))[^\]]*\]"
    phospho_only_sequence = re.sub(non_phospho_bracket, "", base_sequence)

    # Fully clean sequence -- every bracket and every underscore removed.
    clean_sequence = re.sub(r"\[[^\]]*\]", "", base_sequence)
    clean_sequence = clean_sequence.replace("_", "")

    # Count phospho markers on the ORIGINAL sequence (safest -- immune to any
    # weird wrapper mishandling above).
    phospho_count = len(sequence.split("[Phospho (STY)]")) - 1
    all_modifications = re.findall(r"\[([^\]]+)\]", base_sequence)

    phospho_positions: list[int] = []
    if phospho_count > 0:
        phospho_positions = calculate_phospho_positions(phospho_only_sequence)

    return {
        "clean_sequence": clean_sequence,
        "phospho_positions": phospho_positions,
        "phospho_count": phospho_count,
        "all_modifications": all_modifications,
        "phospho_sequence": phospho_only_sequence,
        "base_sequence": base_sequence,
    }


def calculate_phospho_positions(phospho_sequence: str) -> list[int]:
    """Return 1-indexed positions of ``[Phospho (STY)]`` markers.

    Assumes ``phospho_sequence`` contains ONLY phospho brackets (any other
    mods must have been stripped upstream; see
    :func:`extract_sequence_modifications`).

    The bracket marker is placed IMMEDIATELY AFTER the modified residue in
    Spectronaut's convention, so the reported position is the length of the
    sequence read so far (equivalently the 1-indexed AA position of the
    residue that carries the phospho).

    Examples
    --------
    >>> calculate_phospho_positions("S[Phospho (STY)]TS[Phospho (STY)]K")
    [1, 3]
    """
    if "[Phospho (STY)]" not in phospho_sequence:
        return []

    segments = phospho_sequence.split("[Phospho (STY)]")
    positions: list[int] = []
    current_pos = 0
    # Each split segment (except the last) contributes its length + the length
    # of everything before it -- so we accumulate positions as running sums of
    # segment lengths.
    for i in range(len(segments) - 1):
        current_pos += len(segments[i])
        positions.append(current_pos)
    return positions


# ---------------------------------------------------------------------------
# PEP.PeptidePosition parsing
# ---------------------------------------------------------------------------


def extract_first_valid_position(position_str: Any) -> int | None:
    """Parse a Spectronaut ``PEP.PeptidePosition`` cell to a single int.

    Spectronaut records the peptide's start position in the parent protein.
    For peptides that map to multiple proteins in a group, the value is
    ``";"`` separated (e.g. ``"37;188"``). For proteins with multiple mapping
    positions, ``","`` separators appear inside each field.

    We return the FIRST valid integer in reading order (matches the legacy
    Hogrebe R script and Spectronaut's own site collapse plugin). Sites for
    which no valid position exists yield ``None`` and are dropped downstream.

    Parameters
    ----------
    position_str : Any
        String, NaN, ``"None"``, or empty string.

    Returns
    -------
    int | None
        The first valid integer, or ``None`` when parsing fails.
    """
    if pd.isna(position_str) or position_str == "None" or position_str == "":
        return None

    try:
        parts = str(position_str).split(";")
        if parts:
            first_part = parts[0].split(",")[0].strip()
            if first_part and first_part != "None":
                return int(first_part)
    except (ValueError, AttributeError):
        pass
    return None


# ---------------------------------------------------------------------------
# EG.PTMLocalizationProbabilities parsing
# ---------------------------------------------------------------------------


def parse_localization_probabilities(loc_string: Any) -> dict[int, float]:
    """Parse ``EG.PTMLocalizationProbabilities`` -> {position: probability}.

    Spectronaut writes localization probabilities as bracket suffixes with an
    inline percentage::

        _PVS[Phospho (STY): 92.3%]PS[Phospho (STY): 7.6%]S[Phospho (STY): 0.1%]..._

    Returns a dict keyed by 1-indexed AA position with probabilities in
    ``[0, 1]`` (Spectronaut writes percent; we divide by 100).

    Non-phospho brackets are ignored. Positions with no phospho bracket at
    all simply don't appear in the returned dict.

    Parameters
    ----------
    loc_string : Any
        The raw string, or NaN.

    Returns
    -------
    dict[int, float]
        Empty when input is missing / not a string / unparseable.

    Examples
    --------
    >>> parse_localization_probabilities("_PVS[Phospho (STY): 92.3%]PS[Phospho (STY): 7.6%]_")
    {3: 0.923, 5: 0.076}
    """
    if pd.isna(loc_string) or not isinstance(loc_string, str):
        return {}

    # Strip wrapper punctuation. Wrappers may combine ``_``, ``.``, ``*``, or space.
    s = loc_string.strip("_.* ")
    result: dict[int, float] = {}
    pos = 0  # current 1-indexed AA position
    i = 0

    while i < len(s):
        if s[i] == "[":
            # Skip whole bracket. Non-phospho brackets are ignored via the
            # startswith check below.
            end = s.index("]", i)
            bracket_content = s[i + 1 : end]
            if bracket_content.startswith("Phospho (STY)"):
                match = re.search(r":\s*([\d.]+)%", bracket_content)
                if match:
                    result[pos] = float(match.group(1)) / 100.0
            i = end + 1
        elif s[i].isalpha():
            # Advance the AA cursor -- brackets come AFTER the residue, so we
            # increment BEFORE the next bracket is read.
            pos += 1
            i += 1
        else:
            i += 1

    return result


def rank_select_positions(loc_dict: dict[int, float], n: int) -> tuple[list[int], list[float]]:
    """Return the top ``n`` (position, probability) pairs by descending prob.

    Ties are broken by ascending position (matches the legacy R script's
    ``rank(-prob, ties.method='first')`` then take the first N).

    Parameters
    ----------
    loc_dict : dict[int, float]
        Output of :func:`parse_localization_probabilities`.
    n : int
        Number of top positions to keep. Set to the number of phospho groups
        on the peptide (from ``EG.PrecursorId``).

    Returns
    -------
    (positions, probabilities)
        Both lists sorted parallel by descending probability. Empty when
        ``loc_dict`` is empty or ``n <= 0``.

    Examples
    --------
    >>> rank_select_positions({3: 0.9, 8: 0.1}, 1)
    ([3], [0.9])
    >>> rank_select_positions({3: 0.5, 8: 0.5, 12: 0.2}, 2)
    ([3, 8], [0.5, 0.5])
    """
    if not loc_dict or n <= 0:
        return [], []
    ranked = sorted(loc_dict.items(), key=lambda kv: (-kv[1], kv[0]))
    top_n = ranked[:n]
    positions = [pos for pos, _ in top_n]
    probabilities = [prob for _, prob in top_n]
    return positions, probabilities
