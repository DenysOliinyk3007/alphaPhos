"""SILAC / multi-channel channel assignment for Spectronaut reports.

Spectronaut encodes the metabolic label as a residue tag inside ``FG.Id``
(``_AAAAAATAPPS[Phospho (STY)]PGPAQPGPR[Arg10]_.2``).  No other column carries
it: ``EG.PrecursorId`` and ``EG.ModifiedSequence`` have the label stripped, and
``EG.Workflow`` only distinguishes ``LABEL`` from ``LABEL_FREE``.  This module
turns those tags into a channel label.

The plex is defined by :data:`DEFAULT_LABEL_CHANNELS`, which maps a Spectronaut
label tag to a channel name.  A precursor carrying no known tag is the light
channel; a precursor carrying tags from two different channels is a
contradiction and raises.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from alphaphos.constants import COL_FG_ID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plex definition
# ---------------------------------------------------------------------------

#: Channel name used when a precursor carries no label tag.  A peptide without
#: a labelable residue also lands here, which is correct: it is unlabelled.
LIGHT_CHANNEL = "light"

#: Spectronaut label tag -> channel name.  Covers the standard Lys/Arg triplex;
#: a 2-plex experiment simply never produces the medium tags.  Override via
#: ``advanced={"label_channels": {...}}`` for non-standard labels.
DEFAULT_LABEL_CHANNELS: dict[str, str] = {
    "Arg6": "medium",
    "Lys4": "medium",
    "Arg10": "heavy",
    "Lys8": "heavy",
}

#: Order channels appear in, so sample names sort predictably.
CHANNEL_ORDER: tuple[str, ...] = (LIGHT_CHANNEL, "medium", "heavy")

_TAG_RE = re.compile(r"\[([^\]]+)\]")


def validate_label_channels(label_channels: Any) -> dict[str, str]:
    """Check a label-tag -> channel-name mapping and return it as a plain dict."""
    if not isinstance(label_channels, dict) or not label_channels:
        raise ValueError(
            "label_channels must be a non-empty dict mapping a Spectronaut label "
            f"tag to a channel name, e.g. {DEFAULT_LABEL_CHANNELS!r}; "
            f"got {label_channels!r}"
        )
    for tag, channel in label_channels.items():
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"label_channels keys must be non-empty strings, got {tag!r}")
        if not isinstance(channel, str) or not channel:
            raise ValueError(f"label_channels values must be non-empty strings, got {channel!r}")
        if channel == LIGHT_CHANNEL:
            raise ValueError(
                f"label_channels maps tag {tag!r} to {LIGHT_CHANNEL!r}, but the light "
                "channel is defined by the ABSENCE of a label tag. Remove the entry."
            )
    return dict(label_channels)


def label_tags_present(fg_ids: pd.Series, label_channels: dict[str, str]) -> set[str]:
    """Return the subset of ``label_channels`` tags that occur in ``fg_ids``.

    Used to decide whether a report is multi-channel at all.  Scans the unique
    values only -- ``FG.Id`` repeats heavily across runs.
    """
    known = set(label_channels)
    found: set[str] = set()
    for value in pd.unique(fg_ids.dropna()):
        found |= known.intersection(_TAG_RE.findall(str(value)))
        if found == known:
            break
    return found


def derive_channel(
    fg_ids: pd.Series,
    label_channels: dict[str, str] | None = None,
) -> pd.Series:
    """Map each ``FG.Id`` to its channel name.

    Parameters
    ----------
    fg_ids : pandas.Series
        ``FG.Id`` values (the label-bearing precursor id).
    label_channels : dict, optional
        Label tag -> channel name.  Defaults to
        :data:`DEFAULT_LABEL_CHANNELS`.

    Returns
    -------
    pandas.Series
        Channel name per row, same index as ``fg_ids``.  Rows with no known
        tag get :data:`LIGHT_CHANNEL`.

    Raises
    ------
    ValueError
        If any single precursor carries tags belonging to two different
        channels (e.g. ``Arg6`` together with ``Lys8``), which cannot happen
        in a correctly searched experiment and would silently corrupt ratios.
    """
    mapping = validate_label_channels(label_channels or DEFAULT_LABEL_CHANNELS)

    # FG.Id repeats across runs; resolve each distinct value once.
    uniques = pd.unique(fg_ids.dropna())
    resolved: dict[str, str] = {}
    contradictions: dict[str, set[str]] = {}
    for value in uniques:
        channels = {mapping[t] for t in _TAG_RE.findall(str(value)) if t in mapping}
        if len(channels) > 1:
            contradictions[str(value)] = channels
        resolved[value] = channels.pop() if len(channels) == 1 else LIGHT_CHANNEL

    if contradictions:
        example, chans = next(iter(contradictions.items()))
        raise ValueError(
            f"{len(contradictions)} precursor(s) carry label tags from more than one "
            f"channel, e.g. {example!r} -> {sorted(chans)}. A precursor belongs to "
            "exactly one SILAC channel; check the label definitions in the search "
            "settings and the 'label_channels' mapping."
        )

    return fg_ids.map(resolved).astype("string")


def describe_channels(channels: pd.Series) -> dict[str, int]:
    """Row count per channel, ordered by :data:`CHANNEL_ORDER` then alphabetically."""
    counts = channels.value_counts().to_dict()
    ordered = [c for c in CHANNEL_ORDER if c in counts]
    ordered += sorted(c for c in counts if c not in CHANNEL_ORDER)
    return {c: int(counts[c]) for c in ordered}


def detect_multichannel(
    df: pd.DataFrame,
    label_channels: dict[str, str] | None = None,
) -> set[str]:
    """Return the label tags found in ``df``'s ``FG.Id`` column, empty if none.

    An empty set means the report is single-channel (or ``FG.Id`` was not
    exported, in which case a multi-channel report is indistinguishable from a
    label-free one -- see :func:`alphaphos.io.spectronaut.read_psm`).
    """
    if COL_FG_ID not in df.columns:
        return set()
    return label_tags_present(df[COL_FG_ID], label_channels or DEFAULT_LABEL_CHANNELS)


__all__ = [
    "LIGHT_CHANNEL",
    "DEFAULT_LABEL_CHANNELS",
    "CHANNEL_ORDER",
    "validate_label_channels",
    "label_tags_present",
    "derive_channel",
    "describe_channels",
    "detect_multichannel",
]
