"""Colour palettes, figure geometry and layout helpers for :mod:`alphaphos.viz`.

Single source of truth for every hex value and size used by alphaPhos
figures.  Import from here rather than writing a hex code or font size
inline::

    from alphaphos.viz.style import CONDITION_PALETTE, plot_width, shorten_tick_labels

The categorical, embedding and volcano palettes are dicts keyed by group
count rather than one long list: they were subsampled so that the minimum
CIE dE2000 between any two swatches is maximised jointly under normal
vision, deuteranopia and protanopia (Machado et al. 2009 simulation
matrices, severity 1.0).  Slicing the first *n* entries off a long list
separates far worse than choosing *n* stops for that *n*.

Everything here is pure Python with no plotting dependency, so the
constants are importable on a bare-core install.
"""

from __future__ import annotations

from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Shared semantic anchors
#
# One blue and one red are reused wherever the same MEANING recurs (volcano
# poles, diverging heatmap poles, signed bars).  Reusing the exact hex is what
# makes a figure set read as one system: the reader learns "this red = up"
# once.  Do not introduce a second near-identical red or blue for the same role.
# ---------------------------------------------------------------------------
SEMANTIC_BLUE: str = "#1368aa"  # "down" / negative
SEMANTIC_RED: str = "#ef3c2d"  # "up"   / positive

# ---------------------------------------------------------------------------
# Categorical palette — conditions, genotypes, treatments
# Box plots, violins, bar charts, grouped bars, any discrete grouping.
# Source: "ocean-sunset" (teal -> cream -> rust) plus one muted periwinkle.
#
# Revised 2026-10-28: the original near-black swatch #001219 (L* 5) was dropped
# because black replicate points vanish on a box or bar filled with it.  Every
# fill now has L* >= 35.  Re-subsampled from the source minus #001219 plus
# #7e8bdb (L* 60; dE 15.4 from SEMANTIC_BLUE in all three visions) -- without
# that addition the 8-group set falls to dE 9.2, below the >= 10 bar.
#
# min dE2000 (worst of normal / deuteranope / protanope) per entry:
#   2: 49.9   3: 41.5   4: 17.3   5: 24.9   6: 13.6   7: 13.2   8: 11.3
# The 2- and 4-group sets were chosen on min(colourblind dE, grayscale dE) so
# they still survive black-and-white printing (grayscale dE 42.8 / 19.9 / 11.2
# for n = 2 / 3 / 4); from 5 groups up two colours merge in grayscale.
# Above 8 groups the palette can no longer keep every pair >= 10 — fall back
# to Viridis (see categorical_colors).
# ---------------------------------------------------------------------------
CONDITION_PALETTE: dict[int, list[str]] = {
    2: ["#005f73", "#e9d8a6"],
    3: ["#7e8bdb", "#e9d8a6", "#9b2226"],
    4: ["#005f73", "#0a9396", "#e9d8a6", "#ee9b00"],
    5: ["#005f73", "#7e8bdb", "#94d2bd", "#ee9b00", "#ae2012"],
    6: ["#005f73", "#7e8bdb", "#0a9396", "#94d2bd", "#ee9b00", "#bb3e03"],
    7: ["#005f73", "#7e8bdb", "#0a9396", "#94d2bd", "#ee9b00", "#ca6702", "#ae2012"],
    8: ["#005f73", "#7e8bdb", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#ca6702", "#ae2012"],
}
CATEGORICAL_MAX_GROUPS: int = 8

# ---------------------------------------------------------------------------
# Embedding palette — PCA / UMAP / t-SNE / MDS scatters only
# Source: "green-grey-red".  The two palest source swatches are excluded: a
# marker that pale on a white canvas shows only its black outline.
#
# min dE2000 (worst of normal / deuteranope / protanope):
#   2: 48.3   3: 18.3   4: 14.5   5: 11.6
# At 6 groups it falls to 8.4 under protanopia — so 5 is the hard ceiling and
# 6-8 groups fall back to CONDITION_PALETTE.
# ---------------------------------------------------------------------------
EMBEDDING_PALETTE: dict[int, list[str]] = {
    2: ["#004733", "#a5c1ae"],
    3: ["#004733", "#568d66", "#a5c1ae"],
    4: ["#004733", "#568d66", "#a5c1ae", "#cb0b0a"],
    5: ["#004733", "#2b6a4d", "#568d66", "#a5c1ae", "#cb0b0a"],
}
EMBEDDING_MAX_GROUPS: int = 5

# ---------------------------------------------------------------------------
# Heatmap colourscale for NON-NEGATIVE intensities: palest pink (low) ->
# darkest red (high).  A magnitude scale, not a semantic "up", which is why its
# dark end (#aa0000) is deliberately a different red from SEMANTIC_RED.
# ---------------------------------------------------------------------------
HEATMAP_SCALE: list[str] = [
    "#eaa8a8", "#e28080", "#e56b6b", "#e95555", "#ec4040",
    "#ef2b2b", "#de1616", "#cd0000", "#bb0000", "#aa0000",
]  # fmt: skip

# ---------------------------------------------------------------------------
# Diverging scale for data symmetric around zero — z-scores, correlations,
# log2FC, signed loadings.  Always use symmetric zmin=-z, zmax=+z: asymmetric
# bounds shift the white midpoint off zero and misrepresent the sign.
# ---------------------------------------------------------------------------
DIVERGING_NEG: str = SEMANTIC_BLUE
DIVERGING_POS: str = SEMANTIC_RED
DIVERGING_SCALE: list[list] = [
    [0.0, DIVERGING_NEG],
    [0.5, "#f7f7f7"],
    [1.0, DIVERGING_POS],
]

# ---------------------------------------------------------------------------
# Volcano plot — five semantic categories.
# min dE2000 between the five: 16.4 normal / 13.8 deuteranope / 11.5 protanope.
# ---------------------------------------------------------------------------
VOLCANO_SIG_UP: str = SEMANTIC_RED  # FDR sig  AND log2fc >= +threshold
VOLCANO_SIG_DOWN: str = SEMANTIC_BLUE  # FDR sig  AND log2fc <= -threshold
VOLCANO_ALMOST_UP: str = "#f29479"  # FDR sig  AND 0 < log2fc < +threshold
VOLCANO_ALMOST_DOWN: str = "#9dcee2"  # FDR sig  AND -threshold < log2fc < 0
VOLCANO_NS: str = "#e3e3e3"  # not significant

VOLCANO: dict[str, str] = {
    "sig_up": VOLCANO_SIG_UP,
    "sig_down": VOLCANO_SIG_DOWN,
    "almost_up": VOLCANO_ALMOST_UP,
    "almost_down": VOLCANO_ALMOST_DOWN,
    "ns": VOLCANO_NS,
}

# ---------------------------------------------------------------------------
# Semantic palettes for KNOWN biological groups — these win over
# CONDITION_PALETTE when every category of the coloured variable is a key.
# Taken verbatim from published sfPhospho figures; do not re-derive.
# ---------------------------------------------------------------------------
TIMEPOINT_PALETTE: dict[str, str] = {
    "R1": "#e76f51",  # coral
    "P1": "#264653",  # dark teal
    "P24": "#2a9d8f",  # jade
}
FIBRETYPE_PALETTE: dict[str, str] = {
    "Slow-1": "#335c67",  # teal
    "Hybrid-1/2A": "#c9b458",  # mustard
    "Fast-2A": "#e09f3e",  # orange
    "Hybrid-2A/2AX": "#9e2a2b",  # dark red
}
SEMANTIC_PALETTES: tuple[dict[str, str], ...] = (TIMEPOINT_PALETTE, FIBRETYPE_PALETTE)

# ---------------------------------------------------------------------------
# Outlines — every mark carries one.
# ---------------------------------------------------------------------------
OUTLINE = {"color": "black", "width": 0.5}  # scatter markers
OUTLINE_THIN = {"color": "black", "width": 0.3}  # bars
OUTLINE_VIOLIN = {"color": "black", "width": 1.5}  # violin / box contour

# ---------------------------------------------------------------------------
# Marker sizes — per chart type, not per taste.
# ---------------------------------------------------------------------------
MARKER_EMBEDDING = {"size": 14, "line": OUTLINE}
# Replicate points on a box / violin / bar are always BLACK so they read as
# data rather than as another category.
MARKER_POINTS = {"size": 14, "color": "black", "line": OUTLINE}
MARKER_VOLCANO = {"size": 8, "line": OUTLINE}

# Below this group size the points ARE the evidence and must be drawn over the
# summary shape.  Above it (all-site spreads) the cloud is noise.
POINT_OVERLAY_MAX_N: int = 100

# ---------------------------------------------------------------------------
# Typography — fixed by role.
# ---------------------------------------------------------------------------
TICK_FONT = {"family": "Arial", "size": 14}
AXIS_TITLE_FONT = {"family": "Arial", "size": 18}
ANNOTATION_FONT = {"family": "Arial Black", "size": 11, "color": "black"}
MAX_TICK_CHARS: int = 20

# ---------------------------------------------------------------------------
# Reference lines
# ---------------------------------------------------------------------------
REF_LINE_MEDIAN = {"color": "red", "width": 1, "dash": "dash"}
REF_LINE_THRESHOLD = {"color": "black", "width": 1, "dash": "dash"}
ZERO_LINE = {"color": "black", "width": 1}

# ---------------------------------------------------------------------------
# Figure geometry — height is ALWAYS 600.  Width starts at 600 and grows in
# 300 px steps when there are too many categories to fit, capped at 1800.
# ---------------------------------------------------------------------------
PLOT_HEIGHT: int = 600
PLOT_WIDTH_BASE: int = 600
PLOT_WIDTH_STEP: int = 300
PLOT_WIDTH_MAX: int = 1800
CATEGORIES_PER_WIDTH_STEP: int = 6

LAYOUT_BASE = {
    "template": "plotly_white",
    "width": PLOT_WIDTH_BASE,
    "height": PLOT_HEIGHT,
    "font": {"family": "Arial", "size": 11},
    "showlegend": False,
}

# Tried in order by shorten_tick_labels; the first that isolates a
# distinguishing token wins.
DELIMITERS: tuple[str, ...] = ("_", "|", ";", ",", ".", "-", " ")


def plot_width(n_categories: int) -> int:
    """Figure width for a chart with ``n_categories`` categories.

    600 px up to 6 categories, then one 300 px step per additional 6, capped
    at 1800.
    """
    if n_categories <= CATEGORIES_PER_WIDTH_STEP:
        return PLOT_WIDTH_BASE
    extra_steps = -(-(n_categories - CATEGORIES_PER_WIDTH_STEP) // CATEGORIES_PER_WIDTH_STEP)
    return min(PLOT_WIDTH_BASE + extra_steps * PLOT_WIDTH_STEP, PLOT_WIDTH_MAX)


def shorten_tick_labels(labels: Iterable, max_chars: int = MAX_TICK_CHARS) -> list[str]:
    """Reduce long tick labels to the token(s) that differ between them.

    ``["20260101_QE_sfPhospho_SlowFiber_R1", "20260101_QE_sfPhospho_FastFiber_R1"]``
    becomes ``["SlowFiber", "FastFiber"]``.

    Per delimiter: with a uniform token count keep only the varying
    positions; with ragged counts strip the tokens every label shares at the
    front and back.  Labels are returned unchanged when they already fit or
    when shortening would make two labels identical.
    """
    labels = [str(x) for x in labels]
    if not labels or max(len(x) for x in labels) <= max_chars:
        return list(labels)

    n_distinct = len(set(labels))

    def truncate() -> list[str]:
        out = [x if len(x) <= max_chars else x[: max_chars - 1] + "…" for x in labels]
        return out if len(set(out)) == n_distinct else list(labels)

    # With a single distinct label every token looks common, so "keep the
    # varying token" is undefined.
    if n_distinct < 2:
        return truncate()

    def accept(out: list[str]) -> list[str] | None:
        return out if len(set(out)) == n_distinct and all(out) else None

    for delim in DELIMITERS:
        parts = [x.split(delim) for x in labels]
        if {len(p) for p in parts} == {1}:
            continue

        if len({len(p) for p in parts}) == 1:
            n_tokens = len(parts[0])
            varying = [i for i in range(n_tokens) if len({p[i] for p in parts}) > 1]
            if varying:
                hit = accept([delim.join(p[i] for i in varying) for p in parts])
                if hit is not None:
                    return hit

        shortest = min(len(p) for p in parts)
        head = 0
        while head < shortest - 1 and len({p[head] for p in parts}) == 1:
            head += 1
        tail = 0
        while tail < shortest - head - 1 and len({p[-1 - tail] for p in parts}) == 1:
            tail += 1
        if head or tail:
            hit = accept([delim.join(p[head : len(p) - tail]) for p in parts])
            if hit is not None:
                return hit

    return truncate()


def categorical_colors(categories: Iterable, embedding: bool = False) -> dict[str, str]:
    """Map category labels to hexes, picking the palette entry for this group count.

    A semantic palette (timepoints, fibre types) wins when every category is
    one of its keys.  ``embedding=True`` selects the PCA / UMAP / t-SNE
    palette, which falls back to the condition palette above 5 groups.  Past
    8 groups a Viridis sample is used (needs ``plotly``).
    """
    cats = list(dict.fromkeys(str(c) for c in categories))
    n = len(cats)
    for semantic in SEMANTIC_PALETTES:
        if cats and set(cats) <= semantic.keys():
            return {c: semantic[c] for c in cats}

    palette = EMBEDDING_PALETTE if embedding else CONDITION_PALETTE
    ceiling = EMBEDDING_MAX_GROUPS if embedding else CATEGORICAL_MAX_GROUPS
    if n < 2:
        return {cats[0]: palette[2][0]} if cats else {}
    if n <= ceiling:
        return dict(zip(cats, palette[n], strict=True))
    if n <= CATEGORICAL_MAX_GROUPS:
        return dict(zip(cats, CONDITION_PALETTE[n], strict=True))

    from plotly.colors import sample_colorscale

    return dict(
        zip(cats, sample_colorscale("Viridis", [i / (n - 1) for i in range(n)]), strict=True)
    )


def should_show_points(group_sizes: Iterable[int]) -> bool:
    """True when every group is small enough that individual points must be drawn."""
    sizes = list(group_sizes)
    return bool(sizes) and max(sizes) < POINT_OVERLAY_MAX_N


__all__ = [
    "SEMANTIC_BLUE",
    "SEMANTIC_RED",
    "CONDITION_PALETTE",
    "CATEGORICAL_MAX_GROUPS",
    "EMBEDDING_PALETTE",
    "EMBEDDING_MAX_GROUPS",
    "HEATMAP_SCALE",
    "DIVERGING_NEG",
    "DIVERGING_POS",
    "DIVERGING_SCALE",
    "VOLCANO",
    "VOLCANO_SIG_UP",
    "VOLCANO_SIG_DOWN",
    "VOLCANO_ALMOST_UP",
    "VOLCANO_ALMOST_DOWN",
    "VOLCANO_NS",
    "TIMEPOINT_PALETTE",
    "FIBRETYPE_PALETTE",
    "SEMANTIC_PALETTES",
    "OUTLINE",
    "OUTLINE_THIN",
    "OUTLINE_VIOLIN",
    "MARKER_EMBEDDING",
    "MARKER_POINTS",
    "MARKER_VOLCANO",
    "POINT_OVERLAY_MAX_N",
    "TICK_FONT",
    "AXIS_TITLE_FONT",
    "ANNOTATION_FONT",
    "MAX_TICK_CHARS",
    "REF_LINE_MEDIAN",
    "REF_LINE_THRESHOLD",
    "ZERO_LINE",
    "PLOT_HEIGHT",
    "PLOT_WIDTH_BASE",
    "PLOT_WIDTH_STEP",
    "PLOT_WIDTH_MAX",
    "CATEGORIES_PER_WIDTH_STEP",
    "LAYOUT_BASE",
    "DELIMITERS",
    "plot_width",
    "shorten_tick_labels",
    "categorical_colors",
    "should_show_points",
]
