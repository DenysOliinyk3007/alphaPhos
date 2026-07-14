"""Palette constants for alphaphos.qc panels.

Derived from the viz philosophy in ``CLAUDE_viz.md`` -- source of truth
for the colour rules across all alphaPhos plotting.  These constants
are imported by panel functions; do not define new palettes here
without updating ``CLAUDE_viz.md`` first.
"""

from __future__ import annotations

# Categorical condition palette -- purple → very-dark → deep-red ramp
# used for n_conditions <= 10 in PCA / t-SNE / UMAP / condition scatters.
CONDITION_PALETTE_10: tuple[str, ...] = (
    "#6900cc",
    "#5c00b3",
    "#4f0099",
    "#350066",
    "#1b0033",
    "#360000",
    "#6b0000",
    "#a10000",
    "#bc0000",
    "#d60000",
)

# Heatmap ramp -- palest-pink at low end, darkest-red at high end.
HEATMAP_SCALE: tuple[str, ...] = (
    "#eaa8a8",
    "#e28080",
    "#e56b6b",
    "#e95555",
    "#ec4040",
    "#ef2b2b",
    "#de1616",
    "#cd0000",
    "#bb0000",
    "#aa0000",
)

# Volcano semantics.
VOLCANO_SIG_UP = "#ef2b2b"  # significant + log2fc >= +threshold
VOLCANO_SIG_DOWN = "#1957db"  # significant + log2fc <= -threshold
VOLCANO_ALMOST_UP = "#e28080"  # significant + 0 < log2fc < +threshold
VOLCANO_ALMOST_DOWN = "#7ea3f1"  # significant + -threshold < log2fc < 0
VOLCANO_NS = "#e3e3e3"  # not significant

# QC-specific accent colours (borrowed from the palettes above).
QC_FLAG_COLOUR = "#d60000"  # flagged sample (from CONDITION_PALETTE_10[-1])
QC_NEUTRAL_COLOUR = "#1b0033"  # unflagged sample (from CONDITION_PALETTE_10[4])
QC_REFERENCE_LINE_GREY = "#999999"  # threshold lines / gridless reference

# Layout defaults from CLAUDE_viz.md §2.
DEFAULT_TEMPLATE = "plotly_white"
DEFAULT_FONT_SIZE_DASHBOARD = 12
DEFAULT_FONT_SIZE_MANUSCRIPT = 14

# Marker/line convention: every mark has a thin black outline.
DEFAULT_MARKER_OUTLINE = {"color": "black", "width": 0.5}
DEFAULT_BAR_OUTLINE = {"color": "black", "width": 0.3}
