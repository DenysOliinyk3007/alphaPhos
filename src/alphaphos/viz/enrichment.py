"""Figures for kinase-activity and pathway / site-set enrichment results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphaphos.viz._base import checked_table, layout, require_plotly, style_axes
from alphaphos.viz.style import (
    ANNOTATION_FONT,
    MARKER_POINTS,
    OUTLINE,
    OUTLINE_THIN,
    REF_LINE_THRESHOLD,
    SEMANTIC_BLUE,
    SEMANTIC_RED,
    ZERO_LINE,
    plot_width,
    shorten_tick_labels,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

# Signed directions take the semantic red / blue; unsigned results
# (direction="both" / "any", ANOVA-driven) are drawn as plain black points.
_DIRECTION_COLOURS = {
    "up": SEMANTIC_RED,
    "enriched": SEMANTIC_RED,
    "down": SEMANTIC_BLUE,
    "depleted": SEMANTIC_BLUE,
}
# Site-set ora/gsea name the set "set_name"; gene-level pathway_* call it "term".
_TERM_COLUMNS = ("set_name", "term")


def _empty(go, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        font=ANNOTATION_FONT,
        showarrow=False,
    )
    fig.update_layout(**layout())
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def kinase_activity_bar(
    ksea: pd.DataFrame,
    *,
    fdr_threshold: float = 0.05,
    top_n: int = 25,
) -> go.Figure:
    """Horizontal bar of signed kinase activity from ``ap.enrichment.kinase_activity``.

    Shows kinases with ``fdr < fdr_threshold``, the ``top_n`` largest by
    absolute score.  If more pass, tighten the FDR cut rather than raising
    ``top_n``: the figure is capped at 1800 px.
    """
    go = require_plotly()
    ksea = checked_table(ksea, ("kinase", "score", "fdr"), "ksea")
    d = ksea[ksea["fdr"] < fdr_threshold]
    if d.empty:
        return _empty(go, f"no kinases at FDR < {fdr_threshold:g}")
    d = d.loc[d["score"].abs().nlargest(top_n).index].sort_values("score")
    kinases = d["kinase"].astype(str).tolist()
    n_sub = d["n_substrates"] if "n_substrates" in d else pd.Series("?", index=d.index)

    fig = go.Figure(
        go.Bar(
            x=d["score"],
            y=shorten_tick_labels(kinases),
            orientation="h",
            marker={
                "color": [SEMANTIC_RED if s > 0 else SEMANTIC_BLUE for s in d["score"]],
                "line": OUTLINE_THIN,
            },
            customdata=np.column_stack([kinases, d["fdr"], n_sub]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>score %{x:.2f}<br>FDR %{customdata[1]:.2g}"
                "<br>%{customdata[2]} substrates<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line=ZERO_LINE)
    fig.update_layout(**layout(plot_width(len(d))))
    return style_axes(fig, "kinase activity score", "")


def enrichment_dotplot(
    result: pd.DataFrame,
    *,
    fdr_threshold: float = 0.05,
    top_n: int = 20,
) -> go.Figure:
    """Dot plot of the ``top_n`` most significant sets, at x = -log10 FDR.

    Accepts the output of ``ap.enrichment.ora`` / ``gsea`` (site sets) and
    ``pathway_enrichment`` / ``pathway_gsea`` (gene sets).  Dots are red for
    up / enriched and blue for down / depleted; unsigned results are black.
    The dashed line marks ``fdr_threshold``, so sets to its left are shown
    for context, not as hits.
    """
    go = require_plotly()
    term_col = next((c for c in _TERM_COLUMNS if c in result.columns), None)
    if term_col is None:
        raise KeyError(f"result needs one of {_TERM_COLUMNS}; got {list(result.columns)}")
    d = checked_table(result, ("fdr",), "result").dropna(subset=["fdr"]).nsmallest(top_n, "fdr")
    if d.empty:
        return _empty(go, "no tested sets")

    direction = d["direction"].astype(str) if "direction" in d else pd.Series("", index=d.index)
    terms = d[term_col].astype(str)
    # The same term can appear once per direction (split ORA) or library.
    if terms.duplicated(keep=False).any():
        terms = terms + " (" + direction + ")"
    if terms.duplicated(keep=False).any() and "library" in d:
        terms = terms + " [" + d["library"].astype(str) + "]"

    # Most significant at the top.
    order = terms.tolist()[::-1]
    ticks = dict(zip(order, shorten_tick_labels(order), strict=True))
    x = -np.log10(d["fdr"].clip(lower=np.finfo(float).tiny))
    colours = [_DIRECTION_COLOURS.get(v, MARKER_POINTS["color"]) for v in direction]

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=terms,
            mode="markers",
            marker={"size": MARKER_POINTS["size"], "color": colours, "line": OUTLINE},
            customdata=np.column_stack([terms, d["fdr"], direction]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>FDR %{customdata[1]:.2g}<br>%{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=-np.log10(fdr_threshold), line=REF_LINE_THRESHOLD)
    fig.update_layout(**layout())
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=order,
        tickvals=order,
        ticktext=[ticks[t] for t in order],
    )
    return style_axes(fig, "-log10 FDR", "")
