"""Figures for filter-threshold choices."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from alphaphos.viz._base import checked_table, layout, require_plotly, style_axes
from alphaphos.viz.style import ANNOTATION_FONT, OUTLINE, REF_LINE_THRESHOLD, SEMANTIC_BLUE

if TYPE_CHECKING:
    import plotly.graph_objects as go


def retention_curve(
    sweep: pd.DataFrame,
    *,
    x: str = "threshold",
    y: str = "n_sites",
    chosen: float | None = None,
    x_title: str = "threshold",
    y_title: str = "sites retained",
) -> go.Figure:
    """Line of what a filter keeps as its threshold moves.

    The defaults read ``ap.wilson_threshold_sensitivity`` output directly;
    pass ``x`` / ``y`` for any other sweep table.  ``chosen`` marks the
    threshold actually applied.
    """
    go = require_plotly()
    d = checked_table(sweep, (x, y), "sweep").sort_values(x)

    fig = go.Figure(
        go.Scatter(
            x=d[x],
            y=d[y],
            mode="lines+markers",
            line={"color": SEMANTIC_BLUE, "width": 2},
            marker={"size": 8, "color": SEMANTIC_BLUE, "line": OUTLINE},
            hovertemplate=f"{x_title} %{{x}}<br>{y_title} %{{y}}<extra></extra>",
        )
    )
    if chosen is not None:
        fig.add_vline(x=chosen, line=REF_LINE_THRESHOLD)
        fig.add_annotation(
            x=chosen,
            y=float(d[y].max()),
            text=f"chosen = {chosen:g}",
            font=ANNOTATION_FONT,
            showarrow=True,
            arrowhead=0,
            ax=40,
            ay=-20,
        )
    fig.update_layout(**layout())
    return style_axes(fig, x_title, y_title)
