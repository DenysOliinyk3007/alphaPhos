"""Figures for raw-report QC metrics (:mod:`alphaphos.qc.psm_metrics`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphaphos.qc.psm_metrics import DEFAULT_CONTAMINANT_FLAG_FRACTION
from alphaphos.viz._base import checked_table, layout, require_plotly, style_axes
from alphaphos.viz.style import (
    ANNOTATION_FONT,
    OUTLINE_THIN,
    REF_LINE_THRESHOLD,
    categorical_colors,
    plot_width,
    shorten_tick_labels,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go


def contaminant_fraction(fractions: pd.DataFrame) -> go.Figure:
    """Bar per run: % of its precursor intensity that comes from contaminants.

    Takes ``ap.qc.compute_contaminant_fraction_per_sample`` output.  The
    dashed line is the flag threshold recorded in that table's provenance
    (10 % by default).  When every run sits below it, the line is left out
    rather than squashing the bars, and an annotation says so instead.
    """
    go = require_plotly()
    threshold = float(
        fractions.attrs.get("provenance", {}).get(
            "flag_fraction_threshold", DEFAULT_CONTAMINANT_FLAG_FRACTION
        )
    )
    d = checked_table(
        fractions,
        ("contaminant_fraction", "contaminant_intensity", "total_intensity"),
        "fractions",
    )
    runs = [str(r) for r in d.index]
    percent = d["contaminant_fraction"].to_numpy(dtype=float) * 100
    (colour,) = categorical_colors(["runs"]).values()

    fig = go.Figure(
        go.Bar(
            x=runs,
            y=percent,
            marker={"color": colour, "line": OUTLINE_THIN},
            customdata=np.column_stack([d["contaminant_intensity"], d["total_intensity"]]),
            hovertemplate=(
                "<b>%{x}</b><br>%{y:.2f} % of intensity<br>"
                "contaminants %{customdata[0]:.3s} of %{customdata[1]:.3s}<extra></extra>"
            ),
        )
    )
    if np.nanmax(percent, initial=0) >= threshold * 100:
        fig.add_hline(y=threshold * 100, line=REF_LINE_THRESHOLD)
        note = f"flag threshold {threshold:.0%}"
    else:
        note = f"all runs below the {threshold:.0%} flag threshold"
    fig.add_annotation(
        # Above the plot area: inside it the note collides with the tallest bars.
        x=1,
        y=1.02,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="bottom",
        text=note,
        font=ANNOTATION_FONT,
        showarrow=False,
    )

    fig.update_layout(**layout(plot_width(len(runs))))
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=runs,
        tickvals=runs,
        ticktext=shorten_tick_labels(runs),
    )
    fig.update_yaxes(rangemode="tozero")
    return style_axes(fig, "", "contaminant intensity (% of total)")
