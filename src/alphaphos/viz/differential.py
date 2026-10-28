"""Figures for differential-expression results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphaphos.viz._base import checked_table, layout, require_plotly, site_label, style_axes
from alphaphos.viz.style import (
    ANNOTATION_FONT,
    MARKER_VOLCANO,
    REF_LINE_THRESHOLD,
    VOLCANO,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

# Bottom layer first so coloured points sit on top of the grey cloud.
_VOLCANO_Z_ORDER = ("ns", "almost_down", "almost_up", "sig_down", "sig_up")


def volcano_categories(
    result: pd.DataFrame,
    *,
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
) -> pd.Series:
    """Per-site volcano category (``sig_up`` / ``sig_down`` / ``almost_*`` / ``ns``)."""
    fc = result["log2fc"].to_numpy(dtype=float)
    sig = result["fdr"].to_numpy(dtype=float) < fdr_threshold
    cat = np.select(
        [~sig, fc >= log2fc_threshold, fc <= -log2fc_threshold, fc > 0],
        ["ns", "sig_up", "sig_down", "almost_up"],
        default="almost_down",
    )
    return pd.Series(cat, index=result.index, name="volcano_category")


def volcano(
    result: pd.DataFrame,
    *,
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    label: Iterable[str] | None = None,
) -> go.Figure:
    """Volcano plot of a ``diff_exp_limma`` / ``diff_exp_limma_contrasts`` table.

    x = ``log2fc``, y = -log10 ``fdr``.  The y-axis is always the adjusted
    value, so the dashed horizontal line at ``fdr_threshold`` is exactly
    where significance starts.

    Parameters
    ----------
    result
        DE table with ``log2fc`` and ``fdr`` columns, indexed by site key.
    fdr_threshold, log2fc_threshold
        Significance and effect-size cut-offs; state both in the caption.
    label
        Site keys to name on the plot with a short arrow.  Keep it to a
        handful; every point already carries its name in the hover text.
    """
    go = require_plotly()
    df = checked_table(result, ("log2fc", "fdr"), "result").dropna(subset=["log2fc", "fdr"])
    cat = volcano_categories(df, fdr_threshold=fdr_threshold, log2fc_threshold=log2fc_threshold)
    neg_log_fdr = -np.log10(df["fdr"].clip(lower=np.finfo(float).tiny))
    names = [site_label(k) for k in df.index]

    fig = go.Figure()
    for c in _VOLCANO_Z_ORDER:
        mask = (cat == c).to_numpy()
        fig.add_trace(
            go.Scatter(
                x=df["log2fc"][mask],
                y=neg_log_fdr[mask],
                mode="markers",
                marker={**MARKER_VOLCANO, "color": VOLCANO[c]},
                name=c,
                customdata=np.column_stack(
                    [np.asarray(names, dtype=object)[mask], df["fdr"][mask], df.index[mask]]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>log2FC %{x:.2f}<br>"
                    "FDR %{customdata[1]:.2g}"
                    "<extra>%{customdata[2]}</extra>"
                ),
            )
        )

    fig.add_vline(x=log2fc_threshold, line=REF_LINE_THRESHOLD)
    fig.add_vline(x=-log2fc_threshold, line=REF_LINE_THRESHOLD)
    fig.add_hline(y=float(-np.log10(fdr_threshold)), line=REF_LINE_THRESHOLD)

    n_up = int((cat == "sig_up").sum())
    n_down = int((cat == "sig_down").sum())
    for x, anchor, text in ((0.98, "right", f"{n_up} up"), (0.02, "left", f"{n_down} down")):
        fig.add_annotation(
            x=x,
            y=0.98,
            xref="paper",
            yref="paper",
            xanchor=anchor,
            text=text,
            font=ANNOTATION_FONT,
            showarrow=False,
        )

    for i, key in enumerate(() if label is None else label):
        if key not in df.index:
            raise KeyError(f"label site {key!r} not in result")
        fig.add_annotation(
            x=float(df.at[key, "log2fc"]),
            y=float(neg_log_fdr[key]),
            text=site_label(key),
            font=ANNOTATION_FONT,
            showarrow=True,
            arrowhead=0,
            # Alternate sides and step upwards so neighbouring hits don't overprint.
            ax=60 if i % 2 == 0 else -60,
            ay=-25 - 18 * (i // 2),
        )

    fig.update_layout(**layout())
    return style_axes(fig, "log2 fold change", "-log10 FDR")
