"""Sample-level figures: embeddings, intensity distributions, site abundance, heatmaps."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import pandas as pd

from alphaphos.viz._base import (
    group_order,
    layout,
    matrix,
    obs_column,
    require_plotly,
    site_label,
    style_axes,
)
from alphaphos.viz.style import (
    ANNOTATION_FONT,
    AXIS_TITLE_FONT,
    DIVERGING_SCALE,
    HEATMAP_SCALE,
    MARKER_EMBEDDING,
    MARKER_POINTS,
    OUTLINE_VIOLIN,
    REF_LINE_MEDIAN,
    TICK_FONT,
    categorical_colors,
    plot_width,
    shorten_tick_labels,
    should_show_points,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

# basis -> (obsm key, axis-title stem)
_EMBEDDINGS = {"pca": ("X_pca", "PC"), "umap": ("X_umap", "UMAP"), "tsne": ("X_tsne", "t-SNE")}
_MISSING_GROUP = "NA"
# Above this many rows the site names are unreadable at height 600.
_MAX_ROW_LABELS = 50


def _groups(adata: ad.AnnData, column: str) -> tuple[pd.Series, list[str]]:
    values = obs_column(adata, column)
    order = group_order(values)
    labels = values.astype(str).where(values.notna(), _MISSING_GROUP)
    if (labels == _MISSING_GROUP).any():
        order = [*order, _MISSING_GROUP]
    return labels, order


def _sample_order(labels: pd.Series, order: list[str]) -> list[str]:
    """Samples grouped by condition, then by name within a group."""
    rank = {g: i for i, g in enumerate(order)}
    return sorted(labels.index.astype(str), key=lambda s: (rank[labels[s]], s))


def embedding(
    adata: ad.AnnData,
    *,
    basis: Literal["pca", "umap", "tsne"] = "pca",
    color: str = "condition",
    components: tuple[int, int] = (1, 2),
) -> go.Figure:
    """Sample scatter in PCA / UMAP / t-SNE space, coloured by an ``obs`` column.

    Groups are named at their centroid instead of in a legend.  Run the
    matching :mod:`alphaphos.dimred` function first.  UMAP / t-SNE axes carry
    no quantitative meaning and between-cluster distances are not
    interpretable -- say so in the caption.

    Parameters
    ----------
    basis
        Which ``adata.obsm`` embedding to draw.
    color
        Categorical ``adata.obs`` column.
    components
        1-based component numbers for the x and y axes.
    """
    go = require_plotly()
    if basis not in _EMBEDDINGS:
        raise ValueError(f"basis must be one of {sorted(_EMBEDDINGS)}, got {basis!r}")
    key, stem = _EMBEDDINGS[basis]
    if key not in adata.obsm:
        raise KeyError(f"adata.obsm[{key!r}] not found; run ap.dimred.{basis}(adata) first")
    coords = np.asarray(adata.obsm[key])
    cx, cy = components[0] - 1, components[1] - 1
    if min(cx, cy) < 0 or max(cx, cy) >= coords.shape[1]:
        raise ValueError(f"components {components} out of range for {coords.shape[1]} dims")

    labels, order = _groups(adata, color)
    colors = categorical_colors(order, embedding=True)
    names = adata.obs_names.astype(str)

    fig = go.Figure()
    for group in order:
        mask = (labels == group).to_numpy()
        x, y = coords[mask, cx], coords[mask, cy]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={**MARKER_EMBEDDING, "color": colors[group]},
                name=group,
                text=names[mask],
                hovertemplate=f"<b>%{{text}}</b><br>{group}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=float(np.mean(x)),
            y=float(np.mean(y)),
            text=group,
            font=ANNOTATION_FONT,
            showarrow=False,
        )

    titles = [f"{stem}{c}" for c in components]
    ratio = adata.uns.get(basis, {}).get("variance_ratio") if basis == "pca" else None
    if ratio is not None:
        titles = [f"{t} ({ratio[c - 1]:.1%})" for t, c in zip(titles, components, strict=True)]
    fig.update_layout(**layout())
    return style_axes(fig, *titles)


def _box_stats(values: np.ndarray) -> dict[str, float]:
    """Tukey box statistics; whiskers reach the most extreme value within 1.5 IQR."""
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
    return {
        "q1": q1,
        "median": med,
        "q3": q3,
        "lowerfence": inside.min(),
        "upperfence": inside.max(),
    }


def intensity_distribution(
    adata: ad.AnnData,
    *,
    color: str = "condition",
    layer: str | None = None,
    y_title: str = "log2 intensity",
) -> go.Figure:
    """One box per sample of its intensities across all sites.

    The visual check for loading and normalisation: medians should line up
    with the dashed global median.  Individual points are deliberately
    omitted -- each box summarises thousands of sites, so the spread is the
    message.  Boxes are precomputed, so the figure stays small however many
    sites there are.
    """
    go = require_plotly()
    labels, order = _groups(adata, color)
    colors = categorical_colors(order)
    samples = _sample_order(labels, order)
    x = pd.DataFrame(matrix(adata, layer), index=adata.obs_names.astype(str))

    fig = go.Figure()
    for sample in samples:
        values = x.loc[sample].to_numpy()
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        stats = _box_stats(values)
        fig.add_trace(
            go.Box(
                x=[sample],
                **{k: [v] for k, v in stats.items()},
                fillcolor=colors[labels[sample]],
                line=OUTLINE_VIOLIN,
                name=sample,
                hovertext=f"{sample}<br>{labels[sample]}<br>n sites = {values.size}",
                hoverinfo="text",
            )
        )

    finite = x.to_numpy()[np.isfinite(x.to_numpy())]
    if finite.size:
        fig.add_hline(y=float(np.median(finite)), line=REF_LINE_MEDIAN)
    fig.update_layout(**layout(plot_width(len(samples))))
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=samples,
        tickvals=samples,
        ticktext=shorten_tick_labels(samples),
    )
    return style_axes(fig, "", y_title)


def site_abundance(
    adata: ad.AnnData,
    site: str,
    *,
    groupby: str = "condition",
    layer: str | None = None,
    kind: Literal["box", "violin"] = "box",
    y_title: str = "log2 intensity",
) -> go.Figure:
    """Abundance of one site across groups, with every replicate drawn as a point.

    The figure to open from a volcano hit: it shows whether a fold change
    rests on consistent replicates or on one outlier.  Missing values are
    left out; the hover text reports how many samples per group carried a
    value.
    """
    go = require_plotly()
    if kind not in ("box", "violin"):
        raise ValueError(f"kind must be 'box' or 'violin', got {kind!r}")
    if site not in adata.var_names:
        raise KeyError(f"site {site!r} not in adata.var_names")
    labels, order = _groups(adata, groupby)
    colors = categorical_colors(order)
    col = adata.var_names.get_loc(site)
    values = pd.Series(matrix(adata[:, [col]], layer)[:, 0], index=labels.index)
    observed = values[np.isfinite(values)]
    show_points = should_show_points(labels[observed.index].value_counts())

    fig = go.Figure()
    for group in order:
        in_group = labels == group
        y = observed[in_group[observed.index]]
        hover = f"{group}: {len(y)}/{int(in_group.sum())} samples quantified"
        shape = go.Violin if kind == "violin" else go.Box
        extra = {"box_visible": True, "points": False} if kind == "violin" else {"boxpoints": False}
        fig.add_trace(
            shape(
                x=[group] * len(y),
                y=y,
                fillcolor=colors[group],
                line=OUTLINE_VIOLIN,
                name=group,
                hovertext=hover,
                hoverinfo="text",
                **extra,
            )
        )
        if show_points:
            # Transparent box whose only job is to carry the black points on top.
            fig.add_trace(
                go.Box(
                    x=[group] * len(y),
                    y=y,
                    fillcolor="rgba(0,0,0,0)",
                    line={"color": "rgba(0,0,0,0)"},
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    marker=MARKER_POINTS,
                    name=group,
                    text=list(y.index),
                    hovertemplate="%{text}<br>%{y:.2f}<extra></extra>",
                )
            )

    fig.add_annotation(
        # Above the plot area: inside it the label collides with the top group.
        x=0,
        y=1.02,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        text=site_label(site),
        font=ANNOTATION_FONT,
        showarrow=False,
    )
    fig.update_layout(**layout(plot_width(len(order))))
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=order,
        tickvals=order,
        ticktext=shorten_tick_labels(order),
    )
    return style_axes(fig, "", y_title)


def _zscore_rows(m: np.ndarray) -> np.ndarray:
    mean = np.nanmean(m, axis=1, keepdims=True)
    sd = np.nanstd(m, axis=1, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (m - mean) / sd


def _leaf_order(m: np.ndarray) -> np.ndarray:
    """Average-linkage leaf order; NaN cells are filled with the row mean first."""
    from scipy.cluster.hierarchy import leaves_list, linkage

    if m.shape[0] < 3:
        return np.arange(m.shape[0])
    row_mean = np.nanmean(m, axis=1, keepdims=True)
    filled = np.where(np.isnan(m), np.nan_to_num(row_mean), m)
    return leaves_list(linkage(filled, method="average", metric="euclidean"))


def heatmap(
    adata: ad.AnnData,
    *,
    sites: Sequence[str] | None = None,
    groupby: str = "condition",
    layer: str | None = None,
    zscore: bool = True,
    cluster_sites: bool = True,
    cluster_samples: bool = False,
    max_sites: int = 500,
) -> go.Figure:
    """Sites x samples heatmap.

    Parameters
    ----------
    sites
        Site keys to draw, e.g. the DE hits.  ``None`` takes the
        ``max_sites`` most variable sites.
    groupby
        ``obs`` column used to order samples and to label the sample blocks
        (ignored for ordering when ``cluster_samples=True``).
    zscore
        Row-wise z-score on the diverging blue-white-red scale (symmetric
        bounds).  ``False`` draws raw intensities on the pink-to-red ramp,
        bounded at the 1st / 99th percentile so outliers don't flatten it.
    cluster_sites, cluster_samples
        Average-linkage ordering of rows / columns.
    max_sites
        Cap used when ``sites`` is ``None``; larger heatmaps are unreadable
        at a height of 600 px.
    """
    go = require_plotly()
    labels, order = _groups(adata, groupby)
    samples = _sample_order(labels, order)
    x = pd.DataFrame(
        matrix(adata, layer).T, index=adata.var_names, columns=adata.obs_names.astype(str)
    )[samples]

    if sites is None:
        sites = x.var(axis=1).nlargest(max_sites).index
    else:
        missing = [s for s in sites if s not in x.index]
        if missing:
            raise KeyError(f"{len(missing)} site(s) not in adata.var_names, e.g. {missing[:3]}")
    x = x.loc[list(sites)]
    m = _zscore_rows(x.to_numpy()) if zscore else x.to_numpy()

    if cluster_sites:
        rows = _leaf_order(m)
        m, x = m[rows], x.iloc[rows]
    if cluster_samples:
        cols = _leaf_order(m.T)
        m, x = m[:, cols], x.iloc[:, cols]

    finite = m[np.isfinite(m)]
    if zscore:
        z = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
        scale = {"colorscale": DIVERGING_SCALE, "zmin": -z, "zmax": z, "zmid": 0}
        bar_title = "z-score"
    else:
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
        scale = {"colorscale": HEATMAP_SCALE, "zmin": float(lo), "zmax": float(hi)}
        bar_title = "log2 intensity"

    columns = list(x.columns)
    row_names = [site_label(s) for s in x.index]
    fig = go.Figure(
        go.Heatmap(
            z=m,
            # Numeric positions on both axes: block annotations and tick text
            # then never collide with sample names that look like numbers.
            x=list(range(len(columns))),
            y=list(range(len(row_names))),
            customdata=np.array([[f"{r}<br>{c}" for c in columns] for r in row_names]),
            hovertemplate="%{customdata}<br>%{z:.2f}<extra></extra>",
            colorbar={
                "title": {"text": bar_title, "font": AXIS_TITLE_FONT},
                "tickfont": TICK_FONT,
                "thickness": 12,
            },
            **scale,
        )
    )

    if not cluster_samples:
        # Name each contiguous block of samples above the heatmap.
        block = labels[columns].to_numpy()
        starts = [0, *[i for i in range(1, len(block)) if block[i] != block[i - 1]], len(block)]
        for a, b in pairwise(starts):
            fig.add_annotation(
                x=(a + b - 1) / 2,
                y=1.02,
                xref="x",
                yref="paper",
                yanchor="bottom",
                text=block[a],
                font=ANNOTATION_FONT,
                showarrow=False,
            )

    fig.update_layout(**layout(plot_width(len(columns))))
    fig.update_xaxes(
        tickvals=list(range(len(columns))),
        ticktext=shorten_tick_labels(columns),
        tickfont=TICK_FONT,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickvals=list(range(len(row_names))),
        ticktext=row_names,
        showticklabels=len(row_names) <= _MAX_ROW_LABELS,
        tickfont=TICK_FONT,
    )
    return fig
