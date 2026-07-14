"""Plotly panel builders for QC metrics.

Each ``panel_*`` function returns a ``plotly.graph_objects.Figure``.
In a Jupyter notebook the figure renders inline automatically -- no
export step needed::

    fig = ap.qc.panel_retention_time_drift_summary(drift_df=drift)
    fig                        # ← inline display

For a paper figure or a headless run, opt in explicitly::

    fig.write_image('figure.pdf', width=900, height=400)   # kaleido
    fig.write_html('dashboard_snippet.html')

Styling follows the viz philosophy in ``CLAUDE_viz.md``:

- ``template='plotly_white'`` (no gridlines, white background)
- Every mark has a thin black outline
- Palette constants imported from ``_palettes.py``
- Annotations preferred over legends

Panels accept EITHER the raw source (e.g. ``psm_df``) OR the pre-
computed metric DataFrame from ``psm_metrics.compute_*``.  Pre-computed
path is preferred when the same data feeds multiple panels.

Requires ``plotly>=5``; install via ``pip install alphaphos[qc]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from typing import Any

    import plotly.graph_objects as _go

from alphaphos.qc._palettes import (
    DEFAULT_BAR_OUTLINE,
    DEFAULT_FONT_SIZE_DASHBOARD,
    DEFAULT_MARKER_OUTLINE,
    DEFAULT_TEMPLATE,
    QC_FLAG_COLOUR,
    QC_NEUTRAL_COLOUR,
    QC_REFERENCE_LINE_GREY,
)
from alphaphos.qc.psm_metrics import (
    DEFAULT_CONTAMINANT_FLAG_FRACTION,
    DEFAULT_DEPTH_FLAG_Z,
    DEFAULT_OFFSET_FLAG_THRESHOLD_MIN,
    compute_contaminant_fraction_per_sample,
    compute_psm_counts_per_sample,
    compute_retention_time_drift,
    compute_run_tic_per_sample,
)


def _resolve_sample_labels(
    sample_labels: dict[str, str] | pd.Series | pd.DataFrame | None,
    sample_index: pd.Index,
    *,
    condition_column: str = "condition",
) -> list[str] | None:
    """Return a per-sample display label list aligned with ``sample_index``.

    Accepts ``None`` (returns ``None`` -- callers fall back to the raw
    filename), a dict, a Series indexed by sample name, or a DataFrame
    with a ``sample`` column and a condition column.  Samples missing
    from the mapping keep their original name.
    """
    if sample_labels is None:
        return None
    if isinstance(sample_labels, pd.DataFrame):
        if "sample" not in sample_labels.columns:
            raise ValueError("sample_labels DataFrame must have a 'sample' column")
        if condition_column not in sample_labels.columns:
            raise ValueError(
                f"sample_labels DataFrame missing condition column {condition_column!r}"
            )
        mapping = dict(zip(sample_labels["sample"], sample_labels[condition_column], strict=False))
    elif isinstance(sample_labels, pd.Series):
        mapping = sample_labels.to_dict()
    else:
        mapping = dict(sample_labels)
    return [str(mapping.get(s, s)) for s in sample_index]


def _require_plotly() -> Any:
    try:
        import plotly.graph_objects as go

        return go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for alphaphos.qc panels.  Install with "
            "`pip install 'alphaphos[qc]'`."
        ) from exc


def _resolve_drift_df(
    *,
    psm_df: pd.DataFrame | None,
    drift_df: pd.DataFrame | None,
    acquisition_queue: pd.DataFrame | None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Accept either pre-computed drift_df or recompute from psm_df."""
    if drift_df is not None:
        return drift_df
    if psm_df is None:
        raise ValueError("must provide either drift_df (pre-computed) or psm_df (recompute)")
    return compute_retention_time_drift(psm_df, acquisition_queue=acquisition_queue, **kwargs)


def _resolve_counts_df(
    *,
    psm_df: pd.DataFrame | None,
    counts_df: pd.DataFrame | None,
    acquisition_queue: pd.DataFrame | None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Accept either pre-computed counts_df or recompute from psm_df."""
    if counts_df is not None:
        return counts_df
    if psm_df is None:
        raise ValueError("must provide either counts_df (pre-computed) or psm_df (recompute)")
    return compute_psm_counts_per_sample(psm_df, acquisition_queue=acquisition_queue, **kwargs)


def panel_psm_counts_summary(
    psm_df: pd.DataFrame | None = None,
    *,
    counts_df: pd.DataFrame | None = None,
    acquisition_queue: pd.DataFrame | None = None,
    sample_labels: dict[str, str] | pd.Series | pd.DataFrame | None = None,
    metric: str = "n_unique_precursors",
    flag_z_threshold: float = DEFAULT_DEPTH_FLAG_Z,
    width: int = 900,
    height: int = 400,
    **compute_kwargs: Any,
) -> _go.Figure:
    """Bar chart of per-sample identification depth (alphabetical order).

    Flagged samples (``depth_z_score < flag_z_threshold``) are drawn in
    the QC-flag colour; others in the neutral dark-purple.  Reference
    line at the cohort median.

    Parameters
    ----------
    metric
        Which count column to plot: ``"n_unique_precursors"`` (default,
        honest identification depth), ``"n_psms"`` (raw row count,
        inflated by Spectronaut fragment-level rows), or
        ``"n_unique_proteins"`` (protein-group breadth).
    """
    go = _require_plotly()
    df = _resolve_counts_df(
        psm_df=psm_df,
        counts_df=counts_df,
        acquisition_queue=acquisition_queue,
        flag_z_threshold=flag_z_threshold,
        **compute_kwargs,
    )
    if metric not in df.columns:
        raise KeyError(
            f"metric={metric!r} not in counts DataFrame; "
            f"available: {sorted(c for c in df.columns if c.startswith('n_'))}"
        )
    df = df.sort_index()

    flagged = df["is_flagged"].fillna(False).astype(bool).to_numpy()
    colours = [QC_FLAG_COLOUR if f else QC_NEUTRAL_COLOUR for f in flagged]
    values = df[metric].to_numpy()
    hover = [
        (
            f"<b>{sample}</b><br>"
            f"n_psms: {int(row.n_psms):,}<br>"
            f"n_unique_precursors: {int(row.n_unique_precursors):,}<br>"
            f"n_unique_proteins: {int(row.n_unique_proteins):,}<br>"
            f"depth z-score (MAD): {row.depth_z_score:+.2f}"
        )
        for sample, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(df.index.astype(str)),
            y=values,
            marker=dict(color=colours, line=DEFAULT_BAR_OUTLINE),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    # Cohort median reference line (dashed grey)
    prov = df.attrs.get("provenance", {})
    if metric == "n_unique_precursors" and "cohort_median_unique_precursors" in prov:
        median_val = float(prov["cohort_median_unique_precursors"])
        fig.add_hline(
            y=median_val,
            line=dict(color=QC_REFERENCE_LINE_GREY, width=1, dash="dash"),
            annotation_text=f"cohort median = {median_val:,.0f}",
            annotation_position="top left",
        )

    n_flagged = int(flagged.sum())
    tick_labels = _resolve_sample_labels(sample_labels, df.index)
    xaxis_cfg: dict[str, Any] = dict(
        title="condition" if tick_labels is not None else "sample", tickangle=90
    )
    if tick_labels is not None:
        xaxis_cfg["tickmode"] = "array"
        xaxis_cfg["tickvals"] = list(df.index.astype(str))
        xaxis_cfg["ticktext"] = tick_labels
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        width=width,
        height=height,
        font=dict(size=DEFAULT_FONT_SIZE_DASHBOARD),
        showlegend=False,
        xaxis=xaxis_cfg,
        yaxis=dict(title=metric.replace("_", " "), rangemode="tozero"),
        title=(
            f"Identification depth ({metric}) -- {len(df)} samples, "
            f"{n_flagged} flagged (z < {flag_z_threshold})"
        ),
        margin=dict(l=60, r=30, t=60, b=140),
    )
    return fig


def panel_psm_counts_by_order(
    psm_df: pd.DataFrame | None = None,
    *,
    counts_df: pd.DataFrame | None = None,
    acquisition_queue: pd.DataFrame | None = None,
    sample_labels: dict[str, str] | pd.Series | pd.DataFrame | None = None,
    metric: str = "n_unique_precursors",
    flag_z_threshold: float = DEFAULT_DEPTH_FLAG_Z,
    width: int = 900,
    height: int = 400,
    **compute_kwargs: Any,
) -> _go.Figure:
    """Same bar chart as :func:`panel_psm_counts_summary` but on the
    injection-order axis.

    Requires ``counts_df`` with ``injection_order`` (from a compute
    call with ``acquisition_queue=...``) or a fresh
    ``psm_df + acquisition_queue`` pair.  Reveals order-dependent
    identification-depth drift (column ageing, sample-prep batches).
    """
    _ = _require_plotly()
    df = _resolve_counts_df(
        psm_df=psm_df,
        counts_df=counts_df,
        acquisition_queue=acquisition_queue,
        flag_z_threshold=flag_z_threshold,
        **compute_kwargs,
    )
    if "injection_order" not in df.columns or df["injection_order"].isna().all():
        raise ValueError(
            "panel_psm_counts_by_order needs injection_order.  "
            "Pass acquisition_queue= to the counts computation, or use "
            "panel_psm_counts_summary for the order-agnostic panel."
        )
    if metric not in df.columns:
        raise KeyError(
            f"metric={metric!r} not in counts DataFrame; "
            f"available: {sorted(c for c in df.columns if c.startswith('n_'))}"
        )

    import plotly.graph_objects as go

    df = df.dropna(subset=["injection_order"]).sort_values("injection_order")
    flagged = df["is_flagged"].fillna(False).astype(bool).to_numpy()
    colours = [QC_FLAG_COLOUR if f else QC_NEUTRAL_COLOUR for f in flagged]
    values = df[metric].to_numpy()
    labels = _resolve_sample_labels(sample_labels, df.index)
    label_map = dict(zip(df.index.astype(str), labels, strict=False)) if labels else {}
    hover = [
        (
            f"<b>{sample}</b><br>"
            + (f"condition: {label_map[str(sample)]}<br>" if str(sample) in label_map else "")
            + f"injection_order: {int(row.injection_order)}<br>"
            f"{metric}: {int(row[metric]):,}<br>"
            f"z-score (MAD): {row.depth_z_score:+.2f}"
        )
        for sample, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["injection_order"].to_numpy(dtype=float),
            y=values,
            marker=dict(color=colours, line=DEFAULT_BAR_OUTLINE),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    # Cohort median reference line
    prov = df.attrs.get("provenance", {})
    if metric == "n_unique_precursors" and "cohort_median_unique_precursors" in prov:
        median_val = float(prov["cohort_median_unique_precursors"])
        fig.add_hline(
            y=median_val,
            line=dict(color=QC_REFERENCE_LINE_GREY, width=1, dash="dash"),
            annotation_text=f"cohort median = {median_val:,.0f}",
            annotation_position="top left",
        )

    n_flagged = int(flagged.sum())
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        width=width,
        height=height,
        font=dict(size=DEFAULT_FONT_SIZE_DASHBOARD),
        showlegend=False,
        xaxis=dict(title="injection order"),
        yaxis=dict(title=metric.replace("_", " "), rangemode="tozero"),
        title=(
            f"Identification depth ({metric}) vs acquisition order -- "
            f"{len(df)} samples, {n_flagged} flagged"
        ),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


def _resolve_contam_df(
    *,
    psm_df: pd.DataFrame | None,
    contam_df: pd.DataFrame | None,
    acquisition_queue: pd.DataFrame | None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Accept either pre-computed contam_df or recompute from psm_df."""
    if contam_df is not None:
        return contam_df
    if psm_df is None:
        raise ValueError("must provide either contam_df (pre-computed) or psm_df (recompute)")
    return compute_contaminant_fraction_per_sample(
        psm_df, acquisition_queue=acquisition_queue, **kwargs
    )


def panel_contaminant_fraction_summary(
    psm_df: pd.DataFrame | None = None,
    *,
    contam_df: pd.DataFrame | None = None,
    acquisition_queue: pd.DataFrame | None = None,
    sample_labels: dict[str, str] | pd.Series | pd.DataFrame | None = None,
    flag_fraction_threshold: float = DEFAULT_CONTAMINANT_FLAG_FRACTION,
    width: int = 900,
    height: int = 400,
    **compute_kwargs: Any,
) -> _go.Figure:
    """Bar chart of per-sample contaminant fraction (alphabetical order).

    Samples with ``contaminant_fraction > flag_fraction_threshold``
    are drawn in the QC-flag colour. Reference line at the flag
    threshold (dashed grey); cohort median annotated in the top-left.
    """
    go = _require_plotly()
    df = _resolve_contam_df(
        psm_df=psm_df,
        contam_df=contam_df,
        acquisition_queue=acquisition_queue,
        flag_fraction_threshold=flag_fraction_threshold,
        **compute_kwargs,
    )
    df = df.sort_index()

    flagged = df["is_flagged"].fillna(False).astype(bool).to_numpy()
    colours = [QC_FLAG_COLOUR if f else QC_NEUTRAL_COLOUR for f in flagged]
    hover = [
        (
            f"<b>{sample}</b><br>"
            f"contaminant fraction: {row.contaminant_fraction:.3f}"
            f" ({row.contaminant_fraction * 100:.1f}%)<br>"
            f"contaminant intensity: {row.contaminant_intensity:.3g}<br>"
            f"total intensity: {row.total_intensity:.3g}<br>"
            f"n contaminant precursors: {int(row.n_contaminant_precursors):,}"
            f" / {int(row.n_total_precursors):,}"
        )
        for sample, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(df.index.astype(str)),
            y=df["contaminant_fraction"].to_numpy(),
            marker=dict(color=colours, line=DEFAULT_BAR_OUTLINE),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.add_hline(
        y=float(flag_fraction_threshold),
        line=dict(color=QC_REFERENCE_LINE_GREY, width=1, dash="dash"),
        annotation_text=f"flag threshold = {flag_fraction_threshold:.2f}",
        annotation_position="top right",
    )
    prov = df.attrs.get("provenance", {})
    median_val = prov.get("cohort_median_fraction")
    if median_val is not None and pd.notna(median_val):
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
            text=f"cohort median = {median_val:.3f}",
            showarrow=False,
            font=dict(size=11, color=QC_REFERENCE_LINE_GREY),
        )

    tick_labels = _resolve_sample_labels(sample_labels, df.index)
    xaxis_cfg: dict[str, Any] = dict(
        title="condition" if tick_labels is not None else "sample", tickangle=90
    )
    if tick_labels is not None:
        xaxis_cfg["tickmode"] = "array"
        xaxis_cfg["tickvals"] = list(df.index.astype(str))
        xaxis_cfg["ticktext"] = tick_labels

    n_flagged = int(flagged.sum())
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        width=width,
        height=height,
        font=dict(size=DEFAULT_FONT_SIZE_DASHBOARD),
        showlegend=False,
        xaxis=xaxis_cfg,
        yaxis=dict(title="contaminant fraction", rangemode="tozero", tickformat=".0%"),
        title=(
            f"Contaminant fraction -- {len(df)} samples, "
            f"{n_flagged} flagged (>{flag_fraction_threshold * 100:.0f}%)"
        ),
        margin=dict(l=60, r=30, t=60, b=140),
    )
    return fig


def panel_contaminant_fraction_by_order(
    psm_df: pd.DataFrame | None = None,
    *,
    contam_df: pd.DataFrame | None = None,
    acquisition_queue: pd.DataFrame | None = None,
    sample_labels: dict[str, str] | pd.Series | pd.DataFrame | None = None,
    flag_fraction_threshold: float = DEFAULT_CONTAMINANT_FLAG_FRACTION,
    width: int = 900,
    height: int = 400,
    **compute_kwargs: Any,
) -> _go.Figure:
    """Contaminant fraction vs injection_order.

    Reveals order-dependent contamination -- late-run column carryover,
    prep-batch effects, blank-run bleed-through onto the next sample.
    Requires the acquisition queue to be present in the pre-computed
    ``contam_df`` (or a fresh ``psm_df + acquisition_queue`` pair).
    """
    go = _require_plotly()
    df = _resolve_contam_df(
        psm_df=psm_df,
        contam_df=contam_df,
        acquisition_queue=acquisition_queue,
        flag_fraction_threshold=flag_fraction_threshold,
        **compute_kwargs,
    )
    if "injection_order" not in df.columns or df["injection_order"].isna().all():
        raise ValueError(
            "panel_contaminant_fraction_by_order needs injection_order.  Pass "
            "acquisition_queue= to the compute call, or use "
            "panel_contaminant_fraction_summary for the order-agnostic panel."
        )

    df = df.dropna(subset=["injection_order"]).sort_values("injection_order")
    flagged = df["is_flagged"].fillna(False).astype(bool).to_numpy()
    colours = [QC_FLAG_COLOUR if f else QC_NEUTRAL_COLOUR for f in flagged]
    labels = _resolve_sample_labels(sample_labels, df.index)
    label_map = dict(zip(df.index.astype(str), labels, strict=False)) if labels else {}
    hover = [
        (
            f"<b>{sample}</b><br>"
            + (f"condition: {label_map[str(sample)]}<br>" if str(sample) in label_map else "")
            + f"injection_order: {int(row.injection_order)}<br>"
            f"contaminant fraction: {row.contaminant_fraction:.3f}"
            f" ({row.contaminant_fraction * 100:.1f}%)<br>"
            f"n contaminant precursors: {int(row.n_contaminant_precursors):,}"
            f" / {int(row.n_total_precursors):,}"
        )
        for sample, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["injection_order"].to_numpy(dtype=float),
            y=df["contaminant_fraction"].to_numpy(),
            marker=dict(color=colours, line=DEFAULT_BAR_OUTLINE),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.add_hline(
        y=float(flag_fraction_threshold),
        line=dict(color=QC_REFERENCE_LINE_GREY, width=1, dash="dash"),
        annotation_text=f"flag threshold = {flag_fraction_threshold:.2f}",
        annotation_position="top right",
    )

    n_flagged = int(flagged.sum())
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        width=width,
        height=height,
        font=dict(size=DEFAULT_FONT_SIZE_DASHBOARD),
        showlegend=False,
        xaxis=dict(title="injection order"),
        yaxis=dict(title="contaminant fraction", rangemode="tozero", tickformat=".0%"),
        title=(
            f"Contaminant fraction vs acquisition order -- {len(df)} samples, {n_flagged} flagged"
        ),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


def _resolve_tic_df(
    *,
    psm_df: pd.DataFrame | None,
    tic_df: pd.DataFrame | None,
    acquisition_queue: pd.DataFrame | None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Accept either pre-computed tic_df or recompute from psm_df."""
    if tic_df is not None:
        return tic_df
    if psm_df is None:
        raise ValueError("must provide either tic_df (pre-computed) or psm_df (recompute)")
    return compute_run_tic_per_sample(psm_df, acquisition_queue=acquisition_queue, **kwargs)


def panel_run_tic_summary(
    psm_df: pd.DataFrame | None = None,
    *,
    tic_df: pd.DataFrame | None = None,
    acquisition_queue: pd.DataFrame | None = None,
    sample_labels: dict[str, str] | pd.Series | pd.DataFrame | None = None,
    flag_z_threshold: float = DEFAULT_DEPTH_FLAG_Z,
    width: int = 900,
    height: int = 400,
    **compute_kwargs: Any,
) -> _go.Figure:
    """Bar chart of per-sample summed precursor intensity (log2 scale).

    Samples ordered alphabetically.  Flagged samples (log2 TIC below
    cohort median by more than ``flag_z_threshold`` MAD-SDs) are drawn
    in the QC-flag colour.  Reference line at the cohort log2 median.
    """
    go = _require_plotly()
    df = _resolve_tic_df(
        psm_df=psm_df,
        tic_df=tic_df,
        acquisition_queue=acquisition_queue,
        flag_z_threshold=flag_z_threshold,
        **compute_kwargs,
    )
    df = df.sort_index()

    flagged = df["is_flagged"].fillna(False).astype(bool).to_numpy()
    colours = [QC_FLAG_COLOUR if f else QC_NEUTRAL_COLOUR for f in flagged]
    hover = [
        (
            f"<b>{sample}</b><br>"
            f"total_intensity: {row.total_intensity:.3g}<br>"
            f"log2_total_intensity: {row.log2_total_intensity:.2f}<br>"
            f"median_precursor_intensity: {row.median_precursor_intensity:.3g}<br>"
            f"n_precursors_with_intensity: {int(row.n_precursors_with_intensity):,}<br>"
            f"TIC z-score (MAD): {row.tic_z_score:+.2f}"
        )
        for sample, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(df.index.astype(str)),
            y=df["log2_total_intensity"].to_numpy(),
            marker=dict(color=colours, line=DEFAULT_BAR_OUTLINE),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    prov = df.attrs.get("provenance", {})
    if "cohort_median_log2_total_intensity" in prov:
        median_val = float(prov["cohort_median_log2_total_intensity"])
        fig.add_hline(
            y=median_val,
            line=dict(color=QC_REFERENCE_LINE_GREY, width=1, dash="dash"),
            annotation_text=f"cohort log2 median = {median_val:.2f}",
            annotation_position="top left",
        )

    n_flagged = int(flagged.sum())
    tick_labels = _resolve_sample_labels(sample_labels, df.index)
    xaxis_cfg: dict[str, Any] = dict(
        title="condition" if tick_labels is not None else "sample", tickangle=90
    )
    if tick_labels is not None:
        xaxis_cfg["tickmode"] = "array"
        xaxis_cfg["tickvals"] = list(df.index.astype(str))
        xaxis_cfg["ticktext"] = tick_labels
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        width=width,
        height=height,
        font=dict(size=DEFAULT_FONT_SIZE_DASHBOARD),
        showlegend=False,
        xaxis=xaxis_cfg,
        yaxis=dict(title="log2(total precursor intensity)"),
        title=(
            f"Run TIC-proxy (log2 total precursor intensity) -- {len(df)} samples, "
            f"{n_flagged} flagged (z < {flag_z_threshold})"
        ),
        margin=dict(l=60, r=30, t=60, b=140),
    )
    return fig


def panel_run_tic_by_order(
    psm_df: pd.DataFrame | None = None,
    *,
    tic_df: pd.DataFrame | None = None,
    acquisition_queue: pd.DataFrame | None = None,
    sample_labels: dict[str, str] | pd.Series | pd.DataFrame | None = None,
    flag_z_threshold: float = DEFAULT_DEPTH_FLAG_Z,
    width: int = 900,
    height: int = 400,
    **compute_kwargs: Any,
) -> _go.Figure:
    """Same bar chart as :func:`panel_run_tic_summary` but on the
    injection-order axis.  Reveals order-dependent TIC drift (LC/MS
    stability, injection consistency)."""
    _ = _require_plotly()
    df = _resolve_tic_df(
        psm_df=psm_df,
        tic_df=tic_df,
        acquisition_queue=acquisition_queue,
        flag_z_threshold=flag_z_threshold,
        **compute_kwargs,
    )
    if "injection_order" not in df.columns or df["injection_order"].isna().all():
        raise ValueError(
            "panel_run_tic_by_order needs injection_order.  Pass "
            "acquisition_queue= to the TIC computation, or use "
            "panel_run_tic_summary for the order-agnostic panel."
        )

    import plotly.graph_objects as go

    df = df.dropna(subset=["injection_order"]).sort_values("injection_order")
    flagged = df["is_flagged"].fillna(False).astype(bool).to_numpy()
    colours = [QC_FLAG_COLOUR if f else QC_NEUTRAL_COLOUR for f in flagged]
    labels = _resolve_sample_labels(sample_labels, df.index)
    label_map = dict(zip(df.index.astype(str), labels, strict=False)) if labels else {}
    hover = [
        (
            f"<b>{sample}</b><br>"
            + (f"condition: {label_map[str(sample)]}<br>" if str(sample) in label_map else "")
            + f"injection_order: {int(row.injection_order)}<br>"
            f"log2_total_intensity: {row.log2_total_intensity:.2f}<br>"
            f"n_precursors: {int(row.n_precursors_with_intensity):,}<br>"
            f"TIC z-score (MAD): {row.tic_z_score:+.2f}"
        )
        for sample, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["injection_order"].to_numpy(dtype=float),
            y=df["log2_total_intensity"].to_numpy(),
            marker=dict(color=colours, line=DEFAULT_BAR_OUTLINE),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    prov = df.attrs.get("provenance", {})
    if "cohort_median_log2_total_intensity" in prov:
        median_val = float(prov["cohort_median_log2_total_intensity"])
        fig.add_hline(
            y=median_val,
            line=dict(color=QC_REFERENCE_LINE_GREY, width=1, dash="dash"),
            annotation_text=f"cohort log2 median = {median_val:.2f}",
            annotation_position="top left",
        )

    n_flagged = int(flagged.sum())
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        width=width,
        height=height,
        font=dict(size=DEFAULT_FONT_SIZE_DASHBOARD),
        showlegend=False,
        xaxis=dict(title="injection order"),
        yaxis=dict(title="log2(total precursor intensity)"),
        title=(f"Run TIC-proxy vs acquisition order -- {len(df)} samples, {n_flagged} flagged"),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


def panel_retention_time_drift_summary(
    psm_df: pd.DataFrame | None = None,
    *,
    drift_df: pd.DataFrame | None = None,
    acquisition_queue: pd.DataFrame | None = None,
    sample_labels: dict[str, str] | pd.Series | pd.DataFrame | None = None,
    offset_flag_threshold: float = DEFAULT_OFFSET_FLAG_THRESHOLD_MIN,
    width: int = 900,
    height: int = 400,
    **compute_kwargs: Any,
) -> _go.Figure:
    """One point per sample -- median RT offset from cohort median.

    Samples ordered alphabetically on the x-axis.  Flagged samples
    (|median_offset| > ``offset_flag_threshold``) get the QC-flag colour
    outline.  Reference lines at ``y=0`` (solid black) and
    ``y=±threshold`` (dashed grey).

    Renders even without an acquisition queue -- use
    :func:`panel_retention_time_drift_by_order` when the queue is
    available for order-aware diagnostics.
    """
    go = _require_plotly()
    df = _resolve_drift_df(
        psm_df=psm_df,
        drift_df=drift_df,
        acquisition_queue=acquisition_queue,
        offset_flag_threshold=offset_flag_threshold,
        **compute_kwargs,
    )
    # Alphabetical sample ordering
    df = df.sort_index()

    flagged = df["is_flagged"].fillna(False).astype(bool).to_numpy()
    colours = [QC_FLAG_COLOUR if f else QC_NEUTRAL_COLOUR for f in flagged]
    hover = [
        (
            f"<b>{sample}</b><br>"
            f"median offset: {row.median_offset:.3f} min<br>"
            f"IQR offset: {row.iqr_offset:.3f} min<br>"
            f"gradient slope: {row.gradient_slope:.4f}<br>"
            f"n shared precursors: {int(row.n_shared_precursors)}"
        )
        for sample, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(df.index.astype(str)),
            y=df["median_offset"].to_numpy(),
            mode="markers",
            marker=dict(
                size=10,
                color=colours,
                line=DEFAULT_MARKER_OUTLINE,
            ),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    # y=0 reference (solid black)
    fig.add_hline(y=0, line=dict(color="black", width=1))
    # ±threshold references (dashed grey)
    fig.add_hline(
        y=float(offset_flag_threshold),
        line=dict(color=QC_REFERENCE_LINE_GREY, width=1, dash="dash"),
    )
    fig.add_hline(
        y=-float(offset_flag_threshold),
        line=dict(color=QC_REFERENCE_LINE_GREY, width=1, dash="dash"),
    )

    n_flagged = int(flagged.sum())
    tick_labels = _resolve_sample_labels(sample_labels, df.index)
    xaxis_cfg: dict[str, Any] = dict(
        title="condition" if tick_labels is not None else "sample", tickangle=90
    )
    if tick_labels is not None:
        xaxis_cfg["tickmode"] = "array"
        xaxis_cfg["tickvals"] = list(df.index.astype(str))
        xaxis_cfg["ticktext"] = tick_labels
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        width=width,
        height=height,
        font=dict(size=DEFAULT_FONT_SIZE_DASHBOARD),
        showlegend=False,
        xaxis=xaxis_cfg,
        yaxis=dict(title="median RT offset from cohort (min)"),
        title=(
            f"Retention-time drift -- {len(df)} samples, {n_flagged} flagged "
            f"(|offset| > {offset_flag_threshold} min)"
        ),
        margin=dict(l=60, r=30, t=60, b=140),
    )
    return fig


def panel_retention_time_drift_by_order(
    psm_df: pd.DataFrame | None = None,
    *,
    drift_df: pd.DataFrame | None = None,
    acquisition_queue: pd.DataFrame | None = None,
    sample_labels: dict[str, str] | pd.Series | pd.DataFrame | None = None,
    offset_flag_threshold: float = DEFAULT_OFFSET_FLAG_THRESHOLD_MIN,
    width: int = 900,
    height: int = 400,
    **compute_kwargs: Any,
) -> _go.Figure:
    """Median RT offset vs injection_order -- catches column-ageing drift.

    Requires either a ``drift_df`` that already carries
    ``injection_order`` (from a drift computation with
    ``acquisition_queue=...``) or a ``psm_df`` + ``acquisition_queue``.

    Adds a light-grey linear fit line when the Spearman correlation
    (order vs offset) is significant (p < 0.05).  Spearman ρ + p-value
    is annotated in the top-right.
    """
    go = _require_plotly()
    df = _resolve_drift_df(
        psm_df=psm_df,
        drift_df=drift_df,
        acquisition_queue=acquisition_queue,
        offset_flag_threshold=offset_flag_threshold,
        **compute_kwargs,
    )
    if "injection_order" not in df.columns or df["injection_order"].isna().all():
        raise ValueError(
            "panel_retention_time_drift_by_order needs injection_order.  "
            "Pass acquisition_queue= to the drift computation, or use "
            "panel_retention_time_drift_summary for the order-agnostic panel."
        )
    df = df.dropna(subset=["injection_order"]).sort_values("injection_order")

    flagged = df["is_flagged"].fillna(False).astype(bool).to_numpy()
    colours = [QC_FLAG_COLOUR if f else QC_NEUTRAL_COLOUR for f in flagged]
    labels = _resolve_sample_labels(sample_labels, df.index)
    label_map = dict(zip(df.index.astype(str), labels, strict=False)) if labels else {}
    hover = [
        (
            f"<b>{sample}</b><br>"
            + (f"condition: {label_map[str(sample)]}<br>" if str(sample) in label_map else "")
            + f"injection_order: {int(row.injection_order)}<br>"
            f"median offset: {row.median_offset:.3f} min<br>"
            f"n shared precursors: {int(row.n_shared_precursors)}"
        )
        for sample, row in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["injection_order"].to_numpy(dtype=float),
            y=df["median_offset"].to_numpy(),
            mode="markers",
            marker=dict(
                size=10,
                color=colours,
                line=DEFAULT_MARKER_OUTLINE,
            ),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Linear fit line if Spearman is significant
    prov = df.attrs.get("provenance", {})
    spearman_r = prov.get("spearman_r_order_vs_offset")
    spearman_p = prov.get("spearman_p_value")
    slope = prov.get("pearson_r_order_vs_offset")  # for annotation only
    if spearman_p is not None and not pd.isna(spearman_p) and float(spearman_p) < 0.05:
        import numpy as np

        x = df["injection_order"].to_numpy(dtype=float)
        y = df["median_offset"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= 2:
            x_f, y_f = x[finite], y[finite]
            fit_slope, fit_intercept = np.polyfit(x_f, y_f, 1)
            xs = np.array([x_f.min(), x_f.max()])
            ys = fit_slope * xs + fit_intercept
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=QC_REFERENCE_LINE_GREY, width=1.5, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Reference lines (y=0, y=±threshold)
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.add_hline(
        y=float(offset_flag_threshold),
        line=dict(color=QC_REFERENCE_LINE_GREY, width=1, dash="dash"),
    )
    fig.add_hline(
        y=-float(offset_flag_threshold),
        line=dict(color=QC_REFERENCE_LINE_GREY, width=1, dash="dash"),
    )

    # Corner annotation: Spearman ρ + p
    annotation_lines = []
    if spearman_r is not None and not pd.isna(spearman_r):
        annotation_lines.append(f"Spearman ρ = {float(spearman_r):+.3f}")  # noqa: RUF001
    if spearman_p is not None and not pd.isna(spearman_p):
        annotation_lines.append(f"p = {float(spearman_p):.2e}")
    if slope is not None and not pd.isna(slope):
        annotation_lines.append(f"Pearson r = {float(slope):+.3f}")
    if annotation_lines:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            text="<br>".join(annotation_lines),
            showarrow=False,
            font=dict(family="Arial Black", size=11, color="black"),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=4,
        )

    n_flagged = int(flagged.sum())
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        width=width,
        height=height,
        font=dict(size=DEFAULT_FONT_SIZE_DASHBOARD),
        showlegend=False,
        xaxis=dict(title="injection order"),
        yaxis=dict(title="median RT offset from cohort (min)"),
        title=(
            f"Retention-time drift vs acquisition order -- {len(df)} samples, {n_flagged} flagged"
        ),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig
