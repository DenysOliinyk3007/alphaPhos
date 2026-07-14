"""bokeh figure builders for each QC dashboard section.

Each ``plot_*`` takes the corresponding metrics DataFrame (from
:mod:`alphaphos.qc.metrics`) and returns a ``bokeh.plotting.figure``.
Designed to be composed by :mod:`alphaphos.qc.dashboard` but also usable
standalone (e.g. ``show(plot_sty_ratio(df))`` in a notebook).

Color palette: an extended Mann-Lab-ish palette that handles up to ~10
conditions without recycling. For more than that, palettes silently cycle.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# bokeh is an optional ``[qc]`` extra.  Guard the imports so ``import
# alphaphos.qc.plots`` succeeds on a bare-core install; the individual
# plot builders below raise a friendly error when actually called.
try:
    from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
    from bokeh.palettes import Category10_10, Category20_20, Viridis256
    from bokeh.plotting import figure

    _BOKEH_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:  # pragma: no cover - covered by CI bare-core matrix
    ColumnDataSource = HoverTool = Legend = LegendItem = None  # type: ignore[assignment]
    Category10_10 = Category20_20 = Viridis256 = None  # type: ignore[assignment]
    figure = None  # type: ignore[assignment]
    _BOKEH_IMPORT_ERROR = _exc


def _require_bokeh() -> None:
    if _BOKEH_IMPORT_ERROR is not None:
        raise ImportError(
            "alphaphos.qc.plots requires bokeh.  Install with `pip install 'alphaphos[qc]'`."
        ) from _BOKEH_IMPORT_ERROR


# factor_cmap removed — pre-computed color columns are more robust in the
# presence of null/unexpected values in the data source

# A small "kpi color" set picked to be print-friendly + colorblind-OK
_AA_COLORS = {"S": "#1f77b4", "T": "#2ca02c", "Y": "#d62728"}
_MULT_COLORS = {"1": "#a6cee3", "2": "#1f78b4", "3+": "#08306b"}


def _palette_for(categories: list[str]) -> list[str]:
    """Pick a categorical palette sized to the number of categories."""
    n = len(categories)
    if n <= 10:
        return list(Category10_10)[:n]
    return list(Category20_20)[: min(n, 20)]


# ---------------------------------------------------------------------------
# §1. Pipeline waterfall
# ---------------------------------------------------------------------------


def plot_pipeline_waterfall(df: pd.DataFrame, *, width: int = 700, height: int = 320):
    """Bar chart: PSM row count at each pipeline step."""
    if df.empty:
        return _empty_figure("Pipeline waterfall: no data", width=width, height=height)
    stages = df["stage"].tolist()
    p = figure(
        x_range=stages,
        width=width,
        height=height,
        title="§1 Pipeline waterfall — rows / sites at each step",
        toolbar_location="above",
        tools="pan,wheel_zoom,reset,save",
    )
    src = ColumnDataSource(df)
    p.vbar(x="stage", top="n_rows", width=0.7, source=src, fill_color="#4682b4", line_color=None)
    p.xaxis.major_label_orientation = math.pi / 4
    p.yaxis.axis_label = "rows / sites"
    p.add_tools(
        HoverTool(
            tooltips=[
                ("stage", "@stage"),
                ("n", "@n_rows{0,0}"),
                ("dropped vs prev", "@dropped_from_prev{0,0}"),
                ("% dropped", "@pct_dropped_from_prev{0.00}%"),
            ]
        )
    )
    return p


# ---------------------------------------------------------------------------
# §2. Sample QC plots
# ---------------------------------------------------------------------------


def plot_sty_ratio(df: pd.DataFrame, *, width: int = 600, height: int = 360):
    """Stacked-bar S/T/Y composition per sample (manual stack — no transforms)."""
    if df.empty:
        return _empty_figure("S/T/Y ratio: no site_aa metadata", width=width, height=height)
    pivot = df.pivot(index="sample", columns="site_aa", values="pct").fillna(0.0)
    samples = list(pivot.index)
    aas = ["S", "T", "Y"]
    p = figure(
        x_range=samples,
        width=width,
        height=height,
        title="§2 S/T/Y composition (% of detected sites per sample)",
        tools="pan,wheel_zoom,reset,save",
    )
    cumsum = pd.Series(0.0, index=samples)
    for aa in aas:
        vals = pivot.get(aa, pd.Series(0.0, index=samples)).fillna(0.0).astype(float)
        top = (cumsum + vals).tolist()
        bot = cumsum.tolist()
        p.vbar(
            x=samples,
            bottom=bot,
            top=top,
            width=0.8,
            color=_AA_COLORS[aa],
            legend_label=aa,
            line_color=None,
        )
        cumsum = cumsum + vals
    p.y_range.start = 0
    p.y_range.end = 100
    p.yaxis.axis_label = "% of detected sites"
    p.xaxis.major_label_orientation = math.pi / 4
    p.legend.location = "right"
    p.legend.click_policy = "hide"
    return p


def plot_localization_distribution(df: pd.DataFrame, *, width: int = 700, height: int = 360):
    """Per-sample loc-prob density (overlapping density curves, one per sample)."""
    if df.empty:
        return _empty_figure(
            "Localization distribution: no 'localization' layer", width=width, height=height
        )
    samples = sorted(df["sample"].unique())
    colors = _palette_for(samples)
    p = figure(
        width=width,
        height=height,
        title="§2 Localization-probability density per sample",
        tools="pan,wheel_zoom,reset,save,hover",
    )
    grid = np.linspace(0, 1, 100)
    for s, color in zip(samples, colors, strict=False):
        vals = df.loc[df["sample"] == s, "loc_prob"].values
        if len(vals) < 2:
            continue
        # KDE via numpy histogram (avoid scipy dep)
        hist, edges = np.histogram(vals, bins=grid, density=True)
        x = 0.5 * (edges[:-1] + edges[1:])
        p.line(x, hist, line_width=1.5, color=color, legend_label=s, alpha=0.7)
    p.xaxis.axis_label = "localization probability"
    p.yaxis.axis_label = "density"
    p.x_range.start = 0
    p.x_range.end = 1
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "8pt"
    return p


def plot_multiplicity(df: pd.DataFrame, *, width: int = 600, height: int = 360):
    """Stacked-bar multiplicity (M1/M2/M3+) composition per sample (manual stack)."""
    if df.empty:
        return _empty_figure("Multiplicity: no 'multiplicity' metadata", width=width, height=height)
    pivot = df.pivot(index="sample", columns="multiplicity", values="pct").fillna(0.0)
    samples = list(pivot.index)
    mults = ["1", "2", "3+"]
    p = figure(
        x_range=samples,
        width=width,
        height=height,
        title="§2 Multiplicity composition per sample",
        tools="pan,wheel_zoom,reset,save",
    )
    cumsum = pd.Series(0.0, index=samples)
    for m in mults:
        vals = pivot.get(m, pd.Series(0.0, index=samples)).fillna(0.0).astype(float)
        top = (cumsum + vals).tolist()
        bot = cumsum.tolist()
        p.vbar(
            x=samples,
            bottom=bot,
            top=top,
            width=0.8,
            color=_MULT_COLORS[m],
            legend_label=f"M{m}",
            line_color=None,
        )
        cumsum = cumsum + vals
    p.y_range.start = 0
    p.y_range.end = 100
    p.yaxis.axis_label = "% of detected sites"
    p.xaxis.major_label_orientation = math.pi / 4
    p.legend.location = "right"
    return p


def plot_missingness(df: pd.DataFrame, *, width: int = 600, height: int = 360):
    """Bar plot of per-sample %-missing, colored by condition if present."""
    if df.empty:
        return _empty_figure("Missingness: empty data", width=width, height=height)
    samples = df["sample"].tolist()
    p = figure(
        x_range=samples,
        width=width,
        height=height,
        title="§2 Missing-value rate per sample",
        tools="pan,wheel_zoom,reset,save",
    )
    has_cond = "condition" in df.columns
    if has_cond:
        conditions = sorted(df["condition"].astype(str).unique())
        palette = _palette_for(conditions)
        color_map = {c: palette[i] for i, c in enumerate(conditions)}
        df = df.copy()
        df["color"] = df["condition"].astype(str).map(color_map)
        src = ColumnDataSource(df)
        p.vbar(x="sample", top="pct_missing", width=0.7, source=src, color="color", line_color=None)
        # Build legend via LegendItem with no renderers — dummy glyphs at x=[None]
        # crash bokeh's factor-synthetic transform when the figure has a
        # categorical x_range (TypeError: Cannot read properties of null).
        items = []
        for cond in conditions:
            r = p.rect(
                x=samples[:1],
                y=[0],
                width=0,
                height=0,
                fill_color=color_map[cond],
                line_color=None,
                alpha=0,
            )
            items.append(LegendItem(label=cond, renderers=[r]))
        p.add_layout(Legend(items=items, location="center"), "right")
    else:
        src = ColumnDataSource(df)
        p.vbar(
            x="sample", top="pct_missing", width=0.7, source=src, color="#888888", line_color=None
        )
    p.yaxis.axis_label = "% missing"
    p.xaxis.major_label_orientation = math.pi / 4
    p.add_tools(
        HoverTool(
            tooltips=[
                ("sample", "@sample"),
                ("missing", "@n_missing / @n_total"),
                ("%", "@pct_missing{0.00}%"),
            ]
        )
    )
    return p


def plot_n_classI(df: pd.DataFrame, *, width: int = 600, height: int = 360):
    """Bar plot of n_classI per sample with fraction overlay."""
    if df.empty:
        return _empty_figure(
            "Class I per sample: no localization layer", width=width, height=height
        )
    src = ColumnDataSource(df)
    p = figure(
        x_range=df["sample"].tolist(),
        width=width,
        height=height,
        title="§2 Class I sites per sample (loc >= 0.75)",
        tools="pan,wheel_zoom,reset,save",
    )
    p.vbar(x="sample", top="n_classI", width=0.7, source=src, color="#2ca02c", line_color=None)
    p.yaxis.axis_label = "# Class I sites"
    p.xaxis.major_label_orientation = math.pi / 4
    p.add_tools(
        HoverTool(
            tooltips=[
                ("sample", "@sample"),
                ("Class I", "@n_classI{0,0}"),
                ("present total", "@n_present{0,0}"),
                ("fraction Class I", "@fraction_classI{0.00}"),
            ]
        )
    )
    return p


def plot_per_condition_cv_by_aa(df: pd.DataFrame, *, width: int = 600, height: int = 360):
    """Boxplot-style CV-per-condition x per-S/T/Y distribution.

    Since bokeh has no native boxplot, we approximate via quartile bars.
    """
    if df.empty:
        return _empty_figure(
            "Per-condition CV: needs 'condition' and 'site_aa'", width=width, height=height
        )
    # Drop any NaN cv before quantile calc (otherwise factor labels with NaN crash factor_cmap)
    df = df.dropna(subset=["cv", "condition", "site_aa"]).copy()
    df["condition"] = df["condition"].astype(str)
    df["site_aa"] = df["site_aa"].astype(str)
    if df.empty:
        return _empty_figure("Per-condition CV: no usable data", width=width, height=height)
    grouped = df.groupby(["condition", "site_aa"])["cv"]
    q = grouped.quantile([0.25, 0.5, 0.75]).unstack()
    q.columns = ["q1", "median", "q3"]
    q["lower"] = grouped.quantile(0.05).values
    q["upper"] = grouped.quantile(0.95).values
    q = q.reset_index()
    q["x"] = q["condition"].astype(str) + " / " + q["site_aa"].astype(str)
    # Pre-compute color column rather than relying on factor_cmap (which crashes
    # on any unexpected/null value in the data source)
    palette = _palette_for(q["x"].tolist())
    q["color"] = (
        palette[: len(q)]
        if len(q) <= len(palette)
        else (palette * (len(q) // len(palette) + 1))[: len(q)]
    )
    src = ColumnDataSource(q)
    p = figure(
        x_range=q["x"].tolist(),
        width=width,
        height=height,
        title="§2 Within-condition CV by site_aa (linear-space, p5-p95 + IQR)",
        tools="pan,wheel_zoom,reset,save",
    )
    # IQR box
    p.vbar(
        x="x",
        top="q3",
        bottom="q1",
        width=0.5,
        source=src,
        fill_color="color",
        line_color="black",
    )
    # Median dash
    p.segment(
        x0="x",
        y0="median",
        x1="x",
        y1="median",
        source=src,
        line_color="black",
        line_width=3,
    )
    # Whiskers
    p.segment(x0="x", y0="lower", x1="x", y1="q1", source=src, line_color="black")
    p.segment(x0="x", y0="q3", x1="x", y1="upper", source=src, line_color="black")
    p.yaxis.axis_label = "CV (linear scale)"
    p.xaxis.major_label_orientation = math.pi / 4
    p.add_tools(
        HoverTool(
            tooltips=[
                ("group", "@x"),
                ("median", "@median{0.00}"),
                ("IQR", "@q1{0.00}-@q3{0.00}"),
            ]
        )
    )
    return p


# ---------------------------------------------------------------------------
# §3. Reproducibility
# ---------------------------------------------------------------------------


def plot_replicate_correlation(
    corr_df: pd.DataFrame,
    *,
    sample_order: list[str] | None = None,
    width: int = 600,
    height: int = 600,
):
    """Pearson r heatmap between samples (optionally re-ordered).

    Uses pre-computed color column (not a bokeh `LinearColorMapper`) — the
    latter applies a synthetic transform client-side that crashes on
    null/NaN values during glyph initialization.
    """
    if corr_df.empty:
        return _empty_figure("Replicate correlation: empty", width=width, height=height)
    if sample_order is not None:
        keep = [s for s in sample_order if s in corr_df.index]
        corr_df = corr_df.loc[keep, keep]
    samples = list(corr_df.index.astype(str))
    # Long form
    df_long = corr_df.stack(dropna=False).reset_index()
    df_long.columns = ["sample_y", "sample_x", "r"]
    df_long["sample_x"] = df_long["sample_x"].astype(str)
    df_long["sample_y"] = df_long["sample_y"].astype(str)
    # Drop NaN r values entirely (we don't render those cells)
    df_long = df_long.dropna(subset=["r"]).copy()
    # Pre-compute fill_color per cell: linear interpolation Viridis256[low..high]
    low, high = 0.5, 1.0
    n_colors = len(Viridis256)

    def _color_for(r: float) -> str:
        if r is None or pd.isna(r):
            return "#cccccc"
        t = max(0.0, min(1.0, (r - low) / (high - low)))
        i = round(t * (n_colors - 1))
        return Viridis256[i]

    df_long["color"] = df_long["r"].map(_color_for)
    src = ColumnDataSource(df_long)
    p = figure(
        x_range=samples,
        y_range=list(reversed(samples)),
        width=width,
        height=height,
        title="§3 Replicate correlation (Pearson r)",
        tools="pan,wheel_zoom,reset,save",
    )
    p.rect(
        x="sample_x",
        y="sample_y",
        width=1,
        height=1,
        source=src,
        fill_color="color",
        line_color=None,
    )
    p.add_tools(HoverTool(tooltips=[("x", "@sample_x"), ("y", "@sample_y"), ("r", "@r{0.000}")]))
    p.xaxis.major_label_orientation = math.pi / 4
    return p


# ---------------------------------------------------------------------------
# §4. Class I diagnostics
# ---------------------------------------------------------------------------


def plot_classI_comparison(df: pd.DataFrame, *, width: int = 500, height: int = 320):
    """Side-by-side cell counts: per_cell_binary vs condition_aware."""
    if df.empty:
        return _empty_figure("Class I comparison: no data", width=width, height=height)
    policies = df["policy"].tolist()
    palette = ["#4682b4", "#2ca02c", "#d62728", "#ff7f0e"]
    df = df.copy()
    df["color"] = [palette[i % len(palette)] for i in range(len(df))]
    src = ColumnDataSource(df)
    p = figure(
        x_range=policies,
        width=width,
        height=height,
        title="§4 Class I masking — cells retained per policy",
        tools="pan,wheel_zoom,reset,save",
    )
    p.vbar(x="policy", top="n_cells", width=0.6, source=src, color="color", line_color=None)
    p.yaxis.axis_label = "# (site, sample) cells retained"
    p.xaxis.major_label_orientation = math.pi / 8
    p.add_tools(HoverTool(tooltips=[("policy", "@policy"), ("n_cells", "@n_cells{0,0}")]))
    return p


def plot_contaminant_breakdown(df: pd.DataFrame, *, width: int = 600, height: int = 360):
    """Horizontal bar of top contaminant genes by PSM-row count."""
    if df.empty:
        return _empty_figure(
            "Contaminant breakdown: no contaminants (or psm_df not provided)",
            width=width,
            height=height,
        )
    df = df.sort_values("n_rows", ascending=True)
    src = ColumnDataSource(df)
    p = figure(
        y_range=df["gene"].tolist(),
        width=width,
        height=height,
        title="§4 Top contaminants dropped (PSM-row count)",
        tools="pan,wheel_zoom,reset,save",
    )
    p.hbar(y="gene", right="n_rows", height=0.7, source=src, color="#d62728", line_color=None)
    p.xaxis.axis_label = "# PSM rows"
    p.add_tools(HoverTool(tooltips=[("gene", "@gene"), ("n_rows", "@n_rows{0,0}")]))
    return p


# ---------------------------------------------------------------------------
# §5. Imputation diagnostics
# ---------------------------------------------------------------------------


def plot_imputation_summary(df: pd.DataFrame, *, width: int = 600, height: int = 320):
    """Per-sample bar showing MAR vs MNAR cell counts (manual stack)."""
    if df.empty:
        return _empty_figure(
            "Imputation diagnostics: provide impute_audit (return_audit=True)",
            width=width,
            height=height,
        )
    pivot = df.pivot(index="sample_idx", columns="strategy", values="n").fillna(0.0)
    samples = [str(i) for i in pivot.index]
    strategies = list(pivot.columns)
    p = figure(
        x_range=samples,
        width=width,
        height=height,
        title="§5 Hybrid imputation — MAR vs MNAR cells per sample",
        tools="pan,wheel_zoom,reset,save",
    )
    color_map = {"MAR_KNN": "#1f77b4", "MNAR_Gaussian": "#d62728"}
    cumsum = pd.Series(0.0, index=pivot.index)
    for strat in strategies:
        vals = pivot[strat].fillna(0.0).astype(float)
        top = (cumsum + vals).tolist()
        bot = cumsum.tolist()
        p.vbar(
            x=samples,
            bottom=bot,
            top=top,
            width=0.8,
            color=color_map.get(strat, "#888888"),
            legend_label=strat,
            line_color=None,
        )
        cumsum = cumsum + vals
    p.yaxis.axis_label = "# imputed cells"
    p.xaxis.axis_label = "sample index"
    p.legend.location = "top_right"
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_figure(msg: str, *, width: int, height: int):
    p = figure(
        width=width,
        height=height,
        title=msg,
        toolbar_location=None,
        tools="",
        x_range=(0, 1),
        y_range=(0, 1),
    )
    p.text(
        x=[0.5],
        y=[0.5],
        text=[msg],
        text_align="center",
        text_baseline="middle",
        text_color="#888",
        text_font_size="10pt",
    )
    p.axis.visible = False
    p.grid.visible = False
    return p
