"""Shared plumbing for :mod:`alphaphos.viz` figure builders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd

from alphaphos.enrichment.matching import parse_alphaphos_key
from alphaphos.viz.style import (
    AXIS_TITLE_FONT,
    LAYOUT_BASE,
    PLOT_HEIGHT,
    TICK_FONT,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go


def require_plotly() -> Any:
    """Return ``plotly.graph_objects``, or raise pointing at the ``[viz]`` extra."""
    try:
        import plotly.graph_objects as go

        return go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for alphaphos.viz.  Install with `pip install 'alphaphos[viz]'`."
        ) from exc


def layout(width: int | None = None, **overrides: Any) -> dict:
    """``LAYOUT_BASE`` with an optional width; height is never overridable."""
    out = {**LAYOUT_BASE, **overrides}
    if width is not None:
        out["width"] = width
    out["height"] = PLOT_HEIGHT
    return out


def style_axes(fig: go.Figure, x_title: str, y_title: str) -> go.Figure:
    """Axis titles and ticks by role; no gridlines; zero-lines solid black."""
    common = {
        "tickfont": TICK_FONT,
        "showgrid": False,
        "zerolinecolor": "black",
        "zerolinewidth": 1,
    }
    fig.update_xaxes(title={"text": x_title, "font": AXIS_TITLE_FONT}, **common)
    fig.update_yaxes(title={"text": y_title, "font": AXIS_TITLE_FONT}, **common)
    return fig


def checked_table(df: pd.DataFrame, columns: tuple[str, ...], what: str) -> pd.DataFrame:
    """Validate ``columns`` and return a shallow copy of ``df`` with ``attrs`` dropped.

    Result tables carry provenance in ``attrs``; if an entry is ever a numpy
    array, pandas' attrs comparison inside ``concat`` (reached from every
    ``nsmallest`` / ``nlargest``) raises.  Plotting never needs the
    provenance, so it is dropped up front.
    """
    if isinstance(df, tuple):
        raise TypeError(
            f"{what} must be a DataFrame, got a tuple -- diff_exp_limma_observed_only "
            "returns (limma_result, on_off_table); pass the first element."
        )
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{what} lacks column(s) {missing}; available: {list(df.columns)}")
    out = df.copy(deep=False)
    out.attrs = {}
    return out


def site_label(key: str) -> str:
    """``P31749|AKT1|S473|1`` -> ``AKT1 S473``.

    ``M2`` etc. is appended for multiplicity > 1; unparseable keys pass through.
    """
    parsed = parse_alphaphos_key(key)
    if parsed is None:
        return str(key)
    gene = parsed.gene.split(";", 1)[0] or parsed.protein
    label = f"{gene} {parsed.residue}{parsed.position}"
    return label if parsed.multiplicity == 1 else f"{label} M{parsed.multiplicity}"


def group_order(values: pd.Series) -> list[str]:
    """Category order: declared order for a Categorical, else sorted labels."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        present = set(values.astype(str))
        return [str(c) for c in values.cat.categories if str(c) in present]
    return sorted(values.dropna().astype(str).unique())


def obs_column(adata: ad.AnnData, column: str) -> pd.Series:
    if column not in adata.obs.columns:
        raise KeyError(f"{column!r} not in adata.obs; available: {list(adata.obs.columns)}")
    return adata.obs[column]


def matrix(adata: ad.AnnData, layer: str | None) -> np.ndarray:
    """Dense float copy of ``adata.X`` or ``adata.layers[layer]``."""
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"layer {layer!r} not in adata.layers; available: {list(adata.layers)}")
    x = adata.X if layer is None else adata.layers[layer]
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def save_figure(fig: go.Figure, path: str | Path) -> Path:
    """Write ``fig`` in the single format implied by the file suffix.

    ``.html`` writes a self-contained interactive page; ``.pdf`` / ``.svg``
    / ``.png`` write a static image via kaleido at the figure's own width
    and height.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".html":
        # Embedded plotly.js (~4 MB) so the file opens offline on a collaborator's laptop.
        fig.write_html(path, include_plotlyjs=True)
    elif suffix in (".pdf", ".svg", ".png"):
        fig.write_image(path, width=fig.layout.width, height=fig.layout.height)
    else:
        raise ValueError(f"unsupported figure format {suffix!r}; use .html, .pdf, .svg or .png")
    return path
