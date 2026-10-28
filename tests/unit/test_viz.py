"""Tests for :mod:`alphaphos.viz`.

Coverage:
  - Every builder honours the house style (template, geometry, outlines, no legend)
  - Volcano categories and the FDR line placed at the largest passing p-value
  - Palette choice: semantic > embedding (<= 5 groups) > condition fallback
  - Replicate points drawn below n = 100, omitted above
  - Heatmap colour bounds (symmetric z-score, percentile raw) and site cap
  - Enrichment dot plot accepts site-set and gene-set schemas
  - Result tables whose attrs hold numpy arrays (limma-trend prior) still plot
  - ``alphaphos.viz`` imports without plotly and fails with an install hint
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

import alphaphos as ap
from alphaphos.viz.style import (
    CONDITION_PALETTE,
    EMBEDDING_PALETTE,
    PLOT_HEIGHT,
    PLOT_WIDTH_MAX,
    TIMEPOINT_PALETTE,
    VOLCANO,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_adata(n_per_group: int = 4, groups=("ctrl", "trt"), n_sites: int = 60) -> ad.AnnData:
    rng = np.random.default_rng(0)
    n = n_per_group * len(groups)
    x = rng.normal(20, 2, size=(n, n_sites))
    x[rng.random(x.shape) < 0.1] = np.nan
    obs = pd.DataFrame(
        {"condition": np.repeat(groups, n_per_group)},
        index=[f"20260101_QE_run_{g}_R{i}" for g in groups for i in range(n_per_group)],
    )
    var = pd.DataFrame(index=[f"P{j:05d}|GENE{j}|S{j + 10}|1" for j in range(n_sites)])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.obsm["X_pca"] = rng.normal(size=(n, 3))
    adata.uns["pca"] = {"variance_ratio": np.array([0.4, 0.2, 0.1])}
    return adata


def _make_de(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    p = rng.uniform(1e-8, 1, n)
    p[: n // 5] = rng.uniform(1e-8, 1e-4, n // 5)  # a block of true hits
    df = pd.DataFrame(
        {"log2fc": rng.normal(0, 1.5, n), "p_value": p, "t_stat": rng.normal(0, 3, n)},
        index=[f"P{j:05d}|GENE{j}|S{j + 10}|1" for j in range(n)],
    )
    order = np.argsort(p)
    fdr = np.empty(n)
    fdr[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    df["fdr"] = np.minimum(fdr, 1)
    return df


KSEA = pd.DataFrame(
    {
        "kinase": ["AKT1", "MAPK1", "CDK1", "SRC"],
        "score": [3.0, -2.0, 1.5, -4.0],
        "fdr": [0.001, 0.01, 0.2, 0.0001],
        "n_substrates": [12, 30, 8, 5],
    }
)
SITE_ORA = pd.DataFrame(
    {
        "library": "functional_effect",
        "set_name": ["activates_activity", "inhibits_activity"],
        "fdr": [0.001, 0.2],
        "direction": ["enriched", "depleted"],
    }
)
GENE_ORA_SPLIT = pd.DataFrame(
    {
        "library": "KEGG",
        "term": ["MAPK signaling pathway", "MAPK signaling pathway", "Cell cycle"],
        "fdr": [0.001, 0.01, 0.03],
        "direction": ["up", "down", "up"],
    }
)


def _builders():
    adata = _make_adata()
    de = _make_de()
    sweep = pd.DataFrame({"threshold": [0.5, 0.75, 0.9], "n_sites": [900, 700, 300]})
    return {
        "volcano": lambda: ap.viz.volcano(de, label=de.index[:2]),
        "embedding": lambda: ap.viz.embedding(adata),
        "intensity_distribution": lambda: ap.viz.intensity_distribution(adata),
        "site_abundance": lambda: ap.viz.site_abundance(adata, adata.var_names[0]),
        "site_abundance_violin": lambda: ap.viz.site_abundance(
            adata, adata.var_names[0], kind="violin"
        ),
        "heatmap": lambda: ap.viz.heatmap(adata),
        "kinase_activity_bar": lambda: ap.viz.kinase_activity_bar(KSEA),
        "enrichment_dotplot": lambda: ap.viz.enrichment_dotplot(GENE_ORA_SPLIT),
        "retention_curve": lambda: ap.viz.retention_curve(sweep, chosen=0.75),
    }


def _is_outlined(trace) -> bool:
    """Black outline on the mark, or a fully transparent carrier trace whose points are."""
    kind = trace.type
    if kind == "heatmap":
        return True
    if kind in ("box", "violin"):
        if trace.line.color == "black":
            return True
        return trace.line.color == "rgba(0,0,0,0)" and trace.marker.line.color == "black"
    return trace.marker.line.color == "black"


# ---------------------------------------------------------------------------
# House style, for every builder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_builders()))
def test_builder_follows_house_style(name):
    fig = _builders()[name]()
    layout = fig.layout
    assert layout.height == PLOT_HEIGHT
    assert layout.width <= PLOT_WIDTH_MAX and layout.width % 300 == 0
    assert layout.template.layout.paper_bgcolor == "white"  # plotly_white
    assert layout.showlegend is False
    assert not layout.xaxis.showgrid and not layout.yaxis.showgrid
    assert all(_is_outlined(t) for t in fig.data), [t.type for t in fig.data]


# ---------------------------------------------------------------------------
# Volcano
# ---------------------------------------------------------------------------


def test_volcano_categories_and_fdr_line():
    de = _make_de()
    cat = ap.viz.volcano_categories(de, fdr_threshold=0.05, log2fc_threshold=1.0)
    sig = de["fdr"] < 0.05
    assert (cat == "sig_up").sum() == (sig & (de["log2fc"] >= 1)).sum()
    assert (cat == "ns").sum() == (~sig).sum()

    fig = ap.viz.volcano(de)
    colours = {t.name: t.marker.color for t in fig.data}
    assert colours == VOLCANO
    # y is always FDR, so the significance line sits exactly at the threshold.
    assert fig.layout.yaxis.title.text == "-log10 FDR"
    y = np.concatenate([t.y for t in fig.data])
    assert np.sort(y) == pytest.approx(np.sort(-np.log10(de["fdr"])))
    hline = [s for s in fig.layout.shapes if s.y0 == s.y1][0]
    assert hline.y0 == pytest.approx(-np.log10(0.05))


def test_volcano_rejects_observed_only_tuple():
    de = _make_de()
    with pytest.raises(TypeError, match="first element"):
        ap.viz.volcano((de, pd.DataFrame()))


def test_tables_with_array_attrs_still_plot():
    # Any provenance array in attrs makes pandas' nsmallest / nlargest raise
    # (attrs are compared inside concat); plotting must not depend on attrs.
    de = _make_de()
    de.loc[de.index[:3], "fdr"] = np.nan
    de.attrs["prior_variance"] = np.ones(len(de))
    ap.viz.volcano(de)
    ora = GENE_ORA_SPLIT.copy()
    ora.loc[0, "fdr"] = np.nan
    ora.attrs["prior_variance"] = np.ones(3)
    ap.viz.enrichment_dotplot(ora)


# ---------------------------------------------------------------------------
# Palettes and points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        (("a", "b", "c"), EMBEDDING_PALETTE[3]),
        (tuple("abcdef"), CONDITION_PALETTE[6]),  # past the embedding ceiling of 5
        (("P1", "P24", "R1"), [TIMEPOINT_PALETTE[g] for g in ("P1", "P24", "R1")]),
    ],
)
def test_embedding_palette_choice(groups, expected):
    fig = ap.viz.embedding(_make_adata(groups=groups))
    assert [t.marker.color for t in fig.data] == expected
    assert fig.layout.xaxis.title.text == "PC1 (40.0%)"


@pytest.mark.parametrize(("n_per_group", "points"), [(4, True), (120, False)])
def test_site_abundance_points_below_100(n_per_group, points):
    adata = _make_adata(n_per_group=n_per_group, n_sites=5)
    fig = ap.viz.site_abundance(adata, adata.var_names[0])
    has_points = any(t.type == "box" and t.boxpoints == "all" for t in fig.data)
    assert has_points is points


def test_long_sample_names_are_shortened():
    fig = ap.viz.intensity_distribution(_make_adata())
    assert list(fig.layout.xaxis.ticktext[:2]) == ["ctrl_R0", "ctrl_R1"]


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


def test_heatmap_bounds_and_site_cap():
    adata = _make_adata(n_sites=60)
    z = ap.viz.heatmap(adata, max_sites=25).data[0]
    assert z.zmin == -z.zmax
    assert np.asarray(z.z).shape == (25, adata.n_obs)

    raw = ap.viz.heatmap(adata, zscore=False).data[0]
    finite = adata.X[np.isfinite(adata.X)]
    assert raw.zmax == pytest.approx(np.percentile(finite, 99))

    with pytest.raises(KeyError, match="not in adata.var_names"):
        ap.viz.heatmap(adata, sites=["nope"])


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def test_enrichment_dotplot_schemas():
    site = ap.viz.enrichment_dotplot(SITE_ORA)
    assert list(site.data[0].y) == ["activates_activity", "inhibits_activity"]

    # Split-direction ORA repeats a term; the direction keeps the rows apart.
    gene = ap.viz.enrichment_dotplot(GENE_ORA_SPLIT)
    assert len(set(gene.data[0].y)) == 3

    with pytest.raises(KeyError, match="set_name"):
        ap.viz.enrichment_dotplot(KSEA)


def test_kinase_activity_bar_filters_by_fdr():
    fig = ap.viz.kinase_activity_bar(KSEA, fdr_threshold=0.05)
    assert list(fig.data[0].y) == ["SRC", "MAPK1", "AKT1"]  # sorted by score
    empty = ap.viz.kinase_activity_bar(KSEA, fdr_threshold=1e-9)
    assert not empty.data and "no kinases" in empty.layout.annotations[0].text


# ---------------------------------------------------------------------------
# Export and optional dependency
# ---------------------------------------------------------------------------


def test_save_figure_by_suffix(tmp_path):
    fig = ap.viz.retention_curve(pd.DataFrame({"threshold": [0.5, 0.9], "n_sites": [10, 5]}))
    assert ap.viz.save_figure(fig, tmp_path / "f.html").stat().st_size > 0
    with pytest.raises(ValueError, match="unsupported"):
        ap.viz.save_figure(fig, tmp_path / "f.jpeg")


def test_viz_imports_without_plotly():
    code = (
        "import sys; sys.modules['plotly'] = None\n"
        "import alphaphos as ap, pandas as pd\n"
        "try:\n"
        "    ap.viz.volcano(pd.DataFrame({'log2fc': [], 'p_value': [], 'fdr': []}))\n"
        "except ImportError as e:\n"
        "    assert 'alphaphos[viz]' in str(e); print('ok')\n"
    )
    src = Path(ap.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(src)}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert out.stdout.strip() == "ok"
