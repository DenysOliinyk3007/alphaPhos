"""Publication-style Plotly figures for alphaPhos results.

Every builder returns a ``plotly.graph_objects.Figure`` that follows one
visual system: ``plotly_white``, a black outline on every mark,
colourblind-audited palettes keyed by group count, individual points for
groups under 100, height 600 with width growing in 300 px steps to 1800,
annotations instead of legends.  Constants and layout helpers live in
:mod:`alphaphos.viz.style`.

Typical use::

    import alphaphos as ap

    ap.viz.volcano(result)                          # diff_exp_limma table
    ap.viz.site_abundance(adata, site_key)           # replicates behind a hit
    ap.viz.embedding(adata, basis="pca")             # after ap.dimred.pca
    ap.viz.kinase_activity_bar(ksea)
    ap.viz.contaminant_fraction(ap.qc.compute_contaminant_fraction_per_sample(raw_psm))
    ap.viz.save_figure(fig, "figure_2a.pdf")         # one format, from the suffix

Requires ``plotly`` (and ``kaleido`` for static export): install with
``pip install 'alphaphos[viz]'``.  The module imports without them; the
builders raise on first call.
"""

from alphaphos.viz import style
from alphaphos.viz._base import save_figure
from alphaphos.viz.differential import volcano, volcano_categories
from alphaphos.viz.enrichment import enrichment_dotplot, kinase_activity_bar
from alphaphos.viz.filtering import retention_curve
from alphaphos.viz.qc import contaminant_fraction
from alphaphos.viz.samples import embedding, heatmap, intensity_distribution, site_abundance

__all__ = [
    "style",
    "save_figure",
    "volcano",
    "volcano_categories",
    "embedding",
    "intensity_distribution",
    "site_abundance",
    "heatmap",
    "kinase_activity_bar",
    "enrichment_dotplot",
    "retention_curve",
    "contaminant_fraction",
]
