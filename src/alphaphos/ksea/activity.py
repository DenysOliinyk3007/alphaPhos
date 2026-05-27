"""Kinase activity inference via decoupler-py (Saez-Rodriguez framework).

Given a differential-analysis result (sites × stats) and a kinase-substrate
network (from :mod:`alphaphos.ksea.network` or BYO), runs decoupler's
statistical methods to score each kinase's activity. Returns per-kinase
DataFrames with score + p-value + n_targets columns.

Four methods exposed, each wrapping a `dc.mt.*` Method:

- :func:`kinase_activity_ulm` — Univariate Linear Model (the classical
  KSEA z-score equivalent; recommended default).
- :func:`kinase_activity_mlm` — Multivariate Linear Model (controls for
  cross-kinase substrate overlap).
- :func:`kinase_activity_ora` — Over-Representation Analysis (Fisher's
  exact on top-N tail).
- :func:`kinase_activity_gsea` — Gene Set Enrichment Analysis (full-rank,
  weighted K-S like phospho-MEA but on curated substrate sets).

Pipeline position
-----------------
These run downstream of differential analysis. Inputs are a `pd.DataFrame`
of stats per site (typically the limma output from `apt.tl.diff_exp_ebayes`)
and the KS network. Outputs are pandas tables you can sort, filter,
display, and plot.
"""

from __future__ import annotations

import logging

import pandas as pd

from alphaphos.ksea.network import alphaphos_site_to_omnipath

logger = logging.getLogger(__name__)

# decoupler's *.mt methods accept a feature-stat matrix shaped
# (n_observations, n_features). For KSEA from a single contrast we
# build a 1-row "matrix" (one observation = our experiment) with one
# value per site (e.g. the logFC).
_OBS_LABEL = "contrast"


def _prepare_data_and_net(
    diff_results: pd.DataFrame,
    network: pd.DataFrame,
    *,
    id_col: str,
    stat_col: str,
    convert_site_ids: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Common preprocessing for all KSEA methods.

    - Optionally converts alphaPhos site ids to OmniPath ``UniProtAC_AAposition``
      format so they match ``network.target``.
    - Drops sites with missing stat or unconvertible id.
    - Builds the (1 × n_features) data frame decoupler expects.
    - Filters the network to substrates that exist in the data (decoupler
      does this internally but pre-filtering keeps logging useful).

    Returns
    -------
    data, net
    """
    df = diff_results[[id_col, stat_col]].copy().dropna(subset=[stat_col])
    if convert_site_ids:
        df[id_col] = df[id_col].astype(str).map(alphaphos_site_to_omnipath)
        df = df.dropna(subset=[id_col])
    df = df.drop_duplicates(subset=[id_col])
    data = pd.DataFrame(
        [df.set_index(id_col)[stat_col].values], index=[_OBS_LABEL], columns=df[id_col].values
    ).astype(float)
    overlap = sorted(set(network["target"]) & set(data.columns))
    if not overlap:
        raise ValueError(
            "No overlap between diff_results site ids and network targets. "
            "Did you forget convert_site_ids=True (alphaPhos -> OmniPath format)?"
        )
    logger.info(
        "Network/data overlap: %d sites (network has %d targets; data has %d)",
        len(overlap),
        data.shape[1],
        network["target"].nunique(),
    )
    return data, network


def _run_dc_method(
    method,
    data: pd.DataFrame,
    net: pd.DataFrame,
    *,
    min_targets: int,
    **kwargs,
) -> pd.DataFrame:
    """Call a `dc.mt.*` Method once; reshape (1 × n_kinases) output to
    a long per-kinase DataFrame with score, fdr, n_targets columns.

    Note: decoupler 2.x applies Benjamini-Hochberg FDR correction
    internally before returning (see ``decoupler/mt/_run.py`` —
    ``_fdr_bh_axis1_numba`` is called on the p-value matrix). Our
    ``fdr`` column therefore reports **BH-adjusted** p-values across
    all kinases for this contrast.
    """
    score_df, padj_df = method(data=data, net=net, tmin=min_targets, verbose=False, **kwargs)
    out = pd.DataFrame(
        {
            "score": score_df.iloc[0],
            "fdr": padj_df.iloc[0],
        }
    )
    overlap = set(data.columns)
    n_targets = (
        net.loc[net["target"].isin(overlap)]
        .groupby("source")
        .size()
        .reindex(out.index, fill_value=0)
    )
    out["n_targets"] = n_targets.astype(int)
    out.index.name = "kinase"
    return out


def kinase_activity_ulm(
    diff_results: pd.DataFrame,
    network: pd.DataFrame,
    *,
    id_col: str = "protein",
    stat_col: str = "log2fc",
    min_targets: int = 5,
    convert_site_ids: bool = True,
) -> pd.DataFrame:
    """KSEA-equivalent kinase activity via Univariate Linear Model.

    The recommended default for classical KSEA-style results. For each
    kinase, fits a linear model regressing site stats against kinase
    substrate membership. The slope's t-value is the activity score —
    positive = activated, negative = inhibited, p-value = significance.

    Parameters
    ----------
    diff_results
        Differential-analysis output. Must contain ``id_col`` and
        ``stat_col``.
    network
        Kinase-substrate network DataFrame from
        :func:`alphaphos.ksea.fetch_omnipath_ks_network` (or BYO with
        columns ``source``, ``target``, ``weight``).
    id_col, stat_col
        Column names in ``diff_results``. ``stat_col`` is typically
        ``"log2fc"`` for logFC-based KSEA or the limma t-stat for
        moderated-stat-based.
    min_targets
        Minimum number of substrates per kinase (decoupler's ``tmin``).
        Default 5. Kinases with fewer observed substrates are skipped.
    convert_site_ids
        If True (default), convert alphaPhos site ids
        (``P00533~EGFR_Y1172_M1``) to OmniPath format
        (``P00533_Y1172``) before matching against ``network.target``.

    Returns
    -------
    pd.DataFrame indexed by ``kinase`` with columns:

    - ``score`` — ULM t-statistic. Positive = activated, negative = inhibited.
    - ``fdr`` — Benjamini-Hochberg-adjusted p-value (across kinases).
    - ``n_targets`` — number of substrates observed in ``diff_results``.
    """
    import decoupler as dc

    data, net = _prepare_data_and_net(
        diff_results,
        network,
        id_col=id_col,
        stat_col=stat_col,
        convert_site_ids=convert_site_ids,
    )
    return _run_dc_method(dc.mt.ulm, data, net, min_targets=min_targets)


def kinase_activity_mlm(
    diff_results: pd.DataFrame,
    network: pd.DataFrame,
    *,
    id_col: str = "protein",
    stat_col: str = "log2fc",
    min_targets: int = 5,
    convert_site_ids: bool = True,
) -> pd.DataFrame:
    """Kinase activity via Multivariate Linear Model.

    Like :func:`kinase_activity_ulm` but fits all kinases jointly,
    controlling for cross-kinase substrate overlap. Useful when many
    kinases share substrates; the per-kinase coefficients are partial
    effects rather than marginal.

    Same return format as :func:`kinase_activity_ulm`.
    """
    import decoupler as dc

    data, net = _prepare_data_and_net(
        diff_results,
        network,
        id_col=id_col,
        stat_col=stat_col,
        convert_site_ids=convert_site_ids,
    )
    return _run_dc_method(dc.mt.mlm, data, net, min_targets=min_targets)


def kinase_activity_ora(
    diff_results: pd.DataFrame,
    network: pd.DataFrame,
    *,
    id_col: str = "protein",
    stat_col: str = "log2fc",
    top_n: int | float = 0.05,
    min_targets: int = 5,
    convert_site_ids: bool = True,
) -> pd.DataFrame:
    """Kinase activity via Over-Representation Analysis (Fisher's exact).

    Picks the top-N sites by |stat| and tests whether each kinase's
    substrate set is over-represented in that top-N versus the rest.
    Simpler / faster than ULM, but only sensitive to the tail.

    Parameters
    ----------
    top_n
        Either an int (top N sites by absolute stat) or a float in
        ``(0, 1)`` interpreted as the top fraction. Default 0.05 = top 5%.
    """
    import decoupler as dc

    data, net = _prepare_data_and_net(
        diff_results,
        network,
        id_col=id_col,
        stat_col=stat_col,
        convert_site_ids=convert_site_ids,
    )
    # decoupler's ORA wants the data matrix to be a binary {0, 1} indicator
    # of "site is in the significant set". Binarise by |stat| ranking.
    vals = data.iloc[0].abs().sort_values(ascending=False)
    if isinstance(top_n, float) and 0 < top_n < 1:
        k = max(1, round(len(vals) * top_n))
    else:
        k = int(top_n)
    top_set = set(vals.iloc[:k].index)
    binary = data.copy()
    binary.loc[_OBS_LABEL] = [1.0 if c in top_set else 0.0 for c in binary.columns]
    return _run_dc_method(dc.mt.ora, binary, net, min_targets=min_targets)


def kinase_activity_gsea(
    diff_results: pd.DataFrame,
    network: pd.DataFrame,
    *,
    id_col: str = "protein",
    stat_col: str = "log2fc",
    min_targets: int = 5,
    permutation_num: int = 1000,
    seed: int = 42,
    convert_site_ids: bool = True,
) -> pd.DataFrame:
    """Kinase activity via GSEA (weighted K-S on the full rank).

    Uses every site's rank, no hard threshold. Most rigorous of the four
    when the diff_exp result is moderately small (~10k sites) — handles
    overlap statistically rather than via threshold.

    Returns a DataFrame indexed by kinase with ``score`` (NES), ``fdr``,
    and ``n_targets``.
    """
    import decoupler as dc

    data, net = _prepare_data_and_net(
        diff_results,
        network,
        id_col=id_col,
        stat_col=stat_col,
        convert_site_ids=convert_site_ids,
    )
    return _run_dc_method(
        dc.mt.gsea,
        data,
        net,
        min_targets=min_targets,
        times=permutation_num,
        seed=seed,
    )
