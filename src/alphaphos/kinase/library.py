"""Per-site kinase prediction using the Yaffe Kinase Library PWMs.

Wraps the ``kinase_library`` package (Johnson et al., *Nature* 2023 for the
Ser/Thr kinome; Yaromenko et al. 2024 for the tyrosine kinome -- 311 Ser/Thr
+ 78 tyrosine PWMs in kinase-library 1.8) and exposes them in alphaPhos's
AnnData-centric idiom.  For each site, returns the top-k predicted upstream
kinases.

Two metrics
-----------
* ``score`` -- the raw log2 PWM score.  Comparable *across sites for one
  kinase*, but NOT across kinases for one site: promiscuous kinases have high
  baseline scores everywhere (the MEK site MAPK1 T185 ranks PRP4/BMPR1A first
  by score).
* ``percentile`` (default for :func:`predict_kinases`) -- the score's
  percentile against the kinase's score distribution on the reference
  phosphoproteome (Johnson 2023's recommended per-site ranking metric;
  MAPK1 T185 -> MEK2 100 / MEK1 99, RPS6 S235 -> AKT1 100).

Installation note
-----------------
kinase-library 1.8 pins ``numpy~=1.26``, ``pandas~=2.2`` and
``matplotlib~=3.8.3``, which cannot be satisfied next to alphaPhos on Python
3.13 / pandas 3.  The package itself works with current versions::

    pip install --no-deps kinase-library
    pip install biopython adjustText natsort seaborn statsmodels tqdm openpyxl matplotlib

Pipeline position
-----------------
This is **per-site PWM prediction** — sequence-based, works for novel
sites that aren't in any database. Complements (does not replace) the
network-based KSEA approach (``alphaphos.enrichment.kinase_activity``)
which scores kinase *activities* across a differential analysis result.

Sequence format
---------------
The Yaffe library expects 15-mers (±7 around a central S/T/Y), with the
center residue being s/t/y. alphaPhos's ``kinase_sequence`` column (set by
:func:`alphaphos.add_kinase_windows` -- a separate step after
:func:`alphaphos.collapse_sites`) uses the format
``"_<left>*<X>*<right>_"`` with explicit ``*`` markers around the phospho
residue. The conversion is trivial — just strip the ``*`` characters.
Error sentinels (FASTA_ERROR:, POSITION_ERROR:, etc.) are silently dropped.

Two functions exposed
---------------------
- :func:`score_kinases` — compute full (sites × kinases) matrices for the
  requested metrics, written to ``adata.varm['kinase_score_<pool>']`` and
  ``adata.varm['kinase_percentile_<pool>']`` (pool = ``ser_thr`` / ``tyrosine``),
  with provenance in ``adata.uns["alphaphos"]["kinase_scores"]``.
- :func:`predict_kinases` — pick the top-k kinases per site by one metric;
  returns a compact DataFrame indexed by site for downstream tables / plots.
"""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

# Sentinel strings emitted by alphaphos.kinase.annotation.extract_window
# when FASTA lookup or position validation fails. Skip these.
_ERROR_PREFIXES = (
    "FASTA_ERROR:",
    "POSITION_ERROR:",
    "SEQUENCE_MISMATCH:",
    "PARSING_ERROR:",
)

# Metadata columns that PhosphoProteomics.score() / .percentile() prepend to
# their output — everything that isn't one of these is a kinase column.
_KL_META_COLS = frozenset({"id", "seq", "phos_res", "Sequence"})

Metric = Literal["score", "percentile"]
_METRICS: tuple[str, ...] = ("score", "percentile")
UNS_KINASE_SCORES = "kinase_scores"


def require_kinase_library(caller: str):
    """Import ``kinase_library`` or raise with the install recipe."""
    try:
        import kinase_library
    except ImportError as exc:
        raise ImportError(
            f"{caller} requires the kinase-library package (Johnson et al. 2023). Its "
            "dependency pins are stale, so install it without them: "
            "pip install --no-deps kinase-library && pip install biopython adjustText "
            "natsort seaborn statsmodels tqdm openpyxl matplotlib"
        ) from exc
    return kinase_library


@contextlib.contextmanager
def kl_context(quiet: bool) -> Iterator[None]:
    """Run a kinase_library call.

    ``quiet=True`` redirects its unconditional chatter -- "Calculating
    percentiles ..." on stdout and tqdm progress bars on stderr -- into a
    buffer.  This also sidesteps the Windows cp1252 crash on tqdm's block
    characters without touching the process-wide stream configuration (which
    an earlier version did at import).  Exceptions propagate unchanged.
    """
    if not quiet:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _strip_alphaphos_markers(kinase_sequence: str | None) -> str | None:
    """Convert an alphaPhos ``kinase_sequence`` to the Yaffe Kinase Library format.

    alphaPhos format:  ``"_AAVKRGT*S*ELLIQAA_"`` (``*`` markers around phospho residue,
    outer/inner ``_`` for protein-terminal padding).
    Yaffe format:      ``"_AAVKRGTSELLIQAA_"``  (same except no ``*``; the library
    auto-detects the central S/T/Y).

    Returns ``None`` for missing / error-sentinel inputs.
    """
    if not isinstance(kinase_sequence, str) or not kinase_sequence:
        return None
    if kinase_sequence.startswith(_ERROR_PREFIXES):
        return None
    return kinase_sequence.replace("*", "")


def _kinase_columns(scores_df: pd.DataFrame) -> list[str]:
    """Return the kinase column names from a PhosphoProteomics.score() output."""
    return [c for c in scores_df.columns if c not in _KL_META_COLS]


def _to_varm_dataframe(
    scores_df: pd.DataFrame,
    site_index: pd.Index,
) -> pd.DataFrame:
    """Realign a (n_valid_sites x metadata+n_kinases) score DataFrame onto the full
    ``adata.var.index``, with NaN for sites the library dropped.
    """
    kinase_cols = _kinase_columns(scores_df)
    out = pd.DataFrame(np.nan, index=site_index, columns=kinase_cols, dtype=float)
    out.loc[scores_df["id"].values, kinase_cols] = scores_df[kinase_cols].values
    return out


def score_kinases(
    adata: ad.AnnData,
    *,
    sequence_col: str = "kinase_sequence",
    metrics: tuple[Metric, ...] = ("score", "percentile"),
    varm_prefix: str = "kinase",
    round_digits: int = 3,
    quiet: bool = True,
) -> None:
    """Score every site in ``adata`` against every kinase PWM.

    Writes one (sites × kinases) DataFrame per metric and kinase pool to
    ``adata.varm``:

    - ``adata.varm[f"{varm_prefix}_score_ser_thr"]`` / ``..._tyrosine`` --
      raw log2 PWM scores
    - ``adata.varm[f"{varm_prefix}_percentile_ser_thr"]`` / ``..._tyrosine`` --
      percentile of the score against the reference phosphoproteome

    (a pool is only written when it has scoreable sites) and provenance to
    ``adata.uns["alphaphos"]["kinase_scores"]``.

    Parameters
    ----------
    adata
        AnnData whose ``var`` contains a ``kinase_sequence`` column
        (populated by :func:`alphaphos.add_kinase_windows` after
        :func:`alphaphos.collapse_sites`).
    sequence_col
        ``var`` column holding the alphaPhos-format kinase sequences.
    metrics
        Which matrices to compute; default both.  See the module docstring
        for why ``percentile`` is the per-site ranking metric.
    varm_prefix
        Prefix for the varm keys (``<prefix>_<metric>_<pool>``).
    round_digits
        Decimal digits for the ``score`` matrix (kinase-library default 3);
        percentiles use 2.
    quiet
        Silence kinase_library's stdout progress output (default True).

    Notes
    -----
    Sites with missing / error / even-length sequences are dropped and appear
    as all-NaN rows in the varm matrices. The drop count is reported in
    ``adata.uns["alphaphos"]["kinase_scores"]["n_sites_dropped"]``.
    """
    kinase_library = require_kinase_library("score_kinases")
    PhosphoProteomics = kinase_library.PhosphoProteomics
    bad = [m for m in metrics if m not in _METRICS]
    if bad or not metrics:
        raise ValueError(f"metrics must be a non-empty subset of {_METRICS}; got {metrics!r}")

    if sequence_col not in adata.var.columns:
        raise ValueError(
            f"adata.var must contain {sequence_col!r}. Call "
            f"alphaphos.add_kinase_windows(adata, fasta_path='proteome.fasta') "
            f"first to populate it (or pass an explicit sequence_col)."
        )

    # Convert alphaPhos -> Yaffe format and filter to valid odd-length sequences
    seqs = adata.var[sequence_col].map(_strip_alphaphos_markers)
    keep = seqs.notna() & seqs.str.len().mod(2).eq(1)
    n_valid = int(keep.sum())
    n_dropped = len(seqs) - n_valid

    if n_valid == 0:
        raise ValueError(
            f"No valid sequences for kinase scoring "
            f"(got {n_dropped} invalid / missing / even-length). "
            f"Check that {sequence_col!r} is populated and well-formed."
        )

    df_in = pd.DataFrame(
        {
            "id": adata.var.index[keep].astype(str),
            "seq": seqs[keep].values,
        }
    )

    with kl_context(quiet):
        pp = PhosphoProteomics(df_in, seq_col="seq", suppress_warnings=True)

    site_index = adata.var.index
    kinases: dict[str, list[str]] = {"ser_thr": [], "tyrosine": []}
    n_pool = {"ser_thr": len(pp.ser_thr_data), "tyrosine": len(pp.tyrosine_data)}
    n_scored = n_pool["ser_thr"] + n_pool["tyrosine"]
    # n_valid - n_scored = sequences our pre-filter accepted but the library
    # rejected (e.g. odd-length but central residue isn't S/T/Y).
    n_dropped_total = n_dropped + (n_valid - n_scored)

    for pool, n in n_pool.items():
        if n == 0:
            continue
        for metric in metrics:
            with kl_context(quiet):
                if metric == "score":
                    mat = pp.score(kin_type=pool, round_digits=round_digits)
                else:
                    mat = pp.percentile(kin_type=pool)
            adata.varm[f"{varm_prefix}_{metric}_{pool}"] = _to_varm_dataframe(mat, site_index)
            kinases[pool] = _kinase_columns(mat)

    adata.uns.setdefault("alphaphos", {})[UNS_KINASE_SCORES] = {
        "library": "kinase-library (Johnson et al. Nature 2023; Yaromenko et al. 2024)",
        "library_version": _kl_version(kinase_library),
        "metrics": list(metrics),
        "varm_prefix": varm_prefix,
        "n_sites_scored": n_scored,
        "n_sites_dropped": n_dropped_total,
        "n_ser_thr_scored": n_pool["ser_thr"],
        "n_tyrosine_scored": n_pool["tyrosine"],
        "ser_thr_kinases": kinases["ser_thr"],
        "tyrosine_kinases": kinases["tyrosine"],
        "round_digits": round_digits,
    }
    logger.info(
        "score_kinases: scored %d sites (%d S/T + %d Y) for %s; %d sites dropped",
        n_scored,
        n_pool["ser_thr"],
        n_pool["tyrosine"],
        list(metrics),
        n_dropped_total,
    )


def _kl_version(kinase_library) -> str | None:
    try:
        from importlib.metadata import version

        return version("kinase-library")
    except Exception:  # pragma: no cover
        return getattr(kinase_library, "__version__", None)


def predict_kinases(
    adata: ad.AnnData,
    *,
    sequence_col: str = "kinase_sequence",
    top_k: int = 5,
    metric: Metric = "percentile",
    varm_prefix: str = "kinase",
    overwrite: bool = False,
    quiet: bool = True,
) -> pd.DataFrame:
    """Return the top-k predicted upstream kinases per site.

    Runs :func:`score_kinases` first if the matrices for ``metric`` are not
    already in ``adata.varm`` (or if ``overwrite=True``).

    Parameters
    ----------
    adata
        AnnData with a ``kinase_sequence`` column in ``.var``.
    sequence_col
        ``var`` column with the alphaPhos-format kinase sequences.
    top_k
        Number of top kinases to report per site (default 5).
    metric
        ``"percentile"`` (default; Johnson 2023's per-site ranking metric,
        corrects for kinase promiscuity) or ``"score"`` (raw log2 PWM score,
        the pre-0.24 behaviour).
    varm_prefix
        Prefix of the varm keys written by ``score_kinases``.
    overwrite
        If True, re-run scoring even if the varm matrices already exist.
    quiet
        Passed to :func:`score_kinases`.

    Returns
    -------
    pd.DataFrame indexed by ``adata.var.index`` with columns:

    - ``top1_kinase``, ``top1_score``, ..., ``top{k}_kinase``, ``top{k}_score``
      (the ``*_score`` columns hold the chosen ``metric``)
    - ``top_kinases``: comma-joined list of the top-k kinase names
    - ``top_scores``:  comma-joined list of the corresponding values
    - ``kin_type``:    ``"ser_thr"``, ``"tyrosine"``, or ``"none"`` (site dropped)

    ``result.attrs["metric"]`` records which metric was ranked.

    Notes
    -----
    Sites with invalid sequences appear as rows with ``kin_type='none'`` and
    NaN scores. This makes the result safe to ``adata.var.join(...)`` directly.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}; got {metric!r}")
    st_key = f"{varm_prefix}_{metric}_ser_thr"
    y_key = f"{varm_prefix}_{metric}_tyrosine"

    need_score = overwrite or (st_key not in adata.varm and y_key not in adata.varm)
    if need_score:
        score_kinases(adata, sequence_col=sequence_col, varm_prefix=varm_prefix, quiet=quiet)

    out_cols: list[str] = []
    for i in range(1, top_k + 1):
        out_cols.extend([f"top{i}_kinase", f"top{i}_score"])
    out_cols.extend(["top_kinases", "top_scores", "kin_type"])

    out = pd.DataFrame(
        {c: pd.Series(pd.NA, index=adata.var.index, dtype="object") for c in out_cols},
    )
    prov = adata.uns.get("alphaphos", {}).get(UNS_KINASE_SCORES, {})

    def _process_varm(varm_key: str, kin_type: str) -> None:
        if varm_key not in adata.varm:
            return
        mat = adata.varm[varm_key]
        if not isinstance(mat, pd.DataFrame):
            # AnnData converted it to ndarray; bring back column names
            mat = pd.DataFrame(mat, index=adata.var.index, columns=prov[f"{kin_type}_kinases"])
        # Sites with any non-NaN row got scored
        any_score = mat.notna().any(axis=1)
        if not any_score.any():
            return
        scored = mat.loc[any_score]
        kinases = np.asarray(scored.columns)

        # Vectorised top-k: argpartition then sort each row's top-k
        k = min(top_k, scored.shape[1])
        arr = scored.to_numpy(dtype=float)
        top_idx = np.argpartition(-arr, k - 1, axis=1)[:, :k]
        row_idx = np.arange(arr.shape[0])[:, None]
        top_scores = arr[row_idx, top_idx]
        sort_order = np.argsort(-top_scores, axis=1)
        top_idx = top_idx[row_idx, sort_order]
        top_scores = arr[row_idx, top_idx]
        top_names = kinases[top_idx]  # (n_sites, k)

        for i in range(k):
            out.loc[scored.index, f"top{i + 1}_kinase"] = top_names[:, i]
            out.loc[scored.index, f"top{i + 1}_score"] = top_scores[:, i]
        out.loc[scored.index, "top_kinases"] = [",".join(row) for row in top_names]
        out.loc[scored.index, "top_scores"] = [
            ",".join(f"{s:.3f}" for s in row) for row in top_scores
        ]
        out.loc[scored.index, "kin_type"] = kin_type

    _process_varm(st_key, "ser_thr")
    _process_varm(y_key, "tyrosine")

    # Sites neither in ser_thr nor tyrosine got dropped
    out["kin_type"] = out["kin_type"].fillna("none")
    out.attrs["metric"] = metric
    return out
