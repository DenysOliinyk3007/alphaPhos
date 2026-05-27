"""Per-site kinase prediction using the Yaffe Kinase Library PWMs.

Wraps the ``kinase_library`` package (Johnson et al., *Nature* 2023) — 303
Ser/Thr kinase PWMs + 93 tyrosine kinase PWMs — and exposes them in
alphaPhos's AnnData-centric idiom. For each site, returns the top-k
predicted upstream kinases by PWM score.

Pipeline position
-----------------
This is **per-site PWM prediction** — sequence-based, works for novel
sites that aren't in any database. Complements (does not replace) the
network-based KSEA approach (planned, ``alphaphos.ksea``) which scores
kinase *activities* across a differential analysis result.

Sequence format
---------------
The Yaffe library expects 15-mers (±7 around a central S/T/Y), with the
center residue being s/t/y. alphaPhos's ``kinase_sequence`` column (set by
``collapse_sites(fasta_path=...)``) uses the format
``"_<left>*<X>*<right>_"`` with explicit ``*`` markers around the phospho
residue. The conversion is trivial — just strip the ``*`` characters.
Error sentinels (FASTA_ERROR:, POSITION_ERROR:, etc.) are silently dropped.

Two functions exposed
---------------------
- :func:`score_kinases` — compute full (sites × kinases) score matrices,
  written to ``adata.varm['kinase_score_ser_thr']`` and
  ``adata.varm['kinase_score_tyrosine']``.
- :func:`predict_kinases` — pick the top-k kinases per site; returns a
  compact DataFrame indexed by site for use in downstream tables / plots.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

# PeptideCollapse sentinels when FASTA-lookup fails (see
# alphaphos.preprocess.collapse._create_kinase_sequence). Skip these.
_ERROR_PREFIXES = (
    "FASTA_ERROR:",
    "POSITION_ERROR:",
    "SEQUENCE_MISMATCH:",
    "PARSING_ERROR:",
)

# Metadata columns that PhosphoProteomics.score() prepends to its output —
# everything that isn't one of these is a kinase score column.
_KL_META_COLS = frozenset({"id", "seq", "phos_res", "Sequence"})


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
    varm_prefix: str = "kinase_score",
    round_digits: int = 3,
) -> None:
    """Score every site in ``adata`` against every kinase PWM.

    Writes (sites × kinases) DataFrames to ``adata.varm``:

    - ``adata.varm[f"{varm_prefix}_ser_thr"]`` — 303 Ser/Thr kinases (if any
       S- or T-centered sites are present)
    - ``adata.varm[f"{varm_prefix}_tyrosine"]`` — 93 tyrosine kinases (if any
       Y-centered sites)

    And provenance metadata to ``adata.uns["alphaphos_kinase"]``.

    Parameters
    ----------
    adata
        AnnData whose ``var`` contains a ``kinase_sequence`` column
        (populated by ``collapse_sites(fasta_path=...)``).
    sequence_col
        ``var`` column holding the alphaPhos-format kinase sequences.
    varm_prefix
        Prefix for the two output varm keys (``..._ser_thr``, ``..._tyrosine``).
    round_digits
        Decimal digits to round the PWM scores to. Default 3 (matches the
        kinase-library default).

    Notes
    -----
    Sites with missing / error / even-length sequences are silently dropped
    and appear as all-NaN rows in the varm matrices. The drop count is
    reported in ``adata.uns["alphaphos_kinase"]["n_sites_dropped"]``.
    """
    try:
        from kinase_library import PhosphoProteomics
    except ImportError as exc:
        raise ImportError(
            "kinase-library is required for score_kinases(). "
            "Install with: pip install kinase-library"
        ) from exc

    if sequence_col not in adata.var.columns:
        raise ValueError(
            f"adata.var must contain {sequence_col!r}. Re-run "
            f"collapse_sites(fasta_path='proteome.fasta', ...) to populate it "
            f"(or pass an explicit sequence_col)."
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

    pp = PhosphoProteomics(df_in, seq_col="seq", suppress_warnings=True)

    site_index = adata.var.index
    ser_thr_kinases: list[str] = []
    tyrosine_kinases: list[str] = []

    n_st = len(pp.ser_thr_data)
    n_ty = len(pp.tyrosine_data)
    n_scored = n_st + n_ty
    # n_valid - n_scored = sequences our pre-filter accepted but the library
    # rejected (e.g. odd-length but central residue isn't S/T/Y).
    n_dropped_total = n_dropped + (n_valid - n_scored)

    if n_st > 0:
        st = pp.score(kin_type="ser_thr", round_digits=round_digits)
        adata.varm[f"{varm_prefix}_ser_thr"] = _to_varm_dataframe(st, site_index)
        ser_thr_kinases = _kinase_columns(st)
    if n_ty > 0:
        ty = pp.score(kin_type="tyrosine", round_digits=round_digits)
        adata.varm[f"{varm_prefix}_tyrosine"] = _to_varm_dataframe(ty, site_index)
        tyrosine_kinases = _kinase_columns(ty)

    adata.uns["alphaphos_kinase"] = {
        "library": "kinase-library (Johnson et al. Nature 2023)",
        "n_sites_scored": n_scored,
        "n_sites_dropped": n_dropped_total,
        "n_ser_thr_scored": n_st,
        "n_tyrosine_scored": n_ty,
        "ser_thr_kinases": ser_thr_kinases,
        "tyrosine_kinases": tyrosine_kinases,
        "score_type": "score",
        "round_digits": round_digits,
    }
    logger.info(
        "score_kinases: scored %d sites (%d S/T + %d Y); %d sites dropped (invalid/missing)",
        n_scored,
        n_st,
        n_ty,
        n_dropped_total,
    )


def predict_kinases(
    adata: ad.AnnData,
    *,
    sequence_col: str = "kinase_sequence",
    top_k: int = 5,
    varm_prefix: str = "kinase_score",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Return the top-k predicted upstream kinases per site.

    Runs :func:`score_kinases` first if the per-site score matrices aren't
    already in ``adata.varm`` (or if ``overwrite=True``).

    Parameters
    ----------
    adata
        AnnData with a ``kinase_sequence`` column in ``.var``.
    sequence_col
        ``var`` column with the alphaPhos-format kinase sequences.
    top_k
        Number of top kinases to report per site (default 5).
    varm_prefix
        Prefix for the varm keys written by ``score_kinases``.
    overwrite
        If True, re-run scoring even if the varm matrices already exist.

    Returns
    -------
    pd.DataFrame indexed by ``adata.var.index`` with columns:

    - ``top1_kinase``, ``top1_score``, ..., ``top{k}_kinase``, ``top{k}_score``
    - ``top_kinases``: comma-joined list of the top-k kinase names
    - ``top_scores``:  comma-joined list of the corresponding scores
    - ``kin_type``:    ``"ser_thr"``, ``"tyrosine"``, or ``"none"`` (site dropped)

    Notes
    -----
    Sites with invalid sequences appear as rows with ``kin_type='none'`` and
    NaN scores. This makes the result safe to ``adata.var.join(...)`` directly.
    """
    st_key = f"{varm_prefix}_ser_thr"
    y_key = f"{varm_prefix}_tyrosine"

    need_score = overwrite or (st_key not in adata.varm and y_key not in adata.varm)
    if need_score:
        score_kinases(adata, sequence_col=sequence_col, varm_prefix=varm_prefix)

    out_cols: list[str] = []
    for i in range(1, top_k + 1):
        out_cols.extend([f"top{i}_kinase", f"top{i}_score"])
    out_cols.extend(["top_kinases", "top_scores", "kin_type"])

    out = pd.DataFrame(
        {c: pd.Series(pd.NA, index=adata.var.index, dtype="object") for c in out_cols},
    )

    def _process_varm(varm_key: str, kin_type: str) -> None:
        if varm_key not in adata.varm:
            return
        mat = adata.varm[varm_key]
        if not isinstance(mat, pd.DataFrame):
            # AnnData converted it to ndarray; bring back column names
            mat = pd.DataFrame(
                mat,
                index=adata.var.index,
                columns=adata.uns["alphaphos_kinase"][f"{kin_type}_kinases"],
            )
        # Sites with any non-NaN row got scored
        any_score = mat.notna().any(axis=1)
        if not any_score.any():
            return
        scored = mat.loc[any_score]
        kinases = np.asarray(scored.columns)

        # Vectorised top-k: argpartition then sort each row's top-k
        k = min(top_k, scored.shape[1])
        arr = scored.values
        # argpartition for top-k (descending: negate)
        top_idx = np.argpartition(-arr, k - 1, axis=1)[:, :k]
        # Now sort each row's top-k by score desc
        row_idx = np.arange(arr.shape[0])[:, None]
        # Build scores at top_idx then sort
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
    return out
