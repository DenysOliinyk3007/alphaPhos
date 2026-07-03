"""Validation helpers: score alphaPhos enrichment output against the
Hernández-Armenta / Ochoa kinase-activity gold standard.

The Beltrao-lab benchmark (Hernández-Armenta et al. 2017, ``Bioinformatics``
33:1845 and Ochoa et al. 2016 atlas EV3) is the field-standard target for
kinase-activity inference from phosphoproteomics.  It publishes, for each
of 132 (condition, kinase) pairs, whether the kinase is *expected* to be
regulated up or down.  Best inference methods achieve **mean AUC ≈ 0.72**
on this benchmark.

The atlas ships an *expected-answer* table (EV3) and a *reference
activity matrix* (EV2, 215 kinases × 399 conditions), but NOT the raw
per-site phospho fold-change matrix that a tool needs as input for the
other 398 conditions -- those live on ProteomeXchange/PRIDE and need
per-condition fetch + parsing.

This module provides three things:

1. :func:`build_kinase_substrate_library` -- ad-hoc extraction of a
   kinase-substrate GMT from the DB, keyed by ``substrate_gene``.  Not
   part of the shipped v1 library set because
   :func:`alphaphos.enrichment.kinase_activity` and
   :mod:`alphaphos.kinase.library` cover kinase-activity inference more
   directly.  Provided here so the same DB substrate can be tested
   against EV3.
2. :func:`load_ev3` / :func:`load_ev2` -- load the bundled gold-standard
   tables from ``resources/goldstandard/``.
3. :func:`score_against_ev3` -- given the caller's per-kinase activity
   scores for a specific condition, compute the recall of the expected
   direction (up/down) among EV3's expected kinases for that condition,
   plus a simple ROC AUC over all pairs in the condition.

Full 132-pair AUC requires PRIDE-fetching the other 99 conditions --
that's user-side data engineering outside the scope of this module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alphaphos.enrichment.db import (
    COL_ENZYME_GENE,
    COL_IS_CURATED,
    COL_POSITION,
    COL_RESIDUE,
    COL_UNIPROT,
    load_ptm_db,
    site_id,
)

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_GOLDSTANDARD_DIR = _REPO_ROOT / "resources" / "goldstandard"
EV3_PATH = _GOLDSTANDARD_DIR / "goldstandard_OchoaAtlas2016_EV3_expected_regulation.csv"
EV2_PATH = _GOLDSTANDARD_DIR / "reference_OchoaAtlas2016_EV2_KSEA_activities.csv"


# ---------------------------------------------------------------------------
# Kinase-substrate library (validation-only, not part of v1 emit)
# ---------------------------------------------------------------------------


def build_kinase_substrate_library(
    *,
    db_path: str | Path | None = None,
    db: pd.DataFrame | None = None,
    min_set_size: int = 5,
    require_curated: bool = True,
) -> dict[str, list[str]]:
    """Extract one set per kinase from the DB, keyed by kinase gene symbol.

    Members are the canonical ``Protein_AApos`` IDs of the kinase's
    curated substrate sites, deduplicated per set.  Kinases with fewer
    than ``min_set_size`` substrates are dropped.

    Not emitted from :func:`alphaphos.enrichment.emit_libraries` in v1
    because kinase-activity inference is already the mandate of
    :func:`alphaphos.enrichment.kinase_activity` and
    :mod:`alphaphos.kinase.library`.  Built here for benchmark
    validation.

    Parameters
    ----------
    require_curated
        If True (default), restrict to curated DB rows.  Trades recall
        for precision.  Kinase-activity inference is very sensitive to
        substrate-set quality, so ``True`` is the defensible default.

    Returns
    -------
    dict[str, list[str]]
        ``{kinase_gene: [site_ids]}``.  Compatible with the ``libraries``
        argument of :func:`alphaphos.enrichment.ora` / :func:`gsea` when
        wrapped in an outer dict:
        ``{"kinase_substrate": build_kinase_substrate_library()}``.
    """
    if db is None:
        db = load_ptm_db(db_path)

    df = db.dropna(subset=[COL_ENZYME_GENE, COL_UNIPROT, COL_RESIDUE, COL_POSITION])
    if require_curated and COL_IS_CURATED in df.columns:
        df = df[df[COL_IS_CURATED].astype("boolean").fillna(False)]

    sets: dict[str, list[str]] = {}
    for kinase, group in df.groupby(COL_ENZYME_GENE):
        # substrate_uniprot can be semicolon-joined; site_id takes first.
        sids = {
            site_id(u, r, p)
            for u, r, p in zip(
                group[COL_UNIPROT].astype(str),
                group[COL_RESIDUE].astype(str),
                group[COL_POSITION],
                strict=True,
            )
        }
        if len(sids) < min_set_size:
            continue
        sets[str(kinase)] = sorted(sids)
    logger.info(
        "kinase_substrate library: %d kinases with >=%d substrates (curated=%s)",
        len(sets),
        min_set_size,
        require_curated,
    )
    return sets


# ---------------------------------------------------------------------------
# Gold-standard loaders
# ---------------------------------------------------------------------------


def load_ev3(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Ochoa EV3 expected-regulation table.

    Columns: ``Condition ID``, ``Condition``, ``Kinase``,
    ``Directionality`` (``up`` / ``down``), ``Pubmed``, ``Time (min)``.
    """
    return pd.read_csv(Path(path) if path is not None else EV3_PATH)


def load_ev2(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Ochoa EV2 reference kinase-activity matrix.

    Wide format: 215 kinases × 399 conditions.  The first column is
    ``Kinase``; every other column is a condition label such as
    ``EGF_vs_control_5.00m``.
    """
    return pd.read_csv(Path(path) if path is not None else EV2_PATH)


def ev3_expectations_for_condition(
    condition_query: str,
    *,
    ev3: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return EV3 rows whose ``Condition`` matches ``condition_query`` as a
    case-insensitive substring.

    ``condition_query="EGF"`` matches "EGF", "EGF + U0126", "EGFRi
    (PD-153035)" etc.  For a stricter match (e.g. "EGF" without inhibitor
    combinations), post-filter the returned DataFrame.
    """
    if ev3 is None:
        ev3 = load_ev3()
    mask = (
        ev3["Condition"]
        .astype(str)
        .str.contains(condition_query, case=False, na=False, regex=False)
    )
    return ev3[mask].copy()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_against_ev3(
    kinase_scores: pd.Series,
    condition_query: str,
    *,
    ev3: pd.DataFrame | None = None,
    exact_condition: str | None = None,
) -> dict[str, object]:
    """Score a per-kinase activity vector against EV3 expectations.

    Parameters
    ----------
    kinase_scores
        Series indexed by kinase gene symbol, values = signed activity
        (e.g. NES from :func:`gsea` with the kinase-substrate library).
        Positive = up, negative = down.
    condition_query
        Substring to filter EV3 conditions (case-insensitive).  Use
        ``exact_condition`` for a strict match.
    exact_condition
        If given, EV3 rows are filtered by exact equality on the
        ``Condition`` column (overrides the substring match).

    Returns
    -------
    dict
        Report with:

        - ``n_expected`` -- number of EV3 (kinase, direction) pairs
        - ``n_recovered`` -- kinases we scored with the right sign
        - ``recall_direction`` -- ``n_recovered / n_expected``
        - ``auc`` -- ROC AUC treating the signed EV3 direction as label
          and our score as the ranker; only computed if there are both
          up- and down-expected kinases in the condition
        - ``per_kinase`` -- DataFrame with columns
          ``kinase``, ``expected_direction``, ``observed_score``,
          ``observed_direction``, ``direction_correct``
    """
    if ev3 is None:
        ev3 = load_ev3()

    if exact_condition is not None:
        expected = ev3[ev3["Condition"] == exact_condition]
    else:
        expected = ev3_expectations_for_condition(condition_query, ev3=ev3)

    per_kinase_rows: list[dict] = []
    for _, row in expected.iterrows():
        kinase = str(row["Kinase"])
        expected_dir = str(row["Directionality"])
        obs_score = kinase_scores.get(kinase, float("nan"))
        obs_dir = _dir_from_score(obs_score)
        correct = obs_dir == expected_dir if not np.isnan(obs_score) else False
        per_kinase_rows.append(
            {
                "kinase": kinase,
                "expected_direction": expected_dir,
                "observed_score": obs_score,
                "observed_direction": obs_dir,
                "direction_correct": bool(correct),
            }
        )

    # Dedup on (kinase, direction) so a benchmark row appearing in
    # multiple original studies is counted once.  Recall is measured on
    # the deduped set so a highly-cited kinase doesn't dominate.
    per_kinase = pd.DataFrame(per_kinase_rows).drop_duplicates(
        subset=["kinase", "expected_direction"]
    )
    n_recovered = int(per_kinase["direction_correct"].sum())

    auc = _compute_directional_auc(per_kinase)

    return {
        "n_expected": len(per_kinase),
        "n_recovered": n_recovered,
        "recall_direction": n_recovered / max(len(per_kinase), 1),
        "auc": auc,
        "per_kinase": per_kinase,
    }


def _dir_from_score(score: float) -> str:
    if np.isnan(score):
        return "missing"
    return "up" if score > 0 else "down"


def _compute_directional_auc(per_kinase: pd.DataFrame) -> float | None:
    """Compute ROC AUC where the positive class is `expected == "up"`.

    Returns None if only one class is present (AUC undefined).
    Uses the standard rank-based computation without sklearn (avoids a
    dependency for this validation-only utility).
    """
    scores = per_kinase["observed_score"].to_numpy()
    labels = (per_kinase["expected_direction"].to_numpy() == "up").astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    valid = ~np.isnan(scores)
    if valid.sum() < 2:
        return None
    scores = scores[valid]
    labels = labels[valid]
    # Rank-based AUC (Mann-Whitney U): fraction of (pos, neg) pairs
    # where pos > neg, with ties counted as 0.5.
    order = np.argsort(scores)
    ranks = np.empty_like(scores, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)
