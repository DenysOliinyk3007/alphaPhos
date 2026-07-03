"""Emit GMT-format site-set libraries from the PTM functional database.

Three library types ship in v1 (design doc, phase 1):

1. **FUNCTIONAL_EFFECT**  -- one set per canonical effect category
   (``activates_activity``, ``inhibits_activity``, ``induces_ppi``,
   ``disrupts_ppi``, ``alters_stability``).  Drops
   ``alters_localization`` — too few rows (~141) to survive
   ``min_set_size``.  **Differentiator vs existing tools.**
2. **DISEASE_VARIANT**  -- one set per ActiveDriverDB variant dataset
   (``clinvar_site``, ``cancer_TCGA_site``).  **Differentiator.**
3. **FUNCTIONAL_SCORE_BIN**  -- two sets from the Ochoa 2020 functional
   score: top-quartile (``ochoa_top_quartile``) and bottom-quartile
   (``ochoa_bottom_quartile``) as a negative control.

**KINASE_SUBSTRATE** is intentionally NOT emitted as a separate GMT:
:func:`alphaphos.enrichment.kinase_activity` covers kinase-activity
inference directly (decoupler ULM + OmniPath / PTM-DB) and
:mod:`alphaphos.kinase.library` covers sequence-based PWM prediction.
This enrichment module's differentiator is the functional-effect /
disease-variant / functional-score axis, which those tools don't touch.

**KINASE_MOTIF_PREDICTED** is out of scope for v1 — use
``alphaphos.kinase.library.predict_kinases`` on the AnnData directly.

Output format: GMT (Broad Institute convention):

    set_name<TAB>description<TAB>site_1<TAB>site_2<TAB>...

One GMT file per library type + a ``manifest.csv`` summarising all sets
across all libraries (source_dbs, mean curation_confidence, n_sites).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from alphaphos.enrichment.db import (
    COL_DISEASE_VARIANT,
    COL_EFFECT_CANONICAL,
    COL_OCHOA_SCORE,
    COL_POSITION,
    COL_RESIDUE,
    COL_UNIPROT,
    load_ptm_db,
    site_id,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_MIN_SET_SIZE = 5  # PTM-SEA convention

# Curated fixed order for effect sets so the emitted GMT is stable.
FUNCTIONAL_EFFECT_CATEGORIES: tuple[str, ...] = (
    "activates_activity",
    "inhibits_activity",
    "induces_ppi",
    "disrupts_ppi",
    "alters_stability",
    # 'alters_localization' intentionally omitted — too few sites (141)
    # to survive min_set_size, and mixing it with the others would
    # unbalance the multiple-testing correction.
)

FUNCTIONAL_EFFECT_DESCRIPTIONS: dict[str, str] = {
    "activates_activity": "Phosphosite whose modification up-regulates the substrate's activity.",
    "inhibits_activity": "Phosphosite whose modification down-regulates the substrate's activity.",
    "induces_ppi": "Phosphosite whose modification enhances a protein-protein interaction.",
    "disrupts_ppi": "Phosphosite whose modification disrupts a protein-protein interaction.",
    "alters_stability": "Phosphosite whose modification alters substrate stability.",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def emit_libraries(
    output_dir: str | Path,
    *,
    db_path: str | Path | None = None,
    min_set_size: int = DEFAULT_MIN_SET_SIZE,
    require_curated: bool = False,
    drop_chemistry_warned: bool = True,
) -> pd.DataFrame:
    """Emit the v1 site-set libraries to ``output_dir`` as GMT files.

    Parameters
    ----------
    output_dir
        Where to write the GMT files and manifest.
    db_path
        Optional path to the PTM DB parquet. Defaults to
        ``resources/ptm_functional_db.parquet`` (see
        :func:`alphaphos.enrichment.db.load_ptm_db`).
    min_set_size
        Minimum number of unique sites required for a set to be emitted.
        Sets below this are silently dropped (logged at INFO). PTM-SEA
        convention is 5.
    require_curated
        If True, restrict library construction to rows where
        ``is_curated`` is True (i.e. drop the ``experimental_invitro``
        tier). Trades recall for precision.
    drop_chemistry_warned
        If True (default), drop rows flagged as chemistry-atypical
        (e.g. inherited methylation on cysteine from iPTMnet). Almost
        always what you want.

    Returns
    -------
    pandas.DataFrame
        The set-level manifest: one row per set, columns
        ``library``, ``set_name``, ``n_sites``, ``description``,
        plus provenance summaries.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = load_ptm_db(db_path)
    db = _apply_pre_filters(
        db, require_curated=require_curated, drop_chemistry_warned=drop_chemistry_warned
    )

    manifest_rows: list[dict] = []

    functional_sets = _build_functional_effect_sets(db, min_set_size=min_set_size)
    _write_gmt(
        functional_sets, FUNCTIONAL_EFFECT_DESCRIPTIONS, output_dir / "functional_effect.gmt"
    )
    manifest_rows.extend(
        _manifest_entries(functional_sets, "FUNCTIONAL_EFFECT", db, FUNCTIONAL_EFFECT_DESCRIPTIONS)
    )

    disease_sets = _build_disease_variant_sets(db, min_set_size=min_set_size)
    disease_descriptions = {
        "clinvar_site": "Phosphosite with a ClinVar disease-associated variant at the exact residue.",
        "cancer_TCGA_site": "Phosphosite with a TCGA/MC3 cancer mutation at the exact residue.",
    }
    _write_gmt(disease_sets, disease_descriptions, output_dir / "disease_variant.gmt")
    manifest_rows.extend(
        _manifest_entries(disease_sets, "DISEASE_VARIANT", db, disease_descriptions)
    )

    score_sets = _build_functional_score_sets(db, min_set_size=min_set_size)
    score_descriptions = {
        "ochoa_top_quartile": "Phosphosite in the top 25% of Ochoa et al. 2020 functional score (most likely functional).",
        "ochoa_bottom_quartile": "Phosphosite in the bottom 25% of Ochoa et al. 2020 functional score (negative control).",
    }
    _write_gmt(score_sets, score_descriptions, output_dir / "functional_score.gmt")
    manifest_rows.extend(
        _manifest_entries(score_sets, "FUNCTIONAL_SCORE_BIN", db, score_descriptions)
    )

    # Always write a valid header (columns) even if no sets survived the
    # min_set_size / require_curated filters — makes downstream reads safe.
    manifest = pd.DataFrame(
        manifest_rows, columns=["library", "set_name", "n_sites", "description"]
    )
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info(
        "Wrote %d sets across %d libraries to %s",
        len(manifest),
        manifest["library"].nunique(),
        output_dir,
    )
    return manifest


def load_gmt(path: str | Path) -> dict[str, list[str]]:
    """Read a GMT file and return ``{set_name: [site_id, ...]}``.

    The description column (2nd tab-delimited field) is dropped;
    callers who need it should re-read the file directly.
    """
    sets: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            name, _description, *members = fields
            sets[name] = [m for m in members if m]
    return sets


def load_libraries(directory: str | Path) -> dict[str, dict[str, list[str]]]:
    """Load every ``.gmt`` in a directory into a nested dict.

    Returns ``{library_type: {set_name: [site_ids]}}`` keyed on the
    ``.gmt`` filename stem (e.g. ``functional_effect``,
    ``disease_variant``, ``functional_score``).

    Deterministic library ordering follows sorted filename.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"libraries directory not found: {directory}")
    out: dict[str, dict[str, list[str]]] = {}
    for gmt_path in sorted(directory.glob("*.gmt")):
        out[gmt_path.stem] = load_gmt(gmt_path)
    if not out:
        raise FileNotFoundError(f"no .gmt files under {directory}")
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _apply_pre_filters(
    db: pd.DataFrame,
    *,
    require_curated: bool,
    drop_chemistry_warned: bool,
) -> pd.DataFrame:
    n0 = len(db)
    if drop_chemistry_warned and "chemistry_warning_flag" in db.columns:
        db = db[~db["chemistry_warning_flag"].astype(bool)]
        logger.info("drop_chemistry_warned: %d -> %d rows", n0, len(db))
        n0 = len(db)
    if require_curated and "is_curated" in db.columns:
        db = db[db["is_curated"].astype("boolean").fillna(False)]
        logger.info("require_curated: %d -> %d rows", n0, len(db))
    return db


def _site_ids(df: pd.DataFrame) -> list[str]:
    """Convert a per-row DataFrame to a deduplicated list of site_ids."""
    ids = [
        site_id(u, r, p)
        for u, r, p in zip(
            df[COL_UNIPROT].astype(str),
            df[COL_RESIDUE].astype(str),
            df[COL_POSITION],
            strict=True,
        )
    ]
    # Preserve first-occurrence order while deduping (deterministic output).
    seen: set[str] = set()
    out: list[str] = []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _build_functional_effect_sets(db: pd.DataFrame, *, min_set_size: int) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    scoped = db[db[COL_EFFECT_CANONICAL].isin(FUNCTIONAL_EFFECT_CATEGORIES)]
    for category in FUNCTIONAL_EFFECT_CATEGORIES:
        rows = scoped[scoped[COL_EFFECT_CANONICAL] == category]
        members = _site_ids(rows)
        if len(members) >= min_set_size:
            sets[category] = members
        else:
            logger.info(
                "FUNCTIONAL_EFFECT: dropping %r (n=%d < min_set_size=%d)",
                category,
                len(members),
                min_set_size,
            )
    return sets


def _build_disease_variant_sets(db: pd.DataFrame, *, min_set_size: int) -> dict[str, list[str]]:
    # ADB_disease_variant is a semicolon-joined label list like
    # "ClinVar;MC3_TCGA_cancer".  We test containment against the two
    # dataset names.
    variant_map = {
        "clinvar_site": ("clinvar",),
        "cancer_TCGA_site": ("mc3_tcga", "tcga"),
    }
    sets: dict[str, list[str]] = {}
    variants = db[db[COL_DISEASE_VARIANT].notna()]
    for set_name, tokens in variant_map.items():
        lower = variants[COL_DISEASE_VARIANT].astype(str).str.lower()
        mask = lower.apply(lambda s, toks=tokens: any(t in s for t in toks))
        members = _site_ids(variants[mask])
        if len(members) >= min_set_size:
            sets[set_name] = members
        else:
            logger.info(
                "DISEASE_VARIANT: dropping %r (n=%d < min_set_size=%d)",
                set_name,
                len(members),
                min_set_size,
            )
    return sets


def _build_functional_score_sets(db: pd.DataFrame, *, min_set_size: int) -> dict[str, list[str]]:
    scored = db[db[COL_OCHOA_SCORE].notna()]
    if scored.empty:
        return {}
    # Quantiles computed on unique sites, not on rows, so multi-source
    # duplication of a single site doesn't skew the cutoff.
    per_site = scored.drop_duplicates([COL_UNIPROT, COL_RESIDUE, COL_POSITION])
    q25 = per_site[COL_OCHOA_SCORE].quantile(0.25)
    q75 = per_site[COL_OCHOA_SCORE].quantile(0.75)
    logger.info(
        "FUNCTIONAL_SCORE quantiles: q25=%.3f  q75=%.3f  (n_scored_sites=%d)",
        q25,
        q75,
        len(per_site),
    )
    top = per_site[per_site[COL_OCHOA_SCORE] >= q75]
    bottom = per_site[per_site[COL_OCHOA_SCORE] <= q25]
    sets: dict[str, list[str]] = {}
    for name, subset in [("ochoa_top_quartile", top), ("ochoa_bottom_quartile", bottom)]:
        members = _site_ids(subset)
        if len(members) >= min_set_size:
            sets[name] = members
        else:
            logger.info(
                "FUNCTIONAL_SCORE_BIN: dropping %r (n=%d < min_set_size=%d)",
                name,
                len(members),
                min_set_size,
            )
    return sets


def _write_gmt(
    sets: dict[str, list[str]],
    descriptions: dict[str, str],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        for name, members in sets.items():
            description = descriptions.get(name, "")
            row = "\t".join([name, description, *members]) + "\n"
            f.write(row)
    logger.info("Wrote %d sets to %s", len(sets), path)


def _manifest_entries(
    sets: dict[str, list[str]],
    library: str,
    db: pd.DataFrame,
    descriptions: dict[str, str],
) -> list[dict]:
    """Per-set summary row for the manifest CSV."""
    rows: list[dict] = []
    for name, members in sets.items():
        rows.append(
            {
                "library": library,
                "set_name": name,
                "n_sites": len(members),
                "description": descriptions.get(name, ""),
            }
        )
    return rows
