"""Convert the PTM functional database Excel workbook to a parquet used by
``alphaphos.enrichment``.

One-shot developer script.  The Excel workbook (``PTM_functional_
databases_data_final.xlsx``, ~122 MB, 22 sheets, 590k rows) is unwieldy
for runtime use; this script reads ``Master_Integrated``, restricts to
phosphorylation rows for the v1 enrichment engine, normalizes the effect
vocabulary to a canonical 6-way categorical, and writes a compact parquet
under ``resources/ptm_functional_db.parquet``.  The full workbook is
never distributed with the package (see gitignore + Zenodo plan); the
parquet is what alphaphos.enrichment reads.

Usage
-----
    python scripts/build_ptm_db.py \\
        --xlsx test_data/ptmsea_plus_scaffold/PTM_functional_databases_data_final.xlsx \\
        --out  resources/ptm_functional_db.parquet \\
        --fixture tests/data/ptm_db_mini.parquet

The optional ``--fixture`` path emits a hand-sampled ~1000-row subset
(covering every library type) as a small parquet committed to git so CI
can exercise the enrichment module without the full DB present.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
log = logging.getLogger("build_ptm_db")


# Columns we actually keep in the parquet.  Others in the xlsx are dropped.
KEEP_COLUMNS = [
    "substrate_gene",
    "substrate_uniprot",
    "residue",
    "position",
    "modification",
    "enzyme_gene",
    "enzyme_uniprot",
    "effect",
    "interaction_partner",
    "source_dbs",
    "n_sources",
    "curation_tier",
    "curation_confidence",
    "is_curated",
    "n_pmids",
    "Ochoa_functional_score",
    "PSP_reg_functional_effect",
    "ADB_disease_variant",
    "ELM_motifs_protein",
    "chemistry_warning",
]


# Normalize the messy ``effect`` free-text into 6 canonical categories.
# Everything else -> NaN (kept in the row, just not usable for effect-set
# emission).  Order-of-matches follows the substring heuristic below.
EFFECT_RULES: tuple[tuple[str, str], ...] = (
    ("up-regulates activity", "activates_activity"),
    ("up-regulates quantity by stabilization", "alters_stability"),
    ("up-regulates quantity", "alters_stability"),
    ("up-regulates", "activates_activity"),
    ("down-regulates activity", "inhibits_activity"),
    ("down-regulates quantity by destabilization", "alters_stability"),
    ("down-regulates quantity", "alters_stability"),
    ("down-regulates", "inhibits_activity"),
    ("enhances ppi", "induces_ppi"),
    ("induces ppi", "induces_ppi"),
    ("induce", "induces_ppi"),  # SIGNOR shorthand
    ("inhibits ppi", "disrupts_ppi"),
    ("inhibit", "inhibits_activity"),  # SIGNOR shorthand
    ("localization", "alters_localization"),
    ("stability", "alters_stability"),
)


def _normalize_effect(effect: str | float) -> str | float:
    if not isinstance(effect, str) or not effect.strip():
        return np.nan
    e = effect.strip().lower()
    for needle, category in EFFECT_RULES:
        if needle in e:
            return category
    return np.nan  # unknown / unmapped -> not usable for enrichment sets


def _load_master(xlsx_path: Path) -> pd.DataFrame:
    log.info("reading Master_Integrated from %s ...", xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name="Master_Integrated", usecols=KEEP_COLUMNS)
    log.info("  loaded %d rows x %d cols", len(df), df.shape[1])
    return df


def _filter_phospho(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[df["modification"] == "phosphorylation"].copy()
    log.info("phospho filter: %d -> %d rows", n_before, len(df))
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce final dtypes + emit the canonical effect column."""
    df = df.copy()
    df["position"] = df["position"].astype("Int64")
    df["is_curated"] = df["is_curated"].astype("boolean")
    df["Ochoa_functional_score"] = pd.to_numeric(df["Ochoa_functional_score"], errors="coerce")
    df["n_sources"] = df["n_sources"].astype("Int32")
    df["n_pmids"] = df["n_pmids"].astype("Int32")
    df["effect_canonical"] = df["effect"].map(_normalize_effect).astype("category")
    df["chemistry_warning_flag"] = df["chemistry_warning"].notna()
    return df


def _write_full(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info("wrote %s (%.1f MB, %d rows)", out_path, size_mb, len(df))


def _build_fixture(df: pd.DataFrame, out_path: Path, target_n: int = 1000) -> None:
    """Sample ~target_n rows covering every library type for CI tests."""
    rng = np.random.default_rng(0)
    buckets: list[pd.DataFrame] = []

    def _sample(condition: pd.Series, n: int, label: str) -> None:
        pool = df[condition]
        take = min(n, len(pool))
        if take == 0:
            log.warning("fixture: no rows for %s", label)
            return
        picked = pool.sample(n=take, random_state=rng.integers(0, 2**32 - 1))
        log.info("fixture: %s -> %d rows", label, take)
        buckets.append(picked)

    # Kinase-substrate rows (enzyme present)
    _sample(df["enzyme_gene"].notna(), 300, "kinase-substrate")
    # Effect-annotated
    _sample(df["effect_canonical"].notna(), 200, "effect_canonical")
    # Disease variant
    _sample(df["ADB_disease_variant"].notna(), 200, "disease_variant")
    # High Ochoa score
    _sample(df["Ochoa_functional_score"] >= 0.75, 200, "ochoa>=0.75")
    # Low Ochoa score
    _sample(df["Ochoa_functional_score"] < 0.25, 200, "ochoa<0.25")
    # High-confidence curated backbone
    _sample(df["curation_confidence"] == "high", 200, "curation=high")

    fixture = pd.concat(buckets).drop_duplicates(
        subset=["substrate_uniprot", "residue", "position"]
    )
    log.info("fixture total (deduped by site): %d rows", len(fixture))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_kb = out_path.stat().st_size / 1024
    log.info("wrote %s (%.1f KB)", out_path, size_kb)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--xlsx", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="Full parquet output path.")
    p.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Optional small parquet with ~1000 hand-sampled sites for CI.",
    )
    args = p.parse_args(argv)

    if not args.xlsx.exists():
        log.error("xlsx not found: %s", args.xlsx)
        return 2

    df = _load_master(args.xlsx)
    df = _filter_phospho(df)
    df = _normalize_columns(df)

    log.info("effect_canonical value_counts:")
    log.info("\n%s", df["effect_canonical"].value_counts(dropna=False).to_string())
    log.info("curation_confidence value_counts:")
    log.info("\n%s", df["curation_confidence"].value_counts(dropna=False).to_string())

    _write_full(df, args.out)

    if args.fixture is not None:
        _build_fixture(df, args.fixture)

    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
