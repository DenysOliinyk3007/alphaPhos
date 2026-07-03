"""Phase 4 validation: score alphaPhos enrichment on the EGF walkthrough
result against the Hernández-Armenta / Ochoa EV3 gold standard.

Runs GSEA with an ad-hoc kinase-substrate library extracted from the PTM
functional DB, then scores the recovered kinase activities against
EV3's expected up/down directions for EGF-family conditions.

The full 132-pair AUC ≈ 0.72 target requires the raw phospho matrices
for all 99 non-EGF benchmark conditions from PRIDE -- a per-condition
data-fetch task outside this script.  See section 10.4 of
``test_data/ptmsea_plus_scaffold/PhosphoModules_project.md``.

Usage
-----
    python scripts/validate_enrichment_ev3.py \\
        --diff-exp test_data/walkthrough_output/egf_diff_exp_result.tsv \\
        --db resources/ptm_functional_db.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from alphaphos.enrichment import (
    attach_site_ids,
    build_kinase_substrate_library,
    ev3_expectations_for_condition,
    gsea,
    score_against_ev3,
)

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
log = logging.getLogger("validate")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--diff-exp", required=True, type=Path)
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--condition", default="EGF", help="EV3 Condition string (exact by default)")
    p.add_argument(
        "--substring",
        action="store_true",
        help="Match --condition as a case-insensitive substring instead of exactly "
        "(warning: 'EGF' matches VEGF, EGFRi, EGF+U0126 -- rarely what you want)",
    )
    p.add_argument("--min-set-size", type=int, default=5)
    p.add_argument("--n-permutations", type=int, default=5000)
    p.add_argument("--curated-only", action="store_true", default=True)
    args = p.parse_args(argv)

    if not args.diff_exp.exists():
        log.error("diff-exp file not found: %s", args.diff_exp)
        return 2
    if not args.db.exists():
        log.error("db parquet not found: %s -- build via scripts/build_ptm_db.py", args.db)
        return 2

    # ------------------------------------------------------------------
    # 1. Load caller's diff-exp result, canonicalize site IDs
    # ------------------------------------------------------------------
    log.info("loading diff-exp result: %s", args.diff_exp)
    result = pd.read_csv(args.diff_exp, sep="\t", index_col=0)
    log.info("  %d sites", len(result))
    result = attach_site_ids(result, db_path=args.db)
    matched = result.dropna(subset=["site_id"])
    log.info("  %d / %d sites matched to DB", len(matched), len(result))

    # Deduplicate: multiple alphaphos keys can map to the same site_id
    # (multiplicity variants); keep the strongest per site.
    matched = matched.sort_values("t_stat", key=abs, ascending=False)
    matched = matched[~matched["site_id"].duplicated(keep="first")]
    log.info("  %d unique site_ids after dedup", len(matched))

    # ------------------------------------------------------------------
    # 2. Build the validation-only kinase-substrate library from the DB
    # ------------------------------------------------------------------
    ks_lib = build_kinase_substrate_library(
        db_path=args.db,
        min_set_size=args.min_set_size,
        require_curated=args.curated_only,
    )
    log.info("built kinase-substrate library: %d kinases", len(ks_lib))

    # ------------------------------------------------------------------
    # 3. GSEA — rank by log2fc (signed), score every kinase
    # ------------------------------------------------------------------
    ranked = matched.set_index("site_id")["log2fc"]
    gsea_result = gsea(
        ranked,
        {"kinase_substrate": ks_lib},
        min_set_size=args.min_set_size,
        max_set_size=100_000,  # no upper cap for kinase substrate sets
        n_permutations=args.n_permutations,
        seed=42,
    )
    log.info("GSEA scored %d kinases", len(gsea_result))

    # NES per kinase (already signed: positive NES = up, negative = down)
    nes = gsea_result.set_index("set_name")["NES"].dropna()

    # ------------------------------------------------------------------
    # 4. EV3 expectations + scoring
    # ------------------------------------------------------------------
    from alphaphos.enrichment import load_ev3

    ev3 = load_ev3()
    if args.substring:
        expected = ev3_expectations_for_condition(args.condition, ev3=ev3)
        match_desc = f"substring: '{args.condition}'"
    else:
        expected = ev3[ev3["Condition"] == args.condition]
        match_desc = f"exact: '{args.condition}'"
    print("\n" + "=" * 70)
    print(f"EV3 expectations for {match_desc}")
    print("=" * 70)
    print(
        expected[["Condition ID", "Condition", "Kinase", "Directionality"]]
        .drop_duplicates(subset=["Condition", "Kinase", "Directionality"])
        .to_string(index=False)
    )

    report = score_against_ev3(
        nes,
        condition_query=args.condition,
        exact_condition=None if args.substring else args.condition,
    )
    per_kin = report["per_kinase"]

    print("\n" + "=" * 70)
    print(f"alphaPhos validation report vs EV3 ({args.condition})")
    print("=" * 70)
    print(f"  n_expected pairs: {report['n_expected']}")
    print(f"  n_recovered (direction correct): {report['n_recovered']}")
    print(f"  recall_direction: {report['recall_direction']:.2%}")
    if report["auc"] is not None:
        print(f"  ROC AUC (up vs down): {report['auc']:.3f}")
    else:
        print("  ROC AUC: N/A (only one direction in expected set)")
    print()
    print("Per-kinase detail:")
    print(per_kin.to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Top-N enriched kinases from our own tool (for context)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Top 15 kinases by |NES| (alphaPhos GSEA output on EGF data)")
    print("=" * 70)
    top = gsea_result.reindex(gsea_result["NES"].abs().sort_values(ascending=False).index).head(15)
    print(
        top[["set_name", "n_set", "NES", "p_value", "fdr", "direction"]]
        .round(4)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
