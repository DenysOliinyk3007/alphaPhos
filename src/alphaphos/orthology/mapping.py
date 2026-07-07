"""Window-based cross-species phosphosite ortholog mapping to human.

Algorithm
---------
1. **Build the human window index** (once, cached to parquet).  For every
   S/T/Y in the human proteome, extract the ``±window_size`` residue window
   and store ``(window, uniprot, gene, position, residue, is_reviewed)``.
2. **Build a decoy index** the same way from a per-protein shuffle of the
   human proteome that preserves S/T/Y positions (so the phospho-acceptor
   density and window count are identical to target).  This is the
   Elias-Gygi 2007 target-decoy strategy adapted to sequence-window matching.
3. **For each source site** (each row of ``adata.var`` with a populated
   ``kinase_sequence`` column), do exact lookups in both indexes:

   - Target hit + no decoy hit &rarr; confident mapping.
   - Target hit + decoy hit at same window &rarr; ambiguous, still emit but
     flag; contributes to the global-FDR denominator.
   - No target hit &rarr; unmapped.

4. **Report global FDR** = ``n_decoy_hits / max(n_target_hits, 1)``.
5. Emit annotation columns on ``adata.var``: ``human_gene``,
   ``human_uniprot``, ``human_site``, ``human_site_key``, ``site_conserved``,
   ``ortholog_ambiguous``, ``mapping_source``, plus paralog list.

Phase 2 (this release) adds fuzzy fallback (Hamming &le; ``max_mismatches``)
via a pigeonhole segment index, per-site q-values from target-decoy
ranking, and an ``fdr_threshold`` filter that demotes sites above the
threshold to ``mapping_source="below_fdr"``.  Sites where the decoy hit
beats the target hit are demoted to ``mapping_source="decoy_won"``.

Public API
----------
:func:`map_to_human`  -- entry point.
:data:`DEFAULT_ORTHOLOGY_SETTINGS` / :func:`resolve_orthology_settings`
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants + settings
# ---------------------------------------------------------------------------

_PHOSPHO_ACCEPTORS = ("S", "T", "Y")
_STY_SET = frozenset(_PHOSPHO_ACCEPTORS)

# Bundled FASTA paths (relative to package root).  human.fasta is the default
# target; other bundled FASTAs are for source-species use (mouse, rat,
# chinese_hamster, yeast, zebrafish).
_RESOURCES_FASTA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "resources" / "fastas"
_BUNDLED_HUMAN_FASTA = _RESOURCES_FASTA_DIR / "human.fasta"

# Default cache location for the human window index parquet.  Keyed by
# ``sha256(human_fasta_bytes)[:12]`` + window_size + decoy_seed.
_DEFAULT_CACHE_DIR = Path.home() / ".alphaphos" / "orthology"

_UNIPROT_HEADER_RE = re.compile(
    r"^>(?P<db>sp|tr)\|(?P<accession>[A-Z0-9\-]+)\|(?P<name>\S+)\s+"
    r"(?P<description>.*?)"
    r"(?:\s+OS=(?P<organism>[^=]+?))?"
    r"(?:\s+OX=(?P<taxid>\d+))?"
    r"(?:\s+GN=(?P<gene>\S+))?"
    r"(?:\s+PE=(?P<pe>\d+))?"
    r"(?:\s+SV=(?P<sv>\d+))?"
    r"\s*$"
)


DEFAULT_ORTHOLOGY_SETTINGS: dict[str, Any] = {
    "window_size": 7,
    "decoy_seed": 42,
    "prefer_swissprot": True,
    "require_center_sty": True,
    "kinase_sequence_col": "kinase_sequence",
    "score_metric": "identity",  # future: "blosum62"
    # Phase 2 -- fuzzy matching for the divergent tail
    "allow_fuzzy": True,
    "max_mismatches": 2,
    # Phase 2 -- per-site FDR filtering.  Sites with q-value >= fdr_threshold
    # are demoted to ``mapping_source="below_fdr"`` even if a fuzzy match was
    # found.  Pass ``None`` to keep every match regardless of q.
    "fdr_threshold": 0.01,
    # Phase 3 -- broader-window verification pass.  When set to a positive
    # int, the module extracts the ``±verify_window_size`` window from both
    # the source protein and each matched human paralog and reports the
    # per-site identity as a quality signal on ``.var``.  When multiple
    # paralogs are ambiguous at the primary window, the paralog with the
    # fewest verification-window mismatches is preferred as the primary
    # pick (over the gene-name-consistent tiebreak).  Requires a
    # ``source_fasta`` argument to ``map_to_human`` so that source protein
    # sequences can be recovered.  Pass ``None`` to disable the pass.
    "verify_window_size": None,
}

_ALLOWED_SCORE_METRICS = ("identity",)


def resolve_orthology_settings(advanced: dict[str, Any] | None) -> dict[str, Any]:
    """Merge overrides over :data:`DEFAULT_ORTHOLOGY_SETTINGS` with validation."""
    out = dict(DEFAULT_ORTHOLOGY_SETTINGS)
    if not advanced:
        return out
    unknown = set(advanced) - set(DEFAULT_ORTHOLOGY_SETTINGS)
    if unknown:
        raise ValueError(
            f"Unknown advanced keys: {sorted(unknown)}. "
            f"Allowed: {sorted(DEFAULT_ORTHOLOGY_SETTINGS)}"
        )
    out.update(advanced)
    if not isinstance(out["window_size"], int) or out["window_size"] < 1:
        raise ValueError(f"window_size must be a positive int; got {out['window_size']!r}")
    if not isinstance(out["decoy_seed"], int):
        raise ValueError(f"decoy_seed must be int; got {out['decoy_seed']!r}")
    if out["score_metric"] not in _ALLOWED_SCORE_METRICS:
        raise ValueError(
            f"score_metric must be one of {_ALLOWED_SCORE_METRICS}; got {out['score_metric']!r}"
        )
    for bool_key in ("prefer_swissprot", "require_center_sty", "allow_fuzzy"):
        if not isinstance(out[bool_key], bool):
            raise ValueError(f"{bool_key} must be bool; got {type(out[bool_key]).__name__}")
    if not isinstance(out["max_mismatches"], int) or out["max_mismatches"] < 0:
        raise ValueError(
            f"max_mismatches must be a non-negative int; got {out['max_mismatches']!r}"
        )
    # Full window length = 2*window_size + 1; enforce that the fuzzy pigeonhole
    # partition (max_mismatches + 1 equal-ish segments) is workable.
    if out["max_mismatches"] >= (2 * out["window_size"] + 1):
        raise ValueError(
            f"max_mismatches ({out['max_mismatches']}) must be < window length "
            f"({2 * out['window_size'] + 1})"
        )
    fdr = out["fdr_threshold"]
    if fdr is not None:
        if isinstance(fdr, bool) or not isinstance(fdr, (int, float)):
            raise ValueError(f"fdr_threshold must be float in [0, 1] or None; got {fdr!r}")
        if not (0.0 <= float(fdr) <= 1.0):
            raise ValueError(f"fdr_threshold must be in [0, 1]; got {fdr!r}")
    vws = out["verify_window_size"]
    if vws is not None:
        if isinstance(vws, bool) or not isinstance(vws, int) or vws < 1:
            raise ValueError(f"verify_window_size must be a positive int or None; got {vws!r}")
        if vws <= out["window_size"]:
            raise ValueError(
                f"verify_window_size ({vws}) must be strictly greater than "
                f"window_size ({out['window_size']}) -- otherwise the "
                "verification adds no information beyond the primary match."
            )
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def map_to_human(
    adata: ad.AnnData,
    *,
    human_fasta: str | Path | None = None,
    source_fasta: str | Path | None = None,
    cache_dir: str | Path | None = None,
    advanced: dict[str, Any] | None = None,
    copy: bool = False,
) -> ad.AnnData:
    """Map each site in ``adata.var`` to its best-match human ortholog site.

    Requires ``adata.var[kinase_sequence_col]`` to contain the ``±window_size``
    residue window around each site (produced upstream by
    :func:`alphaphos.add_kinase_windows`).

    Parameters
    ----------
    adata
        Site-level (or precursor-level) ``AnnData``.  ``.var`` must carry the
        kinase-sequence window column.
    human_fasta
        Path to a human proteome FASTA.  Defaults to the bundled
        ``resources/fastas/human.fasta``.
    source_fasta
        Path to the source-species FASTA (used only when
        ``advanced["verify_window_size"]`` is set -- the module needs full
        source protein sequences to compute the broader-window
        verification identity).  When omitted, the verification pass is
        silently skipped and the ``verification_*`` columns are populated
        with NaN / -1.
    cache_dir
        Where to persist the human window index parquet.  Default:
        ``~/.alphaphos/orthology/``.  Same FASTA + same settings hits cache.
    advanced
        Overrides for :data:`DEFAULT_ORTHOLOGY_SETTINGS`.
    copy
        If ``True``, mutate a fresh copy of ``adata``.  In-place otherwise.
        In either case the (possibly-mutated) AnnData is returned.

    Returns
    -------
    ``AnnData`` with these new ``.var`` columns::

        human_gene                  str | None
        human_uniprot               str | None (canonical / SwissProt-preferred)
        human_site                  str | None ("S473")
        human_site_key              str | None ("P31749_S473" -- OmniPath-consumable)
        site_conserved              bool
        ortholog_ambiguous          bool (multiple paralogs matched)
        n_paralogs                  int  (total human matches for the window)
        n_paralogs_distinct_genes   int  (distinct human genes among the matches)
        motif_promiscuous           bool (window matches > 3 distinct human genes)
        mapping_source              str  "exact_match" | "approximate" |
                                         "unmapped" | "below_fdr" | "decoy_won"
        mismatches                  int  (0 for exact, 1-max for fuzzy, -1 otherwise)
        mapping_qvalue              float (per-site target-decoy q-value; NaN
                                         if no target/decoy hit)
        verification_mismatches     int  (Hamming vs the human ±verify window,
                                         -1 when not verified)
        verification_identity       float in [0, 1] or NaN (matching fraction
                                         across the ±verify window)

    A companion paralog table lands at
    ``adata.uns["orthology"]["paralogs"]`` (one row per ``(precursor_key, paralog)``
    for ambiguous sites).  Aggregate mapping + FDR stats at
    ``adata.uns["orthology"]["stats"]``.
    """
    if ad is None:  # pragma: no cover
        raise ImportError("anndata is required for map_to_human.")

    settings = resolve_orthology_settings(advanced)
    fasta_path = Path(human_fasta) if human_fasta is not None else _BUNDLED_HUMAN_FASTA
    if not fasta_path.exists():
        raise FileNotFoundError(f"Human FASTA not found: {fasta_path}")

    cache_root = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR

    kseq_col = settings["kinase_sequence_col"]
    if kseq_col not in adata.var.columns:
        raise KeyError(
            f"adata.var is missing the {kseq_col!r} column.  Populate it upstream via "
            f"``adata = ap.add_kinase_windows(adata, fasta_path=<source_fasta>)`` "
            f"before calling map_to_human."
        )

    adata = adata.copy() if copy else adata

    # ---- 1. Build (or load) the target + decoy human window indexes ----
    target_index, decoy_index, index_meta = _load_or_build_indexes(
        fasta_path=fasta_path,
        window_size=settings["window_size"],
        decoy_seed=settings["decoy_seed"],
        cache_root=cache_root,
    )
    window_len = 2 * settings["window_size"] + 1

    # ---- 1b. Segment indexes for fuzzy lookup (built on-demand) ---------
    target_segments = None
    decoy_segments = None
    if settings["allow_fuzzy"] and settings["max_mismatches"] > 0:
        n_segments = settings["max_mismatches"] + 1
        target_segments = _build_segment_indexes(target_index, window_len, n_segments)
        decoy_segments = _build_segment_indexes(decoy_index, window_len, n_segments)

    # ---- 2. Per-site lookup: exact first, fuzzy fallback if enabled -----
    # Strip the ``_LEFT*S*RIGHT_`` ornaments that :func:`alphaphos.add_kinase_windows`
    # writes -- the flanking underscores mark protein-boundary padding and the
    # stars flank the phospho residue.  The lookup index stores raw AA windows,
    # so ornaments must be removed for identity + fuzzy matching to work.
    source_windows = (
        adata.var[kseq_col]
        .astype(str)
        .fillna("")
        .str.replace("_", "", regex=False)
        .str.replace("*", "", regex=False)
        .to_numpy()
    )
    n_sites = len(source_windows)

    # Per-site outputs
    hgene_out: list[str | None] = [None] * n_sites
    huniprot_out: list[str | None] = [None] * n_sites
    hsite_out: list[str | None] = [None] * n_sites
    hkey_out: list[str | None] = [None] * n_sites
    conserved_out = np.zeros(n_sites, dtype=bool)
    ambiguous_out = np.zeros(n_sites, dtype=bool)
    mapping_source_out: list[str] = ["unmapped"] * n_sites
    n_paralogs_out = np.zeros(n_sites, dtype=int)
    n_paralogs_distinct_genes_out = np.zeros(n_sites, dtype=int)
    motif_promiscuous_out = np.zeros(n_sites, dtype=bool)
    mismatches_out = np.full(n_sites, -1, dtype=int)  # -1 = no match
    target_mm_out = np.full(n_sites, -1, dtype=int)  # for FDR
    decoy_mm_out = np.full(n_sites, -1, dtype=int)  # for FDR
    # Verification-window mismatch counts.  -1 = not verified (no verify
    # window or edge-truncated).  Populated when verify_window_size is set.
    verification_mm_out = np.full(n_sites, -1, dtype=int)

    paralog_rows: list[dict[str, Any]] = []
    site_keys = list(adata.var.index)

    n_exact_hits = 0
    n_fuzzy_hits = 0

    max_mm = settings["max_mismatches"] if settings["allow_fuzzy"] else 0

    center_pos = settings["window_size"]  # 0-indexed center of the window
    require_class = settings["require_center_sty"]

    for i, window in enumerate(source_windows):
        if not window or len(window) != window_len:
            continue
        source_center = window[center_pos]
        # -- target lookup (with residue-class filter if enabled) --
        target_hit_window, target_mm = _match_window(
            window,
            entries=target_index,
            segment_indexes=target_segments,
            max_mismatches=max_mm,
            center_pos=center_pos,
            source_center_residue=source_center if require_class else None,
        )
        # -- decoy lookup --
        _, decoy_mm = _match_window(
            window,
            entries=decoy_index,
            segment_indexes=decoy_segments,
            max_mismatches=max_mm,
            center_pos=center_pos,
            source_center_residue=source_center if require_class else None,
        )

        target_mm_out[i] = target_mm
        decoy_mm_out[i] = decoy_mm

        if target_hit_window is not None:
            target_matches = target_index[target_hit_window]
            source_gene = _extract_source_gene(site_keys[i])
            best = _pick_canonical(
                target_matches,
                prefer_swissprot=settings["prefer_swissprot"],
                source_gene=source_gene,
            )
            hgene_out[i] = best["gene"]
            huniprot_out[i] = best["uniprot"]
            hsite_out[i] = f"{best['residue']}{best['position']}"
            hkey_out[i] = f"{best['uniprot']}_{best['residue']}{best['position']}"
            conserved_out[i] = True
            mismatches_out[i] = target_mm
            if target_mm == 0:
                mapping_source_out[i] = "exact_match"
                n_exact_hits += 1
            else:
                mapping_source_out[i] = "approximate"
                n_fuzzy_hits += 1
            if len(target_matches) > 1:
                ambiguous_out[i] = True
                n_paralogs_out[i] = len(target_matches)
                distinct_genes = {m["gene"].strip().upper() for m in target_matches}
                n_paralogs_distinct_genes_out[i] = len(distinct_genes)
                # Motif-promiscuous: same window found in >3 distinct human genes
                # (i.e. this window is a shared motif, not a specific ortholog).
                if len(distinct_genes) > 3:
                    motif_promiscuous_out[i] = True
                for m in target_matches:
                    paralog_rows.append(
                        {
                            "precursor_key": site_keys[i],
                            "human_uniprot": m["uniprot"],
                            "human_gene": m["gene"],
                            "human_site": f"{m['residue']}{m['position']}",
                            "is_reviewed": bool(m["is_reviewed"]),
                            "mismatches": int(target_mm),
                        }
                    )

    # ---- 2b. Broader-window verification pass (if enabled) --------------
    # For each mapped site, extract the ±verify_window_size window from both
    # the source protein and the matched human paralog; count mismatches.
    # When multiple paralogs are ambiguous, re-score all of them and prefer
    # the paralog with the fewest verification mismatches as the new
    # primary pick (overrides the gene-name tiebreak when they disagree).
    verify_size = settings["verify_window_size"]
    n_verified = 0
    n_verify_edge_dropped = 0
    n_verify_reassigned = 0
    if verify_size is not None and source_fasta is not None:
        source_path = Path(source_fasta)
        if not source_path.exists():
            raise FileNotFoundError(f"source_fasta not found: {source_path}")
        source_seqs = {e.uniprot: e.sequence for e in _iter_fasta_entries(source_path)}
        human_seqs = {e.uniprot: e.sequence for e in _iter_fasta_entries(fasta_path)}
        verify_len = 2 * verify_size + 1
        # Group paralog rows by precursor_key for fast lookup
        paralog_by_key: dict[str, list[dict[str, Any]]] = {}
        for row in paralog_rows:
            paralog_by_key.setdefault(row["precursor_key"], []).append(row)

        for i, sk in enumerate(site_keys):
            if not conserved_out[i]:
                continue
            # Parse source coords from the alphaPhos site key
            src_info = _parse_site_key_for_verify(sk)
            if src_info is None:
                n_verify_edge_dropped += 1
                continue
            src_uniprot, src_res, src_pos = src_info
            src_seq = source_seqs.get(src_uniprot)
            if src_seq is None:
                n_verify_edge_dropped += 1
                continue
            src_v = _extract_verify_window(src_seq, src_pos, src_res, verify_size)
            if src_v is None:
                n_verify_edge_dropped += 1
                continue

            # Primary human match
            primary_uniprot = huniprot_out[i]
            primary_site = hsite_out[i]
            if primary_uniprot is None or primary_site is None:
                n_verify_edge_dropped += 1
                continue
            primary_hum_seq = human_seqs.get(primary_uniprot)
            if primary_hum_seq is None:
                n_verify_edge_dropped += 1
                continue
            primary_res, primary_pos = primary_site[0], int(primary_site[1:])
            primary_v = _extract_verify_window(
                primary_hum_seq, primary_pos, primary_res, verify_size
            )
            if primary_v is None:
                n_verify_edge_dropped += 1
                continue

            primary_mm = _hamming(src_v, primary_v, cap=verify_len)
            best_uniprot = primary_uniprot
            best_res = primary_res
            best_pos = primary_pos
            best_mm = primary_mm

            # Re-score paralogs at ±verify (if this site is ambiguous)
            if ambiguous_out[i]:
                for prow in paralog_by_key.get(sk, ()):
                    pu = prow["human_uniprot"]
                    if pu == primary_uniprot:
                        continue
                    p_hum_seq = human_seqs.get(pu)
                    if p_hum_seq is None:
                        continue
                    p_site = prow["human_site"]
                    p_res, p_pos = p_site[0], int(p_site[1:])
                    p_v = _extract_verify_window(p_hum_seq, p_pos, p_res, verify_size)
                    if p_v is None:
                        continue
                    p_mm = _hamming(src_v, p_v, cap=verify_len)
                    if p_mm < best_mm:
                        best_mm = p_mm
                        best_uniprot = pu
                        best_res = p_res
                        best_pos = p_pos

            # If a paralog beat the primary at verify, reassign
            if best_uniprot != primary_uniprot:
                # Look up the paralog row for the new best to get its gene
                new_gene = None
                for prow in paralog_by_key.get(sk, ()):
                    if prow["human_uniprot"] == best_uniprot:
                        new_gene = prow["human_gene"]
                        break
                if new_gene is not None:
                    huniprot_out[i] = best_uniprot
                    hgene_out[i] = new_gene
                    hsite_out[i] = f"{best_res}{best_pos}"
                    hkey_out[i] = f"{best_uniprot}_{best_res}{best_pos}"
                    n_verify_reassigned += 1

            verification_mm_out[i] = best_mm
            n_verified += 1

    # ---- 3. Per-site q-values via target-decoy ranking ------------------
    q_values, is_target_win, is_decoy_win = _compute_qvalues(target_mm_out, decoy_mm_out)

    # ---- 4. Apply FDR filter: sites with q >= threshold get demoted -----
    fdr_th = settings["fdr_threshold"]
    n_below_fdr = 0
    if fdr_th is not None:
        above = conserved_out & (q_values >= float(fdr_th))
        # A site with no q-value (no target or decoy hit) is unaffected.
        above &= np.isfinite(q_values)
        n_below_fdr = int(above.sum())
        for i in np.where(above)[0]:
            hgene_out[i] = None
            huniprot_out[i] = None
            hsite_out[i] = None
            hkey_out[i] = None
            conserved_out[i] = False
            ambiguous_out[i] = False
            n_paralogs_out[i] = 0
            n_paralogs_distinct_genes_out[i] = 0
            motif_promiscuous_out[i] = False
            mapping_source_out[i] = "below_fdr"
            mismatches_out[i] = -1

    # Also demote sites where decoy wins (target hit exists but decoy hits with
    # fewer mismatches): these are spurious, not real orthologs.
    for i in np.where(is_decoy_win & conserved_out)[0]:
        hgene_out[i] = None
        huniprot_out[i] = None
        hsite_out[i] = None
        hkey_out[i] = None
        conserved_out[i] = False
        ambiguous_out[i] = False
        n_paralogs_out[i] = 0
        n_paralogs_distinct_genes_out[i] = 0
        motif_promiscuous_out[i] = False
        mapping_source_out[i] = "decoy_won"
        mismatches_out[i] = -1

    # ---- 5. Assemble .var columns + provenance --------------------------
    adata.var["human_gene"] = hgene_out
    adata.var["human_uniprot"] = huniprot_out
    adata.var["human_site"] = hsite_out
    adata.var["human_site_key"] = hkey_out
    adata.var["site_conserved"] = conserved_out
    adata.var["ortholog_ambiguous"] = ambiguous_out
    adata.var["mapping_source"] = mapping_source_out
    adata.var["n_paralogs"] = n_paralogs_out
    adata.var["n_paralogs_distinct_genes"] = n_paralogs_distinct_genes_out
    adata.var["motif_promiscuous"] = motif_promiscuous_out
    adata.var["mismatches"] = mismatches_out
    adata.var["mapping_qvalue"] = q_values
    # Verification (broader-window) annotations.  ``verification_mm=-1``
    # indicates that verification did not run for this site (either the
    # feature is disabled, the source_fasta wasn't provided, or the site
    # was edge-truncated).  ``verification_identity`` is expressed as
    # matching-residue fraction in [0, 1] and is NaN when unavailable.
    verify_size = settings["verify_window_size"]
    adata.var["verification_mismatches"] = verification_mm_out
    if verify_size is not None:
        vlen = 2 * verify_size + 1
        vid = np.where(
            verification_mm_out >= 0,
            1.0 - verification_mm_out.astype(float) / vlen,
            np.nan,
        )
    else:
        vid = np.full(n_sites, np.nan, dtype=float)
    adata.var["verification_identity"] = vid

    orthology_ns = adata.uns.get("orthology", {})
    if not isinstance(orthology_ns, dict):
        orthology_ns = {}
    orthology_ns["paralogs"] = pd.DataFrame(
        paralog_rows,
        columns=[
            "precursor_key",
            "human_uniprot",
            "human_gene",
            "human_site",
            "is_reviewed",
            "mismatches",
        ],
    )
    total_target_wins = int(is_target_win.sum())
    total_decoy_wins = int(is_decoy_win.sum())
    global_fdr = total_decoy_wins / max(total_target_wins, 1) if total_target_wins else float("nan")
    orthology_ns["stats"] = {
        "n_sites": int(n_sites),
        "n_exact_hits": int(n_exact_hits),
        "n_fuzzy_hits": int(n_fuzzy_hits),
        "n_target_wins": total_target_wins,
        "n_decoy_wins": total_decoy_wins,
        "n_ambiguous": int(ambiguous_out.sum()),
        "n_mapped_after_fdr": int(conserved_out.sum()),
        "n_below_fdr": int(n_below_fdr),
        "n_unmapped": int((~conserved_out).sum()),
        "mapping_rate": float(conserved_out.mean()) if n_sites else 0.0,
        "global_fdr_estimate": float(global_fdr),
        "fdr_threshold": settings["fdr_threshold"],
        "human_fasta": str(fasta_path),
        "window_size": settings["window_size"],
        "max_mismatches": settings["max_mismatches"] if settings["allow_fuzzy"] else 0,
        "decoy_seed": settings["decoy_seed"],
        "target_index_size": int(index_meta["n_target_windows"]),
        "decoy_index_size": int(index_meta["n_decoy_windows"]),
        "n_unique_target_windows": int(index_meta["n_unique_target_windows"]),
        # Verification-pass stats (all zero when verify_window_size is None)
        "verify_window_size": settings["verify_window_size"],
        "n_verified": int(n_verified),
        "n_verify_edge_dropped": int(n_verify_edge_dropped),
        "n_verify_reassigned": int(n_verify_reassigned),
    }
    adata.uns["orthology"] = orthology_ns

    logger.info(
        "map_to_human: %d exact + %d fuzzy = %d/%d sites mapped (%.1f%%), "
        "%d ambiguous, %d demoted below FDR<%s, global-FDR estimate = %.2e",
        n_exact_hits,
        n_fuzzy_hits,
        int(conserved_out.sum()),
        n_sites,
        100 * conserved_out.sum() / max(n_sites, 1),
        int(ambiguous_out.sum()),
        n_below_fdr,
        settings["fdr_threshold"],
        global_fdr,
    )
    if settings["verify_window_size"] is not None and source_fasta is not None:
        logger.info(
            "map_to_human: verification pass (±%d) -- %d verified, %d edge-dropped, "
            "%d paralog primary picks reassigned to a ±%d-better paralog",
            settings["verify_window_size"],
            n_verified,
            n_verify_edge_dropped,
            n_verify_reassigned,
            settings["verify_window_size"],
        )
    return adata


# ---------------------------------------------------------------------------
# Index building + caching
# ---------------------------------------------------------------------------


def _load_or_build_indexes(
    *,
    fasta_path: Path,
    window_size: int,
    decoy_seed: int,
    cache_root: Path,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, Any]]:
    """Load a cached (target, decoy) index pair if present; else build and cache."""
    fasta_hash = _hash_file(fasta_path)[:12]
    cache_dir = cache_root / f"human_w{window_size}_seed{decoy_seed}_{fasta_hash}"
    target_pq = cache_dir / "target.parquet"
    decoy_pq = cache_dir / "decoy.parquet"
    meta_pq = cache_dir / "meta.parquet"

    if target_pq.exists() and decoy_pq.exists() and meta_pq.exists():
        target_df = pd.read_parquet(target_pq)
        decoy_df = pd.read_parquet(decoy_pq)
        meta = pd.read_parquet(meta_pq).iloc[0].to_dict()
        logger.info(
            "map_to_human: loaded cached indexes from %s (%d target windows, %d decoy windows)",
            cache_dir,
            len(target_df),
            len(decoy_df),
        )
    else:
        target_df, decoy_df, meta = _build_indexes(
            fasta_path=fasta_path,
            window_size=window_size,
            decoy_seed=decoy_seed,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        target_df.to_parquet(target_pq)
        decoy_df.to_parquet(decoy_pq)
        pd.DataFrame([meta]).to_parquet(meta_pq)
        logger.info("map_to_human: wrote indexes to %s", cache_dir)

    target_index = _dataframe_to_dict(target_df)
    decoy_index = _dataframe_to_dict(decoy_df)
    return target_index, decoy_index, meta


def _dataframe_to_dict(df: pd.DataFrame) -> dict[str, list[dict]]:
    """{window: [ {uniprot, gene, position, residue, is_reviewed}, ... ]}."""
    out: dict[str, list[dict]] = {}
    for row in df.itertuples(index=False):
        entry = {
            "uniprot": row.uniprot,
            "gene": row.gene,
            "position": int(row.position),
            "residue": row.residue,
            "is_reviewed": bool(row.is_reviewed),
        }
        out.setdefault(row.window, []).append(entry)
    return out


def _build_indexes(
    *,
    fasta_path: Path,
    window_size: int,
    decoy_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Parse FASTA once; emit target + decoy window frames."""
    target_records: list[tuple[str, str, str, int, str, bool]] = []
    decoy_records: list[tuple[str, str, str, int, str, bool]] = []

    rng_master = np.random.default_rng(decoy_seed)

    for entry in _iter_fasta_entries(fasta_path):
        # Target
        for pos, residue, window in _iter_sty_windows(entry.sequence, window_size):
            target_records.append(
                (window, entry.uniprot, entry.gene, pos, residue, entry.is_reviewed)
            )
        # Decoy: shuffle non-STY residues in place, preserving STY positions
        shuffled = _shuffle_preserve_sty(entry.sequence, rng_master)
        for pos, residue, window in _iter_sty_windows(shuffled, window_size):
            decoy_records.append(
                (window, entry.uniprot, entry.gene, pos, residue, entry.is_reviewed)
            )

    columns = ["window", "uniprot", "gene", "position", "residue", "is_reviewed"]
    target_df = pd.DataFrame(target_records, columns=columns)
    decoy_df = pd.DataFrame(decoy_records, columns=columns)
    meta = {
        "n_target_windows": len(target_df),
        "n_decoy_windows": len(decoy_df),
        "n_unique_target_windows": int(target_df["window"].nunique()),
        "window_size": window_size,
        "decoy_seed": decoy_seed,
        "fasta_path": str(fasta_path),
    }
    logger.info(
        "map_to_human: built indexes — %d target windows (%d unique), %d decoy windows",
        meta["n_target_windows"],
        meta["n_unique_target_windows"],
        meta["n_decoy_windows"],
    )
    return target_df, decoy_df, meta


# ---------------------------------------------------------------------------
# FASTA parsing + windowing
# ---------------------------------------------------------------------------


class _FastaEntry:
    __slots__ = ("gene", "is_reviewed", "sequence", "uniprot")

    def __init__(self, uniprot: str, gene: str, sequence: str, is_reviewed: bool) -> None:
        self.uniprot = uniprot
        self.gene = gene
        self.sequence = sequence
        self.is_reviewed = is_reviewed


def _iter_fasta_entries(path: Path):
    """Yield ``_FastaEntry`` per UniProt-formatted entry.

    Parses standard UniProt FASTA headers ``>sp|A|B GN=X OS=... OX=...``.
    Skips entries with unparseable headers or empty sequences.
    """
    current_meta: dict[str, str] | None = None
    current_seq: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_meta is not None:
                    yield _make_entry(current_meta, "".join(current_seq))
                current_meta = _parse_uniprot_header(line)
                current_seq = []
            elif current_meta is not None:
                current_seq.append(line)
        if current_meta is not None:
            yield _make_entry(current_meta, "".join(current_seq))


def _parse_uniprot_header(header: str) -> dict[str, str] | None:
    m = _UNIPROT_HEADER_RE.match(header)
    if m is None:
        return None
    d = m.groupdict()
    d = {k: (v or "") for k, v in d.items()}
    return d


def _make_entry(meta: dict[str, str] | None, sequence: str) -> _FastaEntry | None:
    if not meta or not sequence:
        return None
    uniprot = meta.get("accession", "").strip()
    if not uniprot:
        return None
    gene = meta.get("gene", "").strip() or uniprot  # fall back to accession
    is_reviewed = meta.get("db", "") == "sp"
    return _FastaEntry(uniprot=uniprot, gene=gene, sequence=sequence, is_reviewed=is_reviewed)


def _iter_sty_windows(sequence: str, window_size: int):
    """Yield ``(1-indexed_pos, residue, window)`` for every S/T/Y in ``sequence``.

    Edges are pad-truncated: a site near the N- or C-terminus emits a window
    shorter than ``2 * window_size + 1``, and such windows are dropped (they
    would produce spurious matches).
    """
    seq_len = len(sequence)
    w = window_size
    for i, aa in enumerate(sequence):
        if aa not in _STY_SET:
            continue
        start = i - w
        end = i + w + 1
        if start < 0 or end > seq_len:
            continue
        window = sequence[start:end]
        yield (i + 1, aa, window)


def _shuffle_preserve_sty(seq: str, rng: np.random.Generator) -> str:
    """Shuffle non-S/T/Y residues in-place; keep S/T/Y positions fixed.

    Preserves phospho-acceptor density and window count relative to target.
    Deterministic given ``rng`` state; the caller advances ``rng`` between
    entries for independence.
    """
    seq_arr = np.array(list(seq))
    is_sty = np.isin(seq_arr, list(_STY_SET))
    non_sty_positions = np.where(~is_sty)[0]
    non_sty_residues = seq_arr[non_sty_positions].copy()
    rng.shuffle(non_sty_residues)
    seq_arr[non_sty_positions] = non_sty_residues
    return "".join(seq_arr.tolist())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_canonical(
    matches: list[dict[str, Any]],
    *,
    prefer_swissprot: bool,
    source_gene: str | None = None,
) -> dict[str, Any]:
    """Return the canonical match from a list of same-window human matches.

    Preference order:
      1. Gene-name-consistent (case-insensitive match to ``source_gene``, if
         provided).  This handles the paralog case where the source gene
         name unambiguously points at the correct human ortholog (e.g.
         hamster ``Actb`` &rarr; human ``ACTB``, not the alphabetically-first
         ``ACTA1``).
      2. SwissProt (reviewed) preferred over TrEMBL.
      3. Alphabetical (gene, then uniprot) as last resort.
    """
    if source_gene is not None:
        src = source_gene.strip().upper()
        if src:
            gene_hits = [m for m in matches if m["gene"].strip().upper() == src]
            if gene_hits:
                # Prefer reviewed among gene-consistent hits.
                if prefer_swissprot:
                    reviewed = [m for m in gene_hits if m["is_reviewed"]]
                    if reviewed:
                        reviewed.sort(key=lambda m: (m["gene"], m["uniprot"]))
                        return reviewed[0]
                gene_hits.sort(key=lambda m: (m["gene"], m["uniprot"]))
                return gene_hits[0]
    if prefer_swissprot:
        reviewed = [m for m in matches if m["is_reviewed"]]
        if reviewed:
            reviewed.sort(key=lambda m: (m["gene"], m["uniprot"]))
            return reviewed[0]
    return sorted(matches, key=lambda m: (m["gene"], m["uniprot"]))[0]


def _hash_file(path: Path, block_size: int = 1 << 20) -> str:
    """Stable content hash of a FASTA (for cache keying)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Source-key parsing (gene extraction for gene-name-consistent tiebreak)
# ---------------------------------------------------------------------------


# Accepts alphaPhos site key ``Protein|Gene|Site|Mult`` OR precursor key
# ``Protein|Gene|Peptide|Charge|Mods``.  Gene is always field 1.
def _extract_source_gene(site_key: str) -> str | None:
    """Extract the second pipe-delimited field (gene) from an alphaPhos key.

    Returns None if the key has fewer than 2 fields or the gene field is
    empty / whitespace / ``"nan"``.
    """
    parts = str(site_key).split("|")
    if len(parts) < 2:
        return None
    gene = parts[1].strip()
    if not gene or gene.lower() == "nan":
        return None
    return gene


# ---------------------------------------------------------------------------
# Residue-class equivalence (Ser/Thr vs Tyr)
# ---------------------------------------------------------------------------

# Reviewer-defensible convention (matches PhosphoSitePlus site-group rules):
# Ser and Thr are treated as one residue class (both hydroxyl side-chains,
# often interchangeable across mammalian orthologs).  Tyr is a separate
# class (aromatic).  A fuzzy match that "swaps" the phospho-acceptor between
# these classes is rejected.
_STY_CLASS = {"S": "ST", "T": "ST", "Y": "Y"}


def _same_residue_class(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` are in the same residue class {ST} or {Y}."""
    return _STY_CLASS.get(a) == _STY_CLASS.get(b)


# ---------------------------------------------------------------------------
# Broader-window verification helpers (Phase 3)
# ---------------------------------------------------------------------------

_ALPHAPHOS_SITE_KEY_RE = re.compile(r"^([^|]+)\|[^|]*\|([STY])(\d+)")


def _parse_site_key_for_verify(site_key: str) -> tuple[str, str, int] | None:
    """Extract ``(uniprot, residue, position)`` from an alphaPhos site key.

    Accepts ``Protein|Gene|<AA><pos>|M<mult>`` (site key) or
    ``Protein|Gene|<AA><pos>|<Charge>|Mods`` (precursor-key variants that
    still start with the S/T/Y-position triple).  For precursor keys where
    the third field is a peptide sequence rather than a site position, the
    regex correctly reports None and the site is silently skipped in
    verification (verification is only meaningful when a source position
    is known).
    """
    m = _ALPHAPHOS_SITE_KEY_RE.match(str(site_key))
    if m is None:
        return None
    uniprot = m.group(1).split(";", 1)[0].split("-", 1)[0]  # canonical isoform
    residue = m.group(2)
    position = int(m.group(3))
    return uniprot, residue, position


def _extract_verify_window(
    sequence: str, position_1based: int, residue: str, window_size: int
) -> str | None:
    """Extract the ±``window_size`` window centered on ``position_1based``.

    Returns None when edge-truncated (position too close to N-/C-terminus)
    or when the residue at the requested position doesn't match
    ``residue`` (safety check against off-by-one errors in the source key).
    """
    i = position_1based - 1
    if i < 0 or i >= len(sequence):
        return None
    if sequence[i] != residue:
        return None
    start = i - window_size
    end = i + window_size + 1
    if start < 0 or end > len(sequence):
        return None
    return sequence[start:end]


# ---------------------------------------------------------------------------
# Fuzzy matching (pigeonhole segment index + Hamming distance)
# ---------------------------------------------------------------------------


def _segment_partition(window_len: int, n_segments: int) -> tuple[list[int], list[int]]:
    """Split a window into ``n_segments`` (nearly) equal-length pieces.

    Returns ``(segment_starts, segment_lens)``.  Extra residues (when
    ``window_len`` is not divisible) are distributed to the leading segments.

    Pigeonhole invariant: if two windows of length ``window_len`` differ by
    ``<= max_mismatches`` residues, then at least one of the
    ``max_mismatches + 1`` segments must be identical between them.
    """
    base = window_len // n_segments
    extras = window_len % n_segments
    segment_lens = [base + 1] * extras + [base] * (n_segments - extras)
    segment_starts: list[int] = []
    running = 0
    for L in segment_lens:
        segment_starts.append(running)
        running += L
    return segment_starts, segment_lens


def _build_segment_indexes(
    entries: dict[str, list[dict[str, Any]]],
    window_len: int,
    n_segments: int,
) -> dict[str, Any]:
    """For each of ``n_segments`` positions, build a segment->windows map.

    Returns ``{"starts": [...], "lens": [...], "indexes": [dict, ...]}``.
    Each ``indexes[i]`` maps a segment string to the set of full-window
    strings that have that segment at position ``i``.
    """
    starts, lens = _segment_partition(window_len, n_segments)
    seg_indexes: list[dict[str, list[str]]] = [{} for _ in range(n_segments)]
    for full_window in entries:
        for i, (start, L) in enumerate(zip(starts, lens, strict=True)):
            seg = full_window[start : start + L]
            seg_indexes[i].setdefault(seg, []).append(full_window)
    return {"starts": starts, "lens": lens, "indexes": seg_indexes}


def _hamming(a: str, b: str, cap: int) -> int:
    """Count mismatching positions between two equal-length strings.

    Short-circuits once the count exceeds ``cap`` (returns ``cap + 1`` in that
    case).  Used to skip full comparisons of clearly-diverged candidates.
    """
    d = 0
    for x, y in zip(a, b, strict=True):
        if x != y:
            d += 1
            if d > cap:
                return d
    return d


def _match_window(
    source: str,
    *,
    entries: dict[str, list[dict[str, Any]]],
    segment_indexes: dict[str, Any] | None,
    max_mismatches: int,
    center_pos: int | None = None,
    source_center_residue: str | None = None,
) -> tuple[str | None, int]:
    """Find the best-matching full window in ``entries``.

    Returns ``(matched_full_window_string, n_mismatches)`` or ``(None, -1)``.

    Fast path: exact-match lookup in ``entries``.  Fallback: if
    ``segment_indexes`` is provided and ``max_mismatches > 0``, collect
    candidates via the pigeonhole segment index and Hamming-check each.

    Residue-class filter
    --------------------
    When both ``center_pos`` and ``source_center_residue`` are provided,
    fuzzy hits whose center residue is in a different residue class from
    the source are rejected.  Classes: {S, T} and {Y}.  Exact matches
    (identical windows including the center) are unaffected by this
    filter -- if the source and target center are literally the same
    residue, they're in the same class by definition.
    """
    # Exact-match fast path.  If source==entry, centers match by identity
    # (residue class check is trivially satisfied).
    if source in entries:
        return source, 0
    # Fuzzy fallback
    if segment_indexes is None or max_mismatches <= 0:
        return None, -1
    starts = segment_indexes["starts"]
    lens = segment_indexes["lens"]
    indexes = segment_indexes["indexes"]
    candidates: set[str] = set()
    for i, (start, L) in enumerate(zip(starts, lens, strict=True)):
        seg = source[start : start + L]
        found = indexes[i].get(seg)
        if found:
            candidates.update(found)
    if not candidates:
        return None, -1
    best_window: str | None = None
    best_mm = max_mismatches + 1
    for candidate in candidates:
        # Residue-class filter (reject cross-class swaps like S->Y)
        if (
            center_pos is not None
            and source_center_residue is not None
            and not _same_residue_class(candidate[center_pos], source_center_residue)
        ):
            continue
        mm = _hamming(source, candidate, cap=best_mm - 1)
        if mm < best_mm:
            best_mm = mm
            best_window = candidate
            if mm == 0:
                break
    if best_window is not None and best_mm <= max_mismatches:
        return best_window, best_mm
    return None, -1


# ---------------------------------------------------------------------------
# Target-decoy per-site q-values
# ---------------------------------------------------------------------------


def _compute_qvalues(
    target_mm: np.ndarray,
    decoy_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target-decoy per-site FDR/q-value estimate.

    Each site has a best-target-hit mismatch count (``-1`` if none) and a
    best-decoy-hit mismatch count.  We derive:

    - ``is_target_win[i]`` -- site's target hit is better than its decoy hit
      (fewer mismatches, or a target hit with no decoy hit at all).
    - ``is_decoy_win[i]`` -- decoy strictly better than target.  These are
      spurious "target" hits; they get demoted from the output.
    - ``q_values[i]`` -- proteomics-style q-value: for each rank in the
      score-sorted list, ``q = n_cumulative_decoy_wins / n_cumulative_target_wins``,
      then take the running-min from the bottom up so q is monotone
      non-decreasing walking down the ranks.

    Ties are broken in favour of target wins (paralog ambiguity is real).

    Returns ``(q_values, is_target_win, is_decoy_win)``.  Sites with no
    hit at all have ``q_values[i] = NaN``.
    """
    n = len(target_mm)
    has_target = target_mm >= 0
    has_decoy = decoy_mm >= 0

    is_target_win = has_target & (~has_decoy | (target_mm <= decoy_mm))
    is_decoy_win = has_decoy & has_target & (decoy_mm < target_mm)
    # Decoy-only hits are also decoy wins.
    is_decoy_win |= has_decoy & ~has_target

    # Score = -mm (higher = better).  Sites with no target win get score = -inf
    # so they sort to the bottom; sites with no target hit but a decoy hit still
    # participate in the ranking (they are decoy wins that push q upward).
    scores = np.full(n, -np.inf, dtype=float)
    scores[is_target_win] = -target_mm[is_target_win].astype(float)
    scores[is_decoy_win & ~has_target] = -decoy_mm[is_decoy_win & ~has_target].astype(float)
    # For decoy-wins-with-target-hit, use decoy score (they're competing at the
    # decoy's better level).
    dw_with_target = is_decoy_win & has_target
    scores[dw_with_target] = -decoy_mm[dw_with_target].astype(float)

    q_values = np.full(n, np.nan, dtype=float)
    participating = is_target_win | is_decoy_win
    if not participating.any():
        return q_values, is_target_win, is_decoy_win

    part_idx = np.where(participating)[0]
    order = part_idx[np.argsort(-scores[part_idx], kind="stable")]

    n_target_cum = 0
    n_decoy_cum = 0
    q_per_rank = np.zeros(len(order), dtype=float)
    for rank, idx in enumerate(order):
        if is_target_win[idx]:
            n_target_cum += 1
        if is_decoy_win[idx]:
            n_decoy_cum += 1
        q_per_rank[rank] = n_decoy_cum / max(n_target_cum, 1)

    # Monotone envelope from the bottom up: q_i = min over j >= i of raw_q_j.
    q_monotone = np.minimum.accumulate(q_per_rank[::-1])[::-1]
    for rank, idx in enumerate(order):
        q_values[idx] = q_monotone[rank]

    return q_values, is_target_win, is_decoy_win
