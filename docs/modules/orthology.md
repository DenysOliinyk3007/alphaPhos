# `alphaphos.orthology`

Cross-species phosphosite ortholog mapping to human. Motivated by non-human
phosphoproteomics data (CHO, mouse, rat) where downstream tooling (KSEA,
OmniPath, PhosphoSitePlus, Reactome) is human-centric and cannot run without
site-level ortholog assignment.

## Public API

`ap.orthology.map_to_human(adata, human_fasta=None, cache_dir=None, advanced=None, copy=False)`
&rarr; returns the `AnnData` with per-site human ortholog annotation columns
on `.var`, a paralog side-table at `adata.uns["orthology"]["paralogs"]`, and
mapping statistics at `adata.uns["orthology"]["stats"]`.

**Prerequisite**: `adata.var["kinase_sequence"]` must be populated (by
`ap.add_kinase_windows(adata, fasta_path=<source_fasta>)`).

## Method

**Window-based direct search** — for each source-organism phosphosite, look
up its `±window_size` sequence context against a pre-built index of the
human phospho-acceptor windows.

| Step | What we do |
| --- | --- |
| 1 | Parse the human FASTA. For every S/T/Y position, extract the `±window_size` window and store `(window, uniprot, gene, position, residue, is_reviewed)` in a dict indexed by window string. |
| 2 | Build a **decoy index** the same way from a per-protein shuffle of the human proteome that preserves S/T/Y positions (same length, same phospho-acceptor density, but scrambled AA composition). |
| 3 | For each source site (row of `adata.var` with a populated `kinase_sequence` column), do an **exact-match lookup** in the target and decoy dicts. |
| 4 | If no exact match and `allow_fuzzy=True`, do a **pigeonhole segment fuzzy lookup**: partition the window into `max_mismatches + 1` (nearly) equal segments; find candidates whose segment matches exactly; Hamming-check each. Enforces the **residue class filter** (S/T equivalence class vs Y separate class). |
| 5 | Rank all source sites by their best-target vs best-decoy match score, compute per-site q-values in the proteomics-style target-decoy way. Sites with q &ge; `fdr_threshold` are demoted to `mapping_source="below_fdr"`. Sites where the decoy hit is better than the target hit are demoted to `mapping_source="decoy_won"`. |

## Community precedent

The core principle — **flanking sequence conservation defines orthologous
site identity** — is what PhosphoSitePlus uses for its cross-species "site
groups" (Hornbeck et al. 2012 *Nucleic Acids Res* 40:D261-D270; also in
follow-up PSP papers). iPTMnet (Huang et al. 2018 *Nucleic Acids Res*
46:D542) uses the same idea. Ochoa et al. 2020 (*Nature Biotech*) and
Beltrao et al. 2012 (*Cell* 150:413) apply per-position conservation
scores across many species via MSA — same underlying principle at a
different resolution.

We adapt this to a FOSS-reproducible pipeline built on the bundled human
FASTA, and we add a proteomics-style target-decoy FDR (Elias & Gygi 2007
*Nat Methods* 4:207 adapted to sequence-window matching) so users can
report a controlled false-positive rate.

## When window-based ortholog mapping is defensible

**Well-supported by literature**:

- Mammalian source (mouse, rat, hamster) → human: flanking sequence
  conservation is high (typically 90+% identity in orthologous regions),
  background collision rate in a 15-residue window against a 20k-protein
  proteome is negligible.
- The ±7 window is the length PSP uses for its site groups.
- Well-conserved sites (kinase activation loops, receptor
  autophosphorylation, mitotic sites) have 100% conservation across
  mammals — canonical cases.

**Weaker regime**:

- Deep-divergent species (fungi, protists → human): flanking sequence has
  diverged too much for window-only matching. Fall back to protein-level
  ortholog identification (Ensembl Compara, OrthoDB, BLAST-RBH) + MSA.
- Rapidly-evolving regions (disordered TFs, immune receptors): window may
  differ even for true orthologs.
- Highly duplicated protein families: window match is genuine but paralog
  ambiguous — see below.

## `.var` output columns

| column | type | meaning |
| --- | --- | --- |
| `human_gene` | str \| None | Human gene symbol. `None` if unmapped / below FDR / decoy won. |
| `human_uniprot` | str \| None | Human UniProt accession (canonical / SwissProt-preferred). |
| `human_site` | str \| None | e.g. `"S473"`. |
| `human_site_key` | str \| None | `<uniprot>_<site>` — OmniPath-consumable format. |
| `site_conserved` | bool | Set once mapping survives FDR filter. |
| `ortholog_ambiguous` | bool | Multiple human paralogs matched. |
| `n_paralogs` | int | Total matches (including the picked primary). |
| `n_paralogs_distinct_genes` | int | Distinct human genes among the matches. |
| `motif_promiscuous` | bool | `True` if `n_paralogs_distinct_genes > 3` (window is a shared motif, not a specific ortholog). |
| `mapping_source` | str | `"exact_match"` / `"approximate"` / `"unmapped"` / `"below_fdr"` / `"decoy_won"`. |
| `mismatches` | int | 0 for exact, 1-2 for fuzzy, -1 for anything unmapped. |
| `mapping_qvalue` | float | Proteomics-style q-value (NaN if no target/decoy hit). |

Paralog side-table at `adata.uns["orthology"]["paralogs"]` lists all human
matches per source precursor for the ambiguous cases. Aggregate statistics
at `adata.uns["orthology"]["stats"]`.

## Design choices for reviewer-defensibility

Each has been stress-tested (see [validation](#validation) below).

### 1. Window IS the identity check

We skip protein-level ortholog identification and rely on the `±7` window
match to establish both orthology and site position in one step. This is
what PSP does for its site-group tables. Reviewer objection to expect:
*"why not protein-level ortholog first?"*. Our answer: a 15-residue window
inside a 20,000-protein target proteome has negligible background
collision rate on random data (see validation B4 below); the window
match is itself strong evidence of orthology at that locus.

### 2. Target-decoy FDR via shuffle-preserve-STY

Decoy sequences are per-protein shuffles that keep every S/T/Y position
fixed while scrambling the non-STY residues. This preserves phospho-
acceptor density and window count exactly relative to target, while
destroying all real evolutionary structure. The Elias-Gygi 2007
target-decoy framework then applies unchanged.

### 3. Gene-name-consistent tiebreak

When multiple human paralogs match a source window (e.g. hamster `Actb`
&rarr; ACTA1/2/B/G1/G2/… all share the actin activation-loop window), the
primary pick prefers the human paralog whose gene symbol matches the
source (case-insensitive). Only when no source-gene match exists do we
fall back to SwissProt-preferred alphabetical. This handles the "hamster
`Actb` should map to `ACTB`, not `ACTA1`" case.

### 4. Residue-class enforcement

Fuzzy matches must keep the central phospho-acceptor in the **same residue
class**: `{S, T}` (hydroxyl side-chains) OR `{Y}` (aromatic). An S/T
&harr; Y swap is rejected even if the flanking Hamming distance is within
`max_mismatches`. This is the PSP convention. Disable via
`require_center_sty=False` (matches the old, more-permissive behaviour).

### 5. Motif-promiscuity flag

When the same window appears in more than 3 distinct human genes,
`motif_promiscuous=True` is set. These are cases where the window is a
short conserved motif (kinase substrate consensus, SH3-binding, PDZ-binding)
that appears in many unrelated proteins by parallel evolution or
domain-shuffling, not orthology. Users should treat these with caution.

### 6. Broader-window verification (`verify_window_size`)

Optional Phase-3 feature.  When ``verify_window_size`` is set (typically
15-30) *and* a ``source_fasta`` is provided, the module runs a second
pass over the primary mappings:

1. For each mapped site, extract the ``±verify_window_size`` window from
   both the source protein and the matched human paralog; compute the
   Hamming distance.  Emit as ``.var["verification_mismatches"]`` and
   ``.var["verification_identity"]`` (matching fraction in [0, 1]).
2. When a site is paralog-ambiguous, re-score *all* paralogs at
   ``±verify_window_size`` and prefer the paralog with the fewest
   verification-window mismatches as the primary pick.  This overrides
   the gene-name-consistent tiebreak when a different paralog aligns
   better across the broader flank.

**Benefits (measured on 946k mapped mouse sites, ``±30`` verification):**

- **Primary pick confirmed on 73.3%** of ambiguous cases (the gene-name
  tiebreak was independently right).
- **Tied on 19.6%** (broader flank doesn't discriminate).
- **Reassigned on 7.1%** (broader flank prefers a different paralog).
  These reassignments correct edge cases where gene-name tiebreak was
  suboptimal -- for mouse, this is ~3,000 sites (~0.3% of all mapped).
- **±30 identity distribution across mapped sites**: 20% at 100% identity,
  45% at ≥97%, 81% at ≥87%.  Users can filter on ``verification_identity``
  post-hoc as a quality signal.

**Coverage cost is zero** -- the verification annotation is emitted for
every mapped site; users choose whether to filter on it.

Not enabled by default (backwards-compatible).  Turn on with
``advanced={"verify_window_size": 30}`` and pass ``source_fasta`` to
``map_to_human``.

## Validation

Stress-tested on the bundled FASTA panel (`scripts/orthology_audit.py` is the
runnable version — reproduce any of these numbers). All rates below use
`fdr_threshold=None` to show raw counts.

### Self-mapping sanity (human &rarr; human)

12,937 human sites &rarr; **100% mapped exactly, zero decoys**. Every input
window is by definition present in the target index. ✅ Fundamental
correctness.

### Random-sequence null

1,000 random ±7 windows built from a 17-AA alphabet with an STY center:
**0 matches at any `max_mismatches` from 0 to 3**. ✅ No spurious matches
on genuinely random data.

### Hamster sensitivity curve

500 hamster proteins &rarr; 54,293 sites:

| `max_mismatches` | mapping rate | fuzzy | decoy wins | global FDR |
| --- | --- | --- | --- | --- |
| 0 | 40.8% | 0 | 0 | 0 |
| 1 | 59.6% | 10,227 | 0 | 0 |
| **2 (default)** | **70.7%** | **16,269** | **0** | **0** |
| 3 | 77.0% | 19,699 | 2 | 5&times;10⁻⁵ |
| 4 | 81.2% | 21,959 | 19 | 4&times;10⁻⁴ |

The decoy machinery is discriminating: as `max_mismatches` grows, both
target and decoy hits grow — but target grows much faster. FDR is 0 at
default (H≤2). Legitimizes the H≤2 default.

### Cross-species mapping rate curve

500 proteins from each source species, `max_mismatches=2`:

| source | evol. distance | mapped | rate |
| --- | --- | --- | --- |
| human (self) | 0 | 12,937 / 12,937 | **100.0%** |
| hamster | ~90 Mya | 38,394 / 54,293 | **70.7%** |
| zebrafish | ~450 Mya | 14,054 / 57,447 | **24.5%** |
| yeast | ~1 Bya | 1,822 / 46,577 | **3.9%** |

The method **degrades gracefully with evolutionary distance**. Yeast being
low is a feature, not a bug — it correctly rejects the vast majority of
yeast sites while retaining deep-homology cases (ribosomal, glycolytic,
CDK-family) where flanking sequence really is preserved.

### Strict species-entrapment FDR (bacterial + archaeal)

To measure a true empirical FDR — independent of the internal
shuffle-preserve-STY decoy — we run the module against **complete
bacterial and archaeal proteomes**, whose sites have essentially no
real phospho-orthologs in the human proteome.  Any hit is unambiguously
a false positive.

Full proteomes used (all SwissProt-reviewed):

| Source | Kingdom | n_sites | Mapping rate at `max_mm=2` | Empirical FDR |
| --- | --- | --- | ---:| ---:|
| *M. jannaschii* | archaeon (thermophile) | 62,932 | **0.151%** | 1.2&times;10⁻⁴ |
| *S. solfataricus* | archaeon | 22,184 | 0.334% | 9.5&times;10⁻⁵ |
| *H. salinarum* | archaeon (halophile) | 20,682 | 0.353% | 9.4&times;10⁻⁵ |
| *E. coli* K-12 | bacterium | 181,350 | 0.283% | 6.6&times;10⁻⁴ |
| *B. subtilis* | bacterium | 178,754 | 0.223% | 5.1&times;10⁻⁴ |
| *M. tuberculosis* | bacterium | 104,924 | 0.294% | 4.0&times;10⁻⁴ |
| *S. aureus* | bacterium | 42,970 | 0.358% | 2.0&times;10⁻⁴ |
| **hamster (positive control)** | mammal | **1,401,396** | **55.5%** | (denominator) |
| Internal target-decoy FDR (hamster) | — | — | — | 4.9&times;10⁻⁵ |

**Interpretation:**

- **Signal-to-noise: ~150–370&times;** real biology (55%) vs random
  cross-species collisions (0.15–0.36%).
- **Archaea < bacteria** for entrapment rate — expected because bacteria
  share more ancient housekeeping (glycolysis, translation, ancient
  kinases) with eukaryotes than archaea do.
- *E. coli* has the highest empirical FDR (6.6&times;10⁻⁴) — biggest
  proteome, most opportunities for random collisions, also shares more
  housekeeping.  Still under 10⁻³ absolute.

### FDR calibration caveat (honest)

The **internal shuffle-preserve-STY decoy FDR** (≈5&times;10⁻⁵ at H=2)
is systematically **2-13&times; lower** than the empirical
species-entrapment FDR (9.4&times;10⁻⁵ to 6.6&times;10⁻⁴).  Both are
essentially zero for practical purposes — well below any conventional
1% or 5% cutoff — but reviewers should know the internal FDR is a fast
approximation that slightly underestimates the true false-positive rate.

**Why the discrepancy**: our decoy is per-protein shuffled *human*
sequence.  Bacterial / archaeal sequences have different amino-acid
composition and different segment statistics, so they hit real target
windows more often than they hit the shuffled-human decoy windows.
Species-entrapment catches this composition-drift effect; the internal
decoy does not.

**Recommended practice for manuscripts**: cite both numbers.  Use the
internal T-D FDR as the reported per-run estimate; run species
entrapment as a validation check (small proteomes take <1 min) to
document the true empirical FDR.

### Gold-standard iron-law regression tests

32 canonical mammalian phospho sites — activation loops, receptor
autophosphorylation, cell-cycle regulators, translation control — derived
from bundled mouse and rat FASTAs. `tests/unit/test_orthology.py::TestGoldStandardSites`
requires:

- All 26 exact-match sites map to expected human ortholog (accepting
  paralog side-table entries for genuinely ambiguous cases like
  Mapk1/Mapk3 sharing an activation-loop window).
- All 4 fuzzy-tier (1-2 mm) sites map when fuzzy is on.
- 3+ mismatch sites are correctly held below the default max_mismatches=2 cap.
- Overall recall ≥ 85% on mappable (mm ≤ 2) sites.

## Methodological caveats

Six things reviewers should know we thought about:

1. **We skip protein-level ortholog verification.** Traditional methods
   (RBH, Ensembl Compara) verify protein orthology *first*, then transfer
   sites via MSA. We rely on the window match itself as evidence.
   Justified by the low collision rate empirically (see validation), and
   by the PSP precedent.
2. **Highly-conserved motifs across unrelated proteins are flagged
   `motif_promiscuous=True`.** Users should treat these with caution.
3. **Divergent species mapping rate is genuinely low** — 4% for yeast, 25%
   for zebrafish. This is correct behaviour; window-based methods trade
   sensitivity for specificity at deep divergence.
4. **Decoy strategy is shuffle-preserve-STY.** More conservative than
   plain shuffle (preserves phospho-acceptor density, so decoy window
   count = target window count).
5. **Paralog ambiguity is real biology, not error.** Actin, tubulin,
   ubiquitin, kinase families genuinely share sites. All matches are
   returned in the paralog side-table.
6. **We recommend reporting mapped-vs-unmapped counts** in Methods
   sections. Silent-drop behaviour is disabled; every source site keeps
   a row in the output with its `mapping_source` clearly labelled.

## Example

```python
import alphaphos as ap

# Prerequisite: source-species kinase windows attached to .var
adata = ap.add_kinase_windows(adata, fasta_path=source_fasta)

# Default settings: window_size=7, max_mismatches=2, fdr_threshold=0.01
adata = ap.orthology.map_to_human(adata)

# With broader-window verification (recommended for manuscript figures):
# pass source_fasta and enable verify_window_size.  Emits
# verification_identity/mismatches on .var; refines paralog primary picks.
adata = ap.orthology.map_to_human(
    adata,
    source_fasta=source_fasta,
    advanced={"verify_window_size": 30},
)

# Use human-key annotations for downstream KSEA / pathway analysis
mapped = adata[:, adata.var["site_conserved"]]
ksea = ap.enrichment.kinase_activity(
    diff_result[mapped.var_names].set_index(mapped.var["human_site_key"]),
    stat_col="log2fc",
    network="omnipath",
)

# Inspect paralog cases + promiscuous motifs before reporting
ambig = adata.var[adata.var["ortholog_ambiguous"]]
promiscuous = adata.var[adata.var["motif_promiscuous"]]
paralog_table = adata.uns["orthology"]["paralogs"]

# Methodology / Methods-section stats
stats = adata.uns["orthology"]["stats"]
print(f"Mapped {stats['n_mapped_after_fdr']:,} / {stats['n_sites']:,} sites")
print(f"Global FDR estimate: {stats['global_fdr_estimate']:.2e}")
```

## References

- Hornbeck P.V. et al. 2012. *PhosphoSitePlus: a comprehensive resource for
  investigating the structure and function of experimentally determined
  post-translational modifications in man and mouse.* Nucleic Acids Res
  40:D261-D270.
- Huang H. et al. 2018. *iPTMnet: an integrated resource for protein
  post-translational modification network discovery.* Nucleic Acids Res
  46:D542-D550.
- Ochoa D. et al. 2020. *The functional landscape of the human
  phosphoproteome.* Nature Biotechnology 38:365-373.
- Beltrao P. et al. 2012. *Systematic functional prioritization of protein
  posttranslational modifications.* Cell 150:413-425.
- Elias J.E. & Gygi S.P. 2007. *Target-decoy search strategy for increased
  confidence in large-scale protein identifications by mass spectrometry.*
  Nature Methods 4:207-214.
