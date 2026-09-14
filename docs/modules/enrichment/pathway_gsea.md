# `alphaphos.enrichment.pathway_gsea`

Gene-level **preranked GSEA** on per-site limma output -- the rank-based sibling of
[`pathway_enrichment`](pathway.md).

`pathway_gsea(diff_exp_result, ...)` reads the full log2FC-ranked list of collapsed
per-gene values (no threshold, no significance call), and asks which biological pathways
are coherently up- or down-regulated in that ranking.

## Scientific basis

Classical GSEA (Subramanian et al. 2005 *PNAS* 102:15545-15550): a running-sum enrichment
score walks the ranked gene list, adding weight when a term member is encountered and
subtracting when not; the peak deviation is the raw ES. The normalised ES (NES) accounts
for term-size differences. FDR is empirical, from set-membership permutations.
Implemented via [`gseapy.prerank`](https://gseapy.readthedocs.io/) (Fang et al. 2023).

The phospho-specific wrinkle is that GSEA needs **one number per gene**, but a phospho
experiment produces many sites per gene with different log2FCs. See
[`site_to_gene_agg`](#site_to_gene_agg-in-detail) below.

Complements ORA: `pathway_enrichment` (threshold-based) picks up **concentrated** signals
(strong hits in a small pathway); `pathway_gsea` picks up **coherent** signals (many
sub-threshold moves in the same direction across a pathway).

## Signature

```python
ap.enrichment.pathway_gsea(
    diff_exp_result: pd.DataFrame,
    *,
    stat_col: str = "log2fc",
    libraries: list[str] | None = None,
    site_to_gene_agg: Literal["max_abs", "top_significant"] = "max_abs",
    min_set_size: int = 15,
    max_set_size: int = 500,
    n_permutations: int = 1000,
    seed: int = 42,
    organism: Literal["human", "mouse"] = "human",
    key_column: str | None = None,
    fdr_col: str = "fdr",
    cache_dir: str | Path | None = None,
    threads: int = 1,
) -> pd.DataFrame
```

## Input

A limma `diff_exp_result` DataFrame with alphaPhos site keys `Protein|Gene|Site|Mult` in
the index (or `key_column`). Must carry the `stat_col` column. The `fdr_col` is only
consulted when `site_to_gene_agg="top_significant"`.

## Parameters

| Parameter | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `diff_exp_result` | *required* | Per-site limma output (indexed by alphaPhos keys). | -- |
| `stat_col` | `"log2fc"` | Column used as the ranking metric. | `"t_stat"` (moderated t) -- accounts for per-site variability. |
| `libraries` | `None` &rarr; defaults | Enrichr library slugs to query. | Same defaults as `pathway_enrichment`. Any Enrichr slug. |
| `site_to_gene_agg` | `"max_abs"` | How to collapse sites &rarr; genes. | `"top_significant"`. See below. |
| `min_set_size` | `15` | Skip terms with fewer members in the ranked universe. | gseapy prerank default. Lower to include micro-pathways. |
| `max_set_size` | `500` | Skip terms larger than this. | gseapy prerank default; caps very-broad *gene* sets (the site-level `ap.enrichment.gsea` has no cap by default). |
| `n_permutations` | `1000` | Set-membership permutations for the empirical null. | Raise to `10000` for FDR resolution below 0.001. |
| `seed` | `42` | Reproducibility seed for the permutation sample. | Any int. |
| `organism` | `"human"` | Selects the default library set. | `"mouse"`. |
| `key_column` | `None` | Column carrying site keys if not the index. | Any column name. |
| `fdr_col` | `"fdr"` | Per-site FDR column. | Only used by `site_to_gene_agg="top_significant"`. |
| `cache_dir` | `None` | Where gseapy caches downloaded GMTs (`outdir`). | Any path. |
| `threads` | `1` | Worker threads for the permutation loop. | Raise for large runs; 1 is deterministic on the same seed. |

### `site_to_gene_agg` in detail

| Value | Rule | Trade-off |
| --- | --- | --- |
| `"max_abs"` (default) | Per gene, keep the site with the largest `\|stat\|` and retain its **signed** value. | Captures peak regulation. Ignores contradicting sites on the same gene. |
| `"top_significant"` | Per gene, keep the site with the lowest per-site FDR and retain its signed stat. | Prioritises statistical evidence over effect magnitude. Requires the `fdr_col`. |

`max_abs` is the pragmatic default and matches what phospho-focused tools like PhosR do
by convention. Ties are broken by first occurrence.

## No `background` parameter

Unlike ORA, GSEA's null is built by permuting **set membership** -- the ranked list
itself is the universe. Phospho-only nature is implicit in whichever genes end up ranked
from `diff_exp_result`. No separate background argument is meaningful here.

## Output

A `pd.DataFrame`, one row per (library, term) pair that met `min_set_size`, sorted by
`fdr` ascending within each library:

| Column | Type | Meaning |
| --- | --- | --- |
| `library` | str | Enrichr library slug this term came from. |
| `term` | str | Pathway / term name. |
| `es` | float | Raw enrichment score (peak deviation of the running-sum). |
| `nes` | float | Size-normalised ES -- comparable across terms and libraries. Sign encodes direction. |
| `p_value` | float | Nominal p from the permutation null. |
| `fdr` | float | BH-adjusted q within this library. |
| `n_set` | int | Number of ranked genes in the term (denominator of gseapy's `Tag %`). |
| `n_leading_edge` | int | Number of those in the leading edge (numerator of `Tag %`). |
| `leading_edge` | str | Semicolon-joined gene symbols driving the enrichment. |
| `direction` | str | `"up"` if `nes >= 0` else `"down"`. |

`result.attrs["provenance"]` carries: `method` (`"gsea_prerank"`), `libraries`,
`stat_col`, `site_to_gene_agg`, `n_permutations`, `n_ranked_genes`, `n_sites_used`,
`n_sites_dropped`, `n_unparseable_keys`, `min_set_size`, `max_set_size`, `organism`,
`seed`, `gseapy_version`.

## Example

```python
import alphaphos as ap

result = ap.diff_exp_limma(adata, condition_column="condition", comparison=("EGF", "ctrl"))

# Default: max_abs collapse, 1000 permutations, all default libraries
gsea = ap.enrichment.pathway_gsea(result)

# Top up-regulated KEGG pathways
kegg_up = (
    gsea[(gsea["library"] == "KEGG_2021_Human") & (gsea["direction"] == "up")]
    .sort_values("fdr").head(10)
)
print(kegg_up[["term", "nes", "fdr", "size", "leading_edge"]])

# Alternative: rank by moderated t-stat, top-significant collapse, more permutations
gsea = ap.enrichment.pathway_gsea(
    result,
    stat_col="t_stat",
    site_to_gene_agg="top_significant",
    n_permutations=10000,
    seed=42,
)
```

## Per-library isolation

Libraries are queried one at a time. If a single library errors (network 404, malformed
GMT), it is logged as a warning and skipped -- the rest of the run continues. Missing
libraries do not surface in the output; check the log if you expected a library that is
absent.

## FDR control

BH-adjusted **within each library** across all terms passing `min_set_size`. Cross-library
comparisons are not jointly controlled; report terms alongside their library.
`n_permutations=1000` gives FDR resolution to ~0.001; raise for tighter bounds. Note that
permutation FDR is empirical -- a nominally-zero p from a 1000-permutation run means
"no permutations exceeded the observed ES", not "true p = 0".

## Related

- [`pathway_enrichment`](pathway.md) -- the threshold-based ORA sibling. Run both and
  compare: pathways that surface in both are strongly supported; pathways in ORA only
  suggest a concentrated hit signal; pathways in GSEA only suggest coherent sub-threshold
  motion.
- [`kinase_activity`](ksea.md) -- site-level regulator inference. Complementary
  interpretation ("who is driving this?") vs GSEA's ("which pathways are moving?").

## References

- Subramanian et al. 2005. *Gene set enrichment analysis: a knowledge-based approach for
  interpreting genome-wide expression profiles.* PNAS 102:15545-15550.
- Korotkevich et al. 2019. *Fast gene set enrichment analysis.* bioRxiv 060012. (fgsea --
  the efficient implementation gseapy uses.)
- Fang, Liu, Peltz 2023. *GSEApy: a comprehensive package for performing gene set
  enrichment analysis in Python.* Bioinformatics 39:btac757.
