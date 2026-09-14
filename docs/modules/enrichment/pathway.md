# `alphaphos.enrichment.pathway`

Gene-level pathway ORA (over-representation analysis) on significant phosphosites.

`pathway_enrichment(diff_exp_result, ...)` splits significantly up- and down-regulated
sites, extracts their gene symbols, and runs hypergeometric enrichment against Enrichr
gene-set libraries (GO BP/MF/CC + KEGG + Reactome + MSigDB Hallmark by default).

## Scientific basis

Wraps [`gseapy.enrichr`](https://gseapy.readthedocs.io/) (Fang et al. 2023) which itself
queries the Enrichr web API (Chen et al. 2013; Kuleshov et al. 2016). ORA is Fisher's
exact test on a 2x2 contingency table of hits vs non-hits × in-set vs out-of-set. FDR is
BH-adjusted within each library×direction slice, matching Enrichr's convention.

The **background** is the scientifically critical piece: for a phospho experiment, using
a genome-wide background overstates enrichment because your detection universe is much
smaller than the genome. Defaults enforce a phosphoproteome-restricted background
(all sites tested in `diff_exp_result`), and a proteome background is preferred whenever
one is available.

## Signature

```python
ap.enrichment.pathway_enrichment(
    diff_exp_result: pd.DataFrame,
    *,
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = 0.0,
    direction: Literal["up", "down", "both", "split"] = "split",
    libraries: list[str] | None = None,
    background: Literal["phosphoproteome", "genome"] | list[str] | pd.DataFrame = "phosphoproteome",
    organism: Literal["human", "mouse"] = "human",
    key_column: str | None = None,
    cache_dir: str | Path | None = None,
    stat_col: str = "log2fc",
    fdr_col: str = "fdr",
) -> pd.DataFrame
```

## Input

A limma `diff_exp_result` DataFrame with alphaPhos site keys `Protein|Gene|Site|Mult`
as the index (or in `key_column`). Must carry the `log2fc` and `fdr` columns (or the
columns named by `stat_col` / `fdr_col`). Gene symbols are extracted from each key's
Gene field.

## Parameters

| Parameter | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `diff_exp_result` | *required* | Per-site limma output (indexed by alphaPhos keys). | -- |
| `fdr_threshold` | `0.05` | Per-site FDR cutoff for calling a hit. | Any float in (0, 1]. `1.0` means "already pre-filtered". |
| `log2fc_threshold` | `0.0` | Minimum `\|log2fc\|` for a hit. | Raise to demand larger effect sizes (e.g. `1.0` for 2-fold). |
| `direction` | `"split"` | How to partition hits. | See below. |
| `libraries` | `None` &rarr; defaults | Enrichr library slugs to query. | Any of the Enrichr catalog; see below. |
| `background` | `"phosphoproteome"` | Universe used by the hypergeometric test. | See below -- the scientifically critical choice. |
| `organism` | `"human"` | Selects default libraries + Enrichr backend. | `"mouse"`. |
| `key_column` | `None` | Column carrying site keys if not the index. | Any column name. |
| `cache_dir` | `None` | Where gseapy caches Enrichr GMTs. | Any path. Default uses gseapy's own cache. |
| `stat_col` | `"log2fc"` | Which column carries the signed effect. | Any column name. |
| `fdr_col` | `"fdr"` | Which column carries the per-site FDR. | Any column name. |

### `direction` in detail

| Value | Behaviour |
| --- | --- |
| `"split"` (default) | Run up- and down-regulated sets **separately** and stack the results (adds a `direction` column). Preserves the biologically-meaningful up-vs-down distinction. |
| `"up"` | Only the up-regulated set (`log2fc > log2fc_threshold`). |
| `"down"` | Only the down-regulated set (`log2fc < -log2fc_threshold`). |
| `"both"` | Combine up + down into one gene list. Loses sign information; use only when the question is direction-agnostic. |

### `background` priority (as designed)

| Input | Effect | When to use |
| --- | --- | --- |
| `"phosphoproteome"` (default) | Every parseable gene in `diff_exp_result.index`. | Default -- when no proteome data is available. |
| `list[str]` / `pd.DataFrame` with `gene` column | Caller-supplied genes. | **Preferred** -- pass proteome-identified genes here when you have them. |
| `"genome"` | Enrichr's whole-genome default. Emits `UserWarning`. | Deliberate opt-in only; overstates enrichment for phospho data. |

### Default libraries

| Human | Mouse |
| --- | --- |
| `GO_Biological_Process_2023` | `GO_Biological_Process_2023` |
| `GO_Molecular_Function_2023` | `GO_Molecular_Function_2023` |
| `GO_Cellular_Component_2023` | `GO_Cellular_Component_2023` |
| `KEGG_2021_Human` | `KEGG_2019_Mouse` |
| `Reactome_2022` | `Reactome_2022` |
| `MSigDB_Hallmark_2020` | `MSigDB_Hallmark_2020` |

Pass any list of Enrichr slugs via `libraries=[...]`; see
[Enrichr's library page](https://maayanlab.cloud/Enrichr/#libraries) for the full catalog.

## Output

A `pd.DataFrame`, sorted by `fdr` ascending within each `direction` + `library` slice:

| Column | Type | Meaning |
| --- | --- | --- |
| `direction` | str | `"up"` / `"down"` (or `"both"` if `direction="both"`). |
| `library` | str | Enrichr library slug (which panel this term came from). |
| `term` | str | Pathway / term name. |
| `overlap` | str | `"k/n"` -- foreground genes in the term / term size (`"k/?"` if the library could not be re-read). |
| `n_overlap` | int | Foreground genes in the term (from `genes`). |
| `n_term` | float | Term size in the library; `NaN` when unavailable. gseapy's background mode returns no `Overlap`, so both counts are derived by alphaPhos. |
| `p_value` | float | Nominal p from the hypergeometric test. |
| `fdr` | float | BH-adjusted q within this library × direction. |
| `odds_ratio` | float | Fisher odds ratio for the term. |
| `combined_score` | float | Enrichr's Combined Score (`log(p) × z-score of odds ratio`). |
| `genes` | str | Semicolon-joined list of hit genes in this term. |
| `n_foreground` | int | Size of the foreground (hit-gene) set for this direction. |
| `n_background` | int | Size of the background universe used. |

`result.attrs["provenance"]` carries: `libraries`, `background_type`
(`"phosphoproteome"`/`"custom"`/`"genome"`), `n_background_genes`,
`fdr_threshold`, `log2fc_threshold`, `direction`, `organism`,
`n_foreground_per_direction`, `gseapy_version`.

## Example

```python
import alphaphos as ap

result = ap.diff_exp_limma(adata, condition_column="condition", comparison=("EGF", "ctrl"))

# Default: split up/down, phosphoproteome background
enr = ap.enrichment.pathway_enrichment(result)

# Filter to significant terms at 5% FDR, KEGG only, up-regulated
sig = enr[
    (enr["library"] == "KEGG_2021_Human")
    & (enr["fdr"] < 0.05)
    & (enr["direction"] == "up")
]

# Preferred: proteome-identified background from a companion experiment
proteome_genes = pd.read_csv("proteome_ids.csv")["gene"].tolist()
enr = ap.enrichment.pathway_enrichment(result, background=proteome_genes)
```

## FDR control

BH-adjusted **within each `library` × `direction` slice**, matching Enrichr's convention.
If you compare terms across libraries (KEGG vs Reactome vs Hallmark), the p-values are
not jointly controlled -- BH is applied per library. This is standard practice; a term is
usually reported alongside its library.

## References

- Fang, Liu, Peltz 2023. *GSEApy: a comprehensive package for performing gene set
  enrichment analysis in Python.* Bioinformatics 39:btac757.
- Kuleshov et al. 2016. *Enrichr: a comprehensive gene set enrichment analysis web server
  2016 update.* Nucleic Acids Res 44:W90-W97.
- Reimand et al. 2019. *Pathway enrichment analysis and visualization of omics data using
  g:Profiler, GSEA, Cytoscape and EnrichmentMap.* Nature Protocols 14:482-517.
