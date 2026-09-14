# `alphaphos.enrichment.ksea`

Kinase-activity inference on per-site limma output.

`kinase_activity(diff_exp_result, ...)` scores each kinase in a kinase-substrate network
according to how consistently its substrate sites are moving in a given contrast. Positive
score = activated, negative = inhibited.

## Scientific basis

Wraps [`decoupler-py`](https://github.com/saezlab/decoupler-py) (Badia-i-Mompel et al. 2022
*Bioinformatics Advances* 2:vbac016). Two methods are exposed:

- **ULM** (Univariate Linear Model, default) -- fits `y_site ~ β₀ + β₁ · membership_kinase`
  independently per kinase; the enrichment score is the t-value of the slope. This is the
  modern reformulation of the classical Wiredja 2017 KSEA z-score -- essentially the same
  test as a linear model, numerically equivalent on typical phospho data, and what
  Saez-Rodriguez et al. established as the current field standard.
- **MLM** (Multivariate Linear Model) -- fits all kinases jointly as covariates, so
  substrate-sharing across kinases is accounted for in one regression. More rigorous when
  many kinases share substrates, but the design matrix goes rank-deficient on dense
  networks (like full OmniPath); `kinase_activity` catches the `LinAlgError` and returns a
  message pointing you at ULM or a higher-confidence subset.

Both methods return **BH-adjusted FDR** across the kinases tested in that call
(decoupler applies BH internally).

## Signature

```python
ap.enrichment.kinase_activity(
    diff_exp_result: pd.DataFrame,
    *,
    stat_col: str = "log2fc",
    network: str | pd.DataFrame = "omnipath",
    method: Literal["ulm", "mlm"] = "ulm",
    min_substrates: int = 5,
    organism: Literal["human", "mouse", "rat"] = "human",
    fdr_method: str = "bh",
    seed: int = 42,
    cache_path: str | Path | None = None,
    key_column: str | None = None,
) -> pd.DataFrame
```

## Input

A limma `diff_exp_result` DataFrame (from `ap.diff_exp_limma`) indexed by alphaPhos site
keys `Protein|Gene|Site|Mult`. Site keys are auto-canonicalised to OmniPath's
`Protein_AApos` format. If multiple keys map to the same site (multiplicity variants), the
one with the largest `|log2fc|` is kept.

## Parameters

| Parameter | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `diff_exp_result` | *required* | Per-site limma output (indexed by alphaPhos keys). | -- |
| `stat_col` | `"log2fc"` | Column carrying the signed effect. | `"t_stat"` (moderated t) is also defensible. |
| `network` | `"omnipath"` | Kinase-substrate network. | `"ptm_db"` (our curated PTM DB) or a `pd.DataFrame` with `source`/`target` columns (BYO). |
| `method` | `"ulm"` | Statistical method. | `"mlm"` (multivariate, requires low-density network). |
| `min_substrates` | `5` | Minimum substrates observed per kinase to test. | Decoupler convention; raise to be stricter. |
| `organism` | `"human"` | For OmniPath fetch. | `"mouse"`, `"rat"`. Ignored for `ptm_db` / BYO. |
| `fdr_method` | `"bh"` | Currently only BH supported (applied internally by decoupler). | -- (API forward-compat). |
| `seed` | `42` | Unused -- ULM / MLM are deterministic; kept for backward compatibility. | -- |
| `cache_path` | `None` | OmniPath fetch cache (parquet). | Any path -- first fetch writes here, subsequent runs read from it. |
| `key_column` | `None` | Column carrying site keys if not the index. | Any column name. |

### Network choices in detail

- **`"omnipath"`** -- Enzsub network via the `omnipath` package (community standard, ~50k
  edges after human/mouse/rat filtering). Cached to parquet on first fetch. Best default.
- **`"ptm_db"`** -- ad-hoc built from the bundled 235k-site PTM DB. Can be filtered by
  `curation_confidence` tier via `load_ptm_ks_network(min_curation_confidence="high")`.
  Higher precision, lower coverage. Use for MLM which needs a low-density design.
- **`pd.DataFrame`** (BYO) -- caller-supplied edge table. Must have `source` (kinase),
  `target` (site ID matching your index format), and optionally `weight` (defaults 1.0).
  Extra columns are preserved.

## Output

A `pd.DataFrame`, one row per kinase, sorted by `fdr` ascending:

| Column | Type | Meaning |
| --- | --- | --- |
| `kinase` | str | Kinase symbol (network `source` value; UniProt gene symbol for OmniPath). |
| `score` | float | Signed activity -- t-value of the slope. Positive = activated, negative = inhibited. |
| `fdr` | float | BH-adjusted q-value across kinases tested in this run. |
| `n_substrates` | int | Substrates of this kinase observed in the input (after `min_substrates` filter). |
| `direction` | str | `"up"` if `score >= 0` else `"down"`. |

`result.attrs["provenance"]` carries: `method`, `network`, `organism`, `min_substrates`,
`n_input_sites`, `n_canonicalised_sites`, `n_overlap_with_network`, `n_kinases_tested`,
`seed`, `decoupler_version`.

## Example

```python
import alphaphos as ap

# Assume you have a limma result from ap.diff_exp_limma
result = ap.diff_exp_limma(adata, condition_column="condition", comparison=("EGF", "ctrl"))

# ULM on OmniPath (default)
ksea = ap.enrichment.kinase_activity(result, method="ulm", network="omnipath")

# Significant kinases at 5% FDR
sig = ksea[ksea["fdr"] < 0.05]
print(sig[sig["direction"] == "up"].head())

# Alternative: MLM on the high-confidence curated PTM-DB slice
net = ap.enrichment.load_ptm_ks_network(min_curation_confidence="high")
ksea_mlm = ap.enrichment.kinase_activity(result, method="mlm", network=net)
```

## FDR control

BH-adjusted **within a single call** across all kinases tested. If you run several
contrasts and want to control FDR across them jointly, concatenate the p-values (need to
change decoupler-py's return path) and re-apply BH externally. BH assumes independent /
positive-dependent hypotheses; kinases with shared substrates violate strict independence
but this is what the field uses and reviewers expect.

## References

- Badia-i-Mompel et al. 2022. *decoupler: ensemble of computational methods to infer
  biological activities from omics data.* Bioinformatics Advances 2:vbac016.
- Türei et al. 2021. *Integrated intra- and intercellular signaling knowledge for
  multicellular omics analysis.* Molecular Systems Biology 17:e9923. (OmniPath)
- Wiredja et al. 2017. *The KSEA App: a web-based tool for kinase activity inference from
  quantitative phosphoproteomics.* Bioinformatics 33:3489-3491. (Classical KSEA reference
  point.)
