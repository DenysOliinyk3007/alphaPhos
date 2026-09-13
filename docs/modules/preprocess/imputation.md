# `alphaphos.preprocess.impute`

Two imputers for phosphosite quant matrices:

- [`impute_hybrid`](#impute_hybrid) -- per-cell hybrid MAR/MNAR. **Recommended** default.
- [`impute_knn_site_based`](#impute_knn_site_based) -- pure site-based KNN. Kept for
  Dublin-parity / R-limma-parity workflows.

Both mutate `adata` in place by default (`copy=False`) **and** return it, so both
usage patterns are safe:

```python
ap.impute_hybrid(adata)           # in-place, return ignored
adata = ap.impute_hybrid(adata)   # equivalent, same object, no None-trap
adata2 = ap.impute_hybrid(adata, copy=True)   # explicit fresh copy
```

Both operate on `layers["intensity_log2"]` by default and require that the target
layer has no all-NaN sites (run [`filter_by_completeness`](filter.md) first).

## Scientific basis

DIA phosphosite missingness is a mixture of two mechanisms:

- **MAR** (missing at random) -- the site is detectable but was missed in this sample
  (competition, chromatography, stochastic sampling). Best imputed by borrowing from
  neighbouring sites that behave similarly.
- **MNAR** (missing not at random) -- the site is below the LOD in this sample. Its
  true value lives in the noise tail; borrowing from KNN neighbours (which are all
  above the LOD) systematically overshoots.

`impute_hybrid` classifies each missing cell independently and routes to the appropriate
imputer -- **site-based KNN** for MAR (find k most-similar SITES, not samples, because
n_samples &Lt; n_features), **Perseus-style downshifted Gaussian** for MNAR
(`N(&mu; &minus; offset&middot;&sigma;, factor&middot;&sigma;)`). The site-based direction
of KNN is critical -- sample-based KNN with `n_samples &lt; 10` picks
half-the-dataset as neighbours and produces near-mean imputation.

Reference: Dublin et al. 2020 `impute_phosphosites` (whose defaults `impute_knn_site_based`
reproduces exactly, matching R-limma to numerical precision on the EGF benchmark).
Perseus MNAR: Tyanova et al. 2016 *Nature Protocols*.

---

## `impute_hybrid`

Per-cell hybrid MAR/MNAR imputation. **Recommended** default.

### Signature

```python
ap.impute_hybrid(
    adata: ad.AnnData,
    *,
    mnar_threshold_percentile: float = 30.0,
    knn_n_neighbors: int | None = None,
    knn_weights: Literal["uniform", "distance"] = "uniform",
    gaussian_std_offset: float = 1.8,
    gaussian_std_factor: float = 0.3,
    gaussian_seed: int = 42,
    layer: str | None = "intensity_log2",
    return_audit: bool = False,
    copy: bool = False,
) -> ad.AnnData | tuple[ad.AnnData, pd.DataFrame]
```

### Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `adata` | *required* | `(n_samples, n_sites)` AnnData; target layer must have no all-NaN sites. |
| `mnar_threshold_percentile` | `30.0` | Sites whose mean observed intensity is in the bottom this percentile of the dataset are treated as low-abundance / MNAR for their missing cells. |
| `knn_n_neighbors` | `None` &rarr; `int(sqrt(n_samples))` | k for the MAR-path KNN. |
| `knn_weights` | `"uniform"` | KNN vote weighting. | `"distance"` weights neighbours inversely by distance. |
| `gaussian_std_offset` | `1.8` | Downshift in units of `sigma`. Perseus default. |
| `gaussian_std_factor` | `0.3` | Width factor of the imputed Gaussian. Perseus default. |
| `gaussian_seed` | `42` | RNG seed for reproducible Gaussian draws. |
| `layer` | `"intensity_log2"` | Which layer to impute. Imputing the canonical layer also updates `adata.X` (`.X == layers["intensity_log2"]` contract). `None` targets `adata.X` only. |
| `return_audit` | `False` | If `True`, also return a per-missing-cell DataFrame recording which strategy was applied. |
| `copy` | `False` | If `True`, mutate a fresh copy and return it. If `False`, mutate in-place and return the same object. |

### Per-cell classification

For each missing cell `(site, sample)`:

- If the site has **no observed values anywhere**, OR the site's mean observed intensity
  is below the `mnar_threshold_percentile` of the dataset's overall observed-cell
  distribution &rarr; **MNAR**: draw from
  `N(&mu;_sample &minus; offset&middot;&sigma;_sample, factor&middot;&sigma;_sample)`.
- Otherwise &rarr; **MAR**: impute via site-based KNN.

Split point is per-site, but the imputation is per-cell -- so a mostly-MAR site with a
truly-below-LOD sample can still get a Gaussian value for that sample if the site
qualifies as MNAR overall.

### Output

- **Always returns the AnnData.** When `copy=False` (default) it's the same object
  passed in, mutated in place. When `copy=True`, a fresh copy.
- If `return_audit=True`, returns a `(adata, audit)` tuple. `audit` is a per-cell
  DataFrame with columns `(sample_idx, site_idx, strategy)` for every missing cell,
  where `strategy` is either `"MAR_KNN"` or `"MNAR_Gaussian"`.

### Example

```python
import alphaphos as ap

# In-place, defaults
ap.impute_hybrid(adata)
# or, equivalently:
adata = ap.impute_hybrid(adata)

# Return audit trail alongside the (in-place-mutated) AnnData
adata, audit = ap.impute_hybrid(adata, return_audit=True)
print(audit["strategy"].value_counts())  # e.g. MAR_KNN: 340, MNAR_Gaussian: 82
```

---

## `impute_knn_site_based`

Pure site-based KNN. Drop-in replacement for Dublin's `impute_phosphosites`; same
algorithm, same defaults. Reproduces R-limma on the EGF benchmark at Pearson r = 1.0000
on log2FC.

### Signature

```python
ap.impute_knn_site_based(
    adata: ad.AnnData,
    *,
    n_neighbors: int | None = None,
    weights: Literal["uniform", "distance"] = "uniform",
    layer: str | None = "intensity_log2",
    copy: bool = False,
) -> ad.AnnData
```

### Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `adata` | *required* | `(n_samples, n_sites)` AnnData; every site column must have at least one observed value. |
| `n_neighbors` | `None` &rarr; `int(sqrt(n_samples))` | k. Dublin's convention. |
| `weights` | `"uniform"` | Neighbour vote weighting. |
| `layer` | `"intensity_log2"` | Which layer to impute. Imputing the canonical layer also updates `.X` (`.X == layers["intensity_log2"]` contract). `None` targets `.X` only. |
| `copy` | `False` | If `True`, mutate a fresh copy and return it. If `False`, mutate in-place and return the same object. |

### When to use pure KNN over hybrid

Pick pure KNN when you specifically need Dublin / R-limma parity, or when your prior
domain knowledge says missingness is dominantly MAR (unusual for phospho DIA -- typically
some proportion is truly below LOD). `impute_hybrid` is otherwise the more defensible
default because it doesn't force below-LOD sites through a KNN borrow.

---

## Common caveats

- Both functions assume the target layer is **log2-scale**. Imputing on linear-scale data
  gives nonsense because the Gaussian downshift is calibrated in log-space `sigma`.
- Both **require prior completeness filtering** -- an all-NaN site column raises.
- Both mutate `adata` in place by default and **also return it**, so
  `adata = ap.impute_hybrid(adata)` is safe (never returns `None`). Set `copy=True`
  when you need to preserve the original object.
- Reproducibility: pass `gaussian_seed` (and use `weights="uniform"` for KNN) for
  deterministic output.

## References

- Tyanova, Temu, Cox 2016. *The MaxQuant computational platform for mass
  spectrometry-based shotgun proteomics.* Nature Protocols 11:2301-2319. (Perseus MNAR
  downshifted Gaussian.)
- Dublin et al. 2020 `impute_phosphosites` -- the site-based KNN reference impl.
