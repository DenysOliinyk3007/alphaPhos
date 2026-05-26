"""Helper: add §6.5 cells — test aggregation_method='consolidate' to close
the systematic offset uncovered in §6.
"""
import nbformat as nbf
from pathlib import Path

p = Path("D:/Projects/alphaPhos/docs/benchmark/benchmark.ipynb")
nb = nbf.read(p, as_version=4)

cell_md_section6_5 = """## §6.5 · Test `aggregation_method='consolidate'`

§6 showed the systematic alphaPhos-vs-SN log2 offset is **intensity-dependent**:
near-zero at low quant, grows to ~-1.0 log2 at high quant. The likely cause is
**median vs sum aggregation** of multi-precursor sites:

- alphaPhos default we used: `aggregation_method='median'` (Dublin convention)
- SN's PTM consolidation: `"Linear modeling based"` = the canonical Hogrebe
  ratio-imputation + **sum**. We already ported this as `'consolidate'`.

We re-collapse the same MS2 input with `aggregation_method='consolidate'` and
re-score the result against `sn_classI`. If our diagnosis is right, Pearson r
should rise from 0.92 and the mean log2 diff should drop from -0.52 toward zero.

The simpler `aggregation_method='sum'` is a sanity check (SN's other PTM
consolidation option — plain sum without imputation).
"""

cell_code_recollapse = """import time

# Re-collapse aphos_only_MS2 (cheapest single-quant file) with both alternative
# aggregations. Skip if cached.

ALT_AGGS = ['consolidate', 'sum']

def _recollapse(agg_method):
    raw_path = OUT / f'aphos_only_MS2_{agg_method}_raw.parquet'
    classI_path = OUT / f'aphos_only_MS2_{agg_method}_classI.parquet'
    if raw_path.exists() and classI_path.exists():
        print(f'  [{agg_method}] cached, skipping')
        return

    t0 = time.time()
    df = read_spectronaut(FILES['new_ms2'], quant_level='MS2',
                          drop_decoys=True, pg_qvalue_max=0.01,
                          top_n_attribution=True)
    print(f'  [{agg_method}] loaded {len(df):,} rows ({time.time()-t0:.1f}s)')

    t0 = time.time()
    sites_raw, loc_per_run = collapse_sites(
        df,
        cutoff=0.0,
        collapse_level='PG',
        aggregation_method=agg_method,
        localization_strategy='global_max',
        noise_floor_filter=True,
        add_kinase_sequences=False,
    )
    print(f'  [{agg_method}] collapsed -> {len(sites_raw):,} sites ({time.time()-t0:.1f}s)')

    sites_classI, _ = apply_condition_aware_classI_mask(
        df_sites=sites_raw, loc_per_run=loc_per_run,
        sample_to_condition=condition_df,
        classI_cutoff=0.75, condition_threshold=0.50,
        drop_all_nan=True, return_decision_table=True,
    )
    sites_raw.to_parquet(raw_path)
    sites_classI.to_parquet(classI_path)
    print(f'  [{agg_method}] classI -> {len(sites_classI):,} sites')

for agg in ALT_AGGS:
    print(f'==== aggregation_method={agg} ====')
    _recollapse(agg)
    print()

print('Done. Output files:')
for agg in ALT_AGGS:
    for v in ('raw', 'classI'):
        f = OUT / f'aphos_only_MS2_{agg}_{v}.parquet'
        if f.exists():
            print(f'  {f.name}  ({f.stat().st_size/1e6:.1f} MB)')
"""

cell_code_compare_alt_aggs = """# Compare consolidate / sum / median (the original) against sn_classI
ref = VERSIONS['sn_classI']
sample_cols_ref = condition_df['sample'].tolist()


def score_vs_classI(label, sites_df):
    keyed = alpha_to_matchkey_frame(sites_df)
    keyed = keyed.reindex(columns=sample_cols_ref)
    shared_keys = ref.index.intersection(keyed.index)
    a = keyed.loc[shared_keys]
    b = ref.loc[shared_keys]
    both = pd.DataFrame({'a': a.values.flatten(), 'b': b.values.flatten()}).dropna()
    if len(both) < 50:
        return None
    diff = both['a'] - both['b']
    return {
        'config':       label,
        'n_sites':      len(keyed),
        'shared_keys':  len(shared_keys),
        'paired_cells': len(both),
        'pearson_log2': round(float(both['a'].corr(both['b'])), 4),
        'mean_log2_diff':   round(float(diff.mean()), 4),
        'median_log2_diff': round(float(diff.median()), 4),
        'sd_log2_diff':     round(float(diff.std()), 4),
        'p05_diff':         round(float(diff.quantile(0.05)), 4),
        'p95_diff':         round(float(diff.quantile(0.95)), 4),
        'pct_within_0.1':   round(float((diff.abs() < 0.1).mean() * 100), 2),
        'pct_within_0.5':   round(float((diff.abs() < 0.5).mean() * 100), 2),
    }


rows = []
# Baseline: the median version we already have
rows.append(score_vs_classI('median   (already in §5)', pd.read_parquet(OUT / 'aphos_only_MS2_classI.parquet')))
for agg in ALT_AGGS:
    f = OUT / f'aphos_only_MS2_{agg}_classI.parquet'
    rows.append(score_vs_classI(f'{agg:>9s}', pd.read_parquet(f)))

agg_compare = pd.DataFrame(rows)
print('Cell-level agreement vs sn_classI across alphaPhos aggregation methods:')
print(agg_compare.to_string(index=False))
"""

cell_code_strat_consolidate = """# Re-stratify the consolidate version by SN quant magnitude — does the
# intensity-dependent bias collapse?

best_label = 'aphos_only_MS2_consolidate_classI'
best_df = alpha_to_matchkey_frame(pd.read_parquet(OUT / 'aphos_only_MS2_consolidate_classI.parquet'))
best_df = best_df.reindex(columns=sample_cols_ref)

shared_keys = ref.index.intersection(best_df.index)
A2 = best_df.loc[shared_keys]
B2 = ref.loc[shared_keys]

d_long = (A2 - B2).stack(dropna=False).dropna().rename('log2_diff').reset_index()
d_long.columns = ['match_key', 'sample', 'log2_diff']
b_long = B2.stack(dropna=False).dropna().rename('sn_log2').reset_index()
b_long.columns = ['match_key', 'sample', 'sn_log2']
d_long = d_long.merge(b_long, on=['match_key', 'sample'], how='left')
d_long['sn_quant_bin'] = pd.cut(
    d_long['sn_log2'],
    bins=[-np.inf, 4, 6, 8, 10, 12, np.inf],
    labels=['<4', '4-6', '6-8', '8-10', '10-12', '>12'],
)

g = d_long.groupby('sn_quant_bin', observed=True)['log2_diff']
strat = pd.DataFrame({
    'n':              g.size(),
    'mean':           g.mean().round(3),
    'median':         g.median().round(3),
    'p05':            g.quantile(0.05).round(3),
    'p95':            g.quantile(0.95).round(3),
    'pct_within_0.1': (g.apply(lambda s: (s.abs() < 0.1).mean() * 100)).round(2),
})
print(f'Stratification by SN quant magnitude — {best_label!r}:')
print(strat)
"""

section6_idx = None
for i, c in enumerate(nb.cells):
    if c.cell_type == 'markdown' and c.source.lstrip().startswith('## §6 ·'):
        section6_idx = i
        break

# Find end of §6 (right before §7 markdown)
end_of_section6 = None
for i in range(section6_idx + 1, len(nb.cells)):
    if nb.cells[i].cell_type == 'markdown' and nb.cells[i].source.lstrip().startswith('## §7'):
        end_of_section6 = i
        break

insert_at = end_of_section6 if end_of_section6 is not None else len(nb.cells)

new_cells = [
    nbf.v4.new_markdown_cell(cell_md_section6_5),
    nbf.v4.new_code_cell(cell_code_recollapse),
    nbf.v4.new_code_cell(cell_code_compare_alt_aggs),
    nbf.v4.new_code_cell(cell_code_strat_consolidate),
]
for off, cell in enumerate(new_cells):
    nb.cells.insert(insert_at + off, cell)

nbf.write(nb, p)
print(f'Notebook now has {len(nb.cells)} cells.')
print(f'§6.5 inserted at index {insert_at}, 4 cells (md + 3 code).')
