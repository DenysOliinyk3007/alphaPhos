"""Helper: add §6 deep-diff cells to benchmark.ipynb."""
import nbformat as nbf
from pathlib import Path

p = Path("D:/Projects/alphaPhos/docs/benchmark/benchmark.ipynb")
nb = nbf.read(p, as_version=4)

cell_md_section6 = """## §6 · Deep diff — `aphos_new_MS2_classI` vs `sn_classI`

We have established that alphaPhos + MS2 + condition-aware classI matches SN
Class I at Jaccard 0.98 / Pearson r 0.92 / mean log2 diff -0.52. Now we
characterize the residual differences.

1. **Site-set differences** — which sites are in only one of the two?
2. **Cell-level NaN patterns** — for shared sites, are there cells where
   only one side has a value?
3. **Top disagreement sites** — the worst per-site log2 diffs
4. **Stratifications** — how does the diff depend on multiplicity, S/T/Y,
   localization probability, quant magnitude?
5. **Bland-Altman** — visualize the systematic vs random components of disagreement.
"""

cell_code_6_setup = """# Focus pair
APHOS_KEY = 'aphos_new_MS2_classI'  # old/new_MS2/only_MS2 are identical
SN_KEY    = 'sn_classI'

A = VERSIONS[APHOS_KEY]
B = VERSIONS[SN_KEY]

a_only = sorted(set(A.index) - set(B.index))
b_only = sorted(set(B.index) - set(A.index))
shared = sorted(set(A.index) & set(B.index))

print(f'aphos_new_MS2_classI sites: {len(A):,}')
print(f'sn_classI            sites: {len(B):,}')
print()
print(f'  shared (in both):     {len(shared):,}')
print(f'  aphos-only:           {len(a_only):,}')
print(f'  sn-only:              {len(b_only):,}')
print()
print('Examples of aphos-only keys (first 10):')
for k in a_only[:10]:
    print(f'  {k}')
print()
print('Examples of sn-only keys (first 10):')
for k in b_only[:10]:
    print(f'  {k}')
"""

cell_code_6_site_set_chars = """# Site-set differences — break down by multiplicity and amino acid
def key_components(keys):
    rows = []
    for k in keys:
        parts = k.split('|')
        if len(parts) == 4:
            rows.append({'protein': parts[0], 'aa': parts[1],
                         'pos': int(parts[2]), 'mult': int(parts[3])})
    return pd.DataFrame(rows)

shared_df  = key_components(shared).assign(set='shared')
a_only_df  = key_components(a_only).assign(set='aphos_only')
b_only_df  = key_components(b_only).assign(set='sn_only')
all_keys_df = pd.concat([shared_df, a_only_df, b_only_df], ignore_index=True)

print('--- by multiplicity ---')
print(pd.crosstab(all_keys_df['mult'], all_keys_df['set'], margins=True))
print()
print('--- by amino acid ---')
print(pd.crosstab(all_keys_df['aa'], all_keys_df['set'], margins=True))
print()
print('--- per-set distribution by multiplicity (column %) ---')
print((pd.crosstab(all_keys_df['mult'], all_keys_df['set'], normalize='columns') * 100).round(1))
"""

cell_code_6_nan_patterns = """# Cell-level NaN patterns on the shared site set
sample_cols = condition_df['sample'].tolist()
A_shared = A.loc[shared, sample_cols]
B_shared = B.loc[shared, sample_cols]

a_present = A_shared.notna()
b_present = B_shared.notna()

n_both    = int((a_present & b_present).sum().sum())
n_a_only  = int((a_present & ~b_present).sum().sum())
n_b_only  = int((~a_present & b_present).sum().sum())
n_neither = int((~a_present & ~b_present).sum().sum())
n_total = a_present.size

print(f'Shared sites: {len(shared):,}, samples: {a_present.shape[1]}, total cells: {n_total:,}')
print()
print(f'  cell present in BOTH      : {n_both:>7,}  ({n_both/n_total*100:5.1f}%)')
print(f'  cell present ONLY in aphos: {n_a_only:>7,}  ({n_a_only/n_total*100:5.1f}%)')
print(f'  cell present ONLY in SN   : {n_b_only:>7,}  ({n_b_only/n_total*100:5.1f}%)')
print(f'  cell NaN in BOTH          : {n_neither:>7,}  ({n_neither/n_total*100:5.1f}%)')
print()
print('Interpretation:')
print('  aphos-only cells: aphos kept a measurement SN censored as Filtered')
print('  (per-condition majority-keep vs per-cell threshold; expected).')
print('  SN-only cells: SN has a value where aphos has NaN (less expected; investigate).')
"""

cell_code_6_top_disagree = """# Per-site median log2 diff
diff_long = (A_shared - B_shared).stack(dropna=False)
diff_long = diff_long.dropna().rename('log2_diff').reset_index()
diff_long.columns = ['match_key', 'sample', 'log2_diff']

per_site = diff_long.groupby('match_key').agg(
    n_cells=('log2_diff', 'size'),
    median_log2_diff=('log2_diff', 'median'),
    max_abs_log2_diff=('log2_diff', lambda s: s.abs().max()),
)

components = key_components(per_site.index.tolist()).assign(match_key=per_site.index.tolist()).set_index('match_key')
per_site = per_site.join(components)

print('Top 20 sites by |median log2 diff| (alphaphos - SN classI):')
top = per_site.reindex(per_site['median_log2_diff'].abs().sort_values(ascending=False).index).head(20)
print(top.to_string())
print()
print('Per-site median diff distribution:')
print(per_site['median_log2_diff'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3))
"""

cell_code_6_strat = """# Stratify cell-level diff by multiplicity, amino acid, and quant magnitude
diff_full = diff_long.copy()
parts_split = diff_full['match_key'].str.split('|', expand=True)
diff_full['aa'] = parts_split[1]
diff_full['mult'] = parts_split[3].astype(int)

sn_long = B_shared.stack(dropna=False).dropna().rename('sn_log2').reset_index()
sn_long.columns = ['match_key', 'sample', 'sn_log2']
diff_full = diff_full.merge(sn_long, on=['match_key', 'sample'], how='left')
diff_full['sn_quant_bin'] = pd.cut(
    diff_full['sn_log2'],
    bins=[-np.inf, 4, 6, 8, 10, 12, np.inf],
    labels=['<4', '4-6', '6-8', '8-10', '10-12', '>12'],
)

def _strat(by):
    g = diff_full.groupby(by, observed=True)['log2_diff']
    return pd.DataFrame({
        'n':              g.size(),
        'mean':           g.mean().round(3),
        'median':         g.median().round(3),
        'p05':            g.quantile(0.05).round(3),
        'p95':            g.quantile(0.95).round(3),
        'pct_within_0.1': (g.apply(lambda s: (s.abs() < 0.1).mean() * 100)).round(2),
    })

print('--- by multiplicity ---')
print(_strat('mult'))
print()
print('--- by amino acid ---')
print(_strat('aa'))
print()
print('--- by SN classI quant magnitude (log2) ---')
print(_strat('sn_quant_bin'))
"""

cell_code_6_bland_altman = """# Bland-Altman: mean vs difference (log2 space)
import plotly.express as px

ba = diff_full.dropna(subset=['sn_log2']).copy()
ba['mean_log2'] = ba['sn_log2'] + ba['log2_diff'] / 2

n_plot = min(10000, len(ba))
sub = ba.sample(n=n_plot, random_state=0)

fig = px.scatter(
    sub,
    x='mean_log2', y='log2_diff',
    color=sub['mult'].astype(str),
    opacity=0.35,
    hover_data={'match_key': True, 'mean_log2': ':.2f',
                'log2_diff': ':.3f', 'mult': True, 'aa': True},
    title=f'Bland-Altman (subsample of {n_plot:,} cells) - alphaPhos vs SN Class I',
    labels={'mean_log2': 'mean log2 intensity',
            'log2_diff': 'log2 diff (alphaphos - SN)',
            'color': 'multiplicity'},
)
mean_diff = ba['log2_diff'].mean()
sd_diff = ba['log2_diff'].std()
fig.add_hline(y=mean_diff, line_dash='dash', annotation_text=f'mean = {mean_diff:.3f}')
fig.add_hline(y=mean_diff + 1.96*sd_diff, line_dash='dot',
              annotation_text=f'+1.96 SD = {mean_diff + 1.96*sd_diff:.2f}')
fig.add_hline(y=mean_diff - 1.96*sd_diff, line_dash='dot',
              annotation_text=f'-1.96 SD = {mean_diff - 1.96*sd_diff:.2f}')
fig.add_hline(y=0, line_color='black', line_width=1)
fig.update_layout(height=500, width=900, template='plotly_white')
fig.show()
"""

section6_idx = None
for i, c in enumerate(nb.cells):
    if c.cell_type == 'markdown' and c.source.lstrip().startswith('## §6'):
        section6_idx = i
        break
assert section6_idx is not None

nb.cells[section6_idx].source = cell_md_section6
cells_to_insert = [
    nbf.v4.new_code_cell(cell_code_6_setup),
    nbf.v4.new_code_cell(cell_code_6_site_set_chars),
    nbf.v4.new_code_cell(cell_code_6_nan_patterns),
    nbf.v4.new_code_cell(cell_code_6_top_disagree),
    nbf.v4.new_code_cell(cell_code_6_strat),
    nbf.v4.new_code_cell(cell_code_6_bland_altman),
]
for off, cell in enumerate(cells_to_insert, start=1):
    nb.cells.insert(section6_idx + off, cell)

nbf.write(nb, p)
print(f'Notebook now has {len(nb.cells)} cells.')
print(f'§6 markdown at index {section6_idx}, followed by 6 code cells.')
