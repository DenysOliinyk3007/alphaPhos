"""§8: apply condition_aware_classI mask to sn_all (instead of using sn_classI),
then re-run filter + impute + limma and compare to alphaphos+sum+classI.
"""
import nbformat as nbf
from pathlib import Path

p = Path("D:/Projects/alphaPhos/docs/benchmark/benchmark.ipynb")
nb = nbf.read(p, as_version=4)

cell_md = """## §8 · Same filtering policy on both sides

In §7 we compared `aphos_only_MS2_sum_classI` against `sn_classI`. But those
two have different filtering policies:

- alphaphos: per-condition majority rule (`apply_condition_aware_classI_mask`,
  default threshold 0.50)
- SN classI: per-cell binary threshold (Spectronaut's published 0.75 filter)

To isolate the *algorithm* (sum vs SN consolidation) from the *filtering policy*,
we re-do the comparison after applying **the same condition-aware mask** to
`sn_all` (Spectronaut's unfiltered PTM site report). This puts both versions
on identical filtering ground; remaining differences come from the collapse
itself.
"""

cell_code_apply_mask_to_sn = """# Convert sn_all to alphaphos-compatible site matrix and apply the mask.
sn_all_for_mask = sn_all_sites.copy()
sn_all_for_mask['PTM_Collapse_key'] = sn_all_for_mask.index
sn_all_for_mask = sn_all_for_mask.reset_index(drop=True)

print(f'sn_all (pre-mask): {len(sn_all_for_mask):,} sites')

sn_all_classI_aware, sn_decision = apply_condition_aware_classI_mask(
    df_sites=sn_all_for_mask,
    loc_per_run=sn_all_loc,
    sample_to_condition=condition_df,
    classI_cutoff=0.75,
    condition_threshold=0.50,
    drop_all_nan=True,
    return_decision_table=True,
)
print(f'sn_all + condition_aware: {len(sn_all_classI_aware):,} sites')
print(f'sn_classI (binary):       {len(sn_classI_sites):,} sites for reference')

# Build the samples-as-rows form for the downstream pipeline (matching §7)
sn_aware_wide = (sn_all_classI_aware.set_index('PTM_Collapse_key')
                                    [condition_df['sample'].tolist()].T)
sn_aware_wide.index.name = 'sample'
sn_aware_wide['condition'] = sn_aware_wide.index.map(s2c_map)

# Match the renaming we did in §7 (replace '|' with '~' so filter_phosphosites recognises site cols)
sn_aware_wide_n = normalize_site_col_names(sn_aware_wide)
print(f'\\nsn_aware_wide_n: {sn_aware_wide_n.shape}  (rows=samples, cols=sites+condition)')
"""

cell_code_filter_impute_limma = """# Filter + impute + limma — identical settings to §7
print('Filtering ...')
sn_aware_filtered = filter_phosphosites(sn_aware_wide_n, how='condition',
                                         cutoff=0.7, condition_col='condition')
n_kept = sum('~' in str(c) for c in sn_aware_filtered.columns)
print(f'  sites kept: {n_kept:,}')

print('Imputing ...')
sn_aware_imputed = impute_phosphosites(sn_aware_filtered)
print(f'  shape: {sn_aware_imputed.shape}')

print('Running limma ...')
limma_sn_aware = run_limma_via_rscript(sn_aware_imputed, 'sn_all_condition_aware')
print(f'  features: {len(limma_sn_aware):,}')
print(f'  significant (adj.P<0.05 & |logFC|>0.585): {int(limma_sn_aware[\"significant\"].sum())}')
print(f'  raw P<0.05: {int((limma_sn_aware[\"P.Value\"] < 0.05).sum())}')
"""

cell_code_compare_three = """# Three-way compare: aphos+sum+classI vs sn_classI (§7) vs sn_all+condition_aware (§8)

limma_sn_aware['match_key'] = limma_sn_aware['feature'].map(sn_feat_to_matchkey)
print(f'unmapped sn_aware feature ids: {int(limma_sn_aware[\"match_key\"].isna().sum())}')

sig_aware = set(limma_sn_aware.loc[limma_sn_aware['significant'], 'match_key'].dropna())
print()
print('Significant-hit overlap:')
print(f'  aphos sig:                  {len(sig_a):>5,}')
print(f'  sn_classI sig (§7):         {len(sig_s):>5,}')
print(f'  sn_all+cond_aware sig (§8): {len(sig_aware):>5,}')
print()
print(f'  aphos ∩ sn_classI       : {len(sig_a & sig_s):>5,}  J = {len(sig_a & sig_s)/max(1,len(sig_a | sig_s)):.3f}')
print(f'  aphos ∩ sn_all+aware    : {len(sig_a & sig_aware):>5,}  J = {len(sig_a & sig_aware)/max(1,len(sig_a | sig_aware)):.3f}')
print(f'  sn_classI ∩ sn_all+aware: {len(sig_s & sig_aware):>5,}  J = {len(sig_s & sig_aware)/max(1,len(sig_s | sig_aware)):.3f}')

# logFC concordance: aphos vs sn_all+aware
lfc_aware = (limma_aphos[['match_key', 'logFC', 'P.Value', 'adj.P.Val']]
             .merge(limma_sn_aware[['match_key', 'logFC', 'P.Value', 'adj.P.Val']],
                    on='match_key', suffixes=('_aphos', '_snaware'))
             .dropna(subset=['logFC_aphos', 'logFC_snaware']))

pear_aware  = float(lfc_aware['logFC_aphos'].corr(lfc_aware['logFC_snaware']))
spear_aware = float(lfc_aware['logFC_aphos'].corr(lfc_aware['logFC_snaware'], method='spearman'))
med_diff_aware = float((lfc_aware['logFC_aphos'] - lfc_aware['logFC_snaware']).abs().median())

print()
print(f'logFC concordance (aphos+sum vs sn_all+aware) on {len(lfc_aware):,} shared features:')
print(f'  Pearson r:    {pear_aware:.4f}   (was {pear:.4f} vs sn_classI)')
print(f'  Spearman r:   {spear_aware:.4f}   (was {spear:.4f} vs sn_classI)')
print(f'  median |diff|: {med_diff_aware:.4f}  (was {(lfc[\"logFC_aphos\"] - lfc[\"logFC_sn\"]).abs().median():.4f})')

print()
print('Summary:')
print(f'  Filtering policy:   §7 used SN binary, §8 uses condition_aware on both sides')
print(f'  Significant overlap:  Jaccard {len(sig_a & sig_s)/max(1,len(sig_a | sig_s)):.3f} -> {len(sig_a & sig_aware)/max(1,len(sig_a | sig_aware)):.3f}')
print(f'  logFC Pearson r:      {pear:.4f}            -> {pear_aware:.4f}')
"""

cell_code_viz = """# Updated logFC scatter: aphos+sum vs sn_all+condition_aware
def _class2(row):
    is_a = row['adj.P.Val_aphos']  < 0.05 and abs(row['logFC_aphos'])  > 0.585
    is_s = row['adj.P.Val_snaware'] < 0.05 and abs(row['logFC_snaware']) > 0.585
    if is_a and is_s: return 'both_sig'
    if is_a:           return 'aphos_only'
    if is_s:           return 'sn_aware_only'
    return 'not_sig'

lfc_aware['class'] = lfc_aware.apply(_class2, axis=1)
print(lfc_aware['class'].value_counts())

import plotly.express as px
fig = px.scatter(
    lfc_aware, x='logFC_snaware', y='logFC_aphos', color='class',
    color_discrete_map={'both_sig': '#e60000', 'aphos_only': '#1f77b4',
                        'sn_aware_only': '#ff7f0e', 'not_sig': '#cccccc'},
    opacity=0.5,
    title=f'logFC concordance (same filtering policy) — Pearson r = {pear_aware:.4f}',
    labels={'logFC_snaware': 'sn_all+condition_aware logFC',
            'logFC_aphos':   'alphaphos+sum logFC'},
    hover_data={'match_key': True, 'adj.P.Val_aphos': ':.3g', 'adj.P.Val_snaware': ':.3g'},
)
mn = float(lfc_aware[['logFC_aphos','logFC_snaware']].min().min())
mx = float(lfc_aware[['logFC_aphos','logFC_snaware']].max().max())
fig.add_shape(type='line', x0=mn, x1=mx, y0=mn, y1=mx, line=dict(dash='dash', color='black', width=1))
fig.update_layout(height=550, width=750, template='plotly_white')
fig.show()
"""

new_cells = [
    nbf.v4.new_markdown_cell(cell_md),
    nbf.v4.new_code_cell(cell_code_apply_mask_to_sn),
    nbf.v4.new_code_cell(cell_code_filter_impute_limma),
    nbf.v4.new_code_cell(cell_code_compare_three),
    nbf.v4.new_code_cell(cell_code_viz),
]
nb.cells.extend(new_cells)
nbf.write(nb, p)
print(f'Notebook now has {len(nb.cells)} cells (§8 appended at end).')
