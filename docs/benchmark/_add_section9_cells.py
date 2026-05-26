"""§9: Sample QC — PCA + per-sample CV for aphos_imputed and sn_imputed."""
import nbformat as nbf
from pathlib import Path

p = Path("D:/Projects/alphaPhos/docs/benchmark/benchmark.ipynb")
nb = nbf.read(p, as_version=4)

cell_md = """## §9 · Sample QC — PCA + per-sample CV

Two technical checks comparing `aphos_imputed` and `sn_imputed` (the post-filter,
post-imputation matrices that fed into limma in §7):

1. **PCA** — Z-score each site across samples, PCA on samples × sites.
   Do withEGF / woEGF replicates separate the same way in both versions?
2. **Per-site CV within condition** — distribution of linear-space CV across
   replicates. Tighter CV = better technical reproducibility.

If both versions show clean condition separation in PC1 and comparable CV
distributions, technical reproducibility is equivalent.
"""

cell_code_pca = """from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run_pca(imputed_wide, label):
    site_cols = [c for c in imputed_wide.columns if '~' in str(c)]
    X = imputed_wide[site_cols].values
    X_z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_z)
    var = pca.explained_variance_ratio_ * 100
    return scores, var


aphos_pca, aphos_var = run_pca(aphos_imputed, 'aphos')
sn_pca,    sn_var    = run_pca(sn_imputed, 'sn')

aphos_meta = aphos_imputed[['condition']].copy()
aphos_meta[['PC1', 'PC2']] = aphos_pca
sn_meta = sn_imputed[['condition']].copy()
sn_meta[['PC1', 'PC2']]    = sn_pca

print(f'aphos PC1 / PC2 explained variance: {aphos_var[0]:.1f}% / {aphos_var[1]:.1f}%')
print(f'sn    PC1 / PC2 explained variance: {sn_var[0]:.1f}% / {sn_var[1]:.1f}%')

# Plot side-by-side
fig = make_subplots(rows=1, cols=2, subplot_titles=[
    f'aphos+sum+classI ({aphos_var[0]:.1f}% / {aphos_var[1]:.1f}%)',
    f'sn_classI ({sn_var[0]:.1f}% / {sn_var[1]:.1f}%)',
])

palette = {'withEGF': '#e60000', 'woEGF': '#1f77b4'}
for col, (df, n_sites) in enumerate([(aphos_meta, len([c for c in aphos_imputed.columns if '~' in str(c)])),
                                      (sn_meta,    len([c for c in sn_imputed.columns    if '~' in str(c)]))], 1):
    for cond, sub in df.groupby('condition'):
        fig.add_trace(go.Scatter(
            x=sub['PC1'], y=sub['PC2'],
            mode='markers',
            marker=dict(size=14, color=palette.get(cond, '#888'), line=dict(width=1, color='black')),
            name=f'{cond} (n={len(sub)})',
            text=sub.index, hoverinfo='text+x+y',
            showlegend=(col == 1),
        ), row=1, col=col)
    fig.update_xaxes(title_text='PC1', row=1, col=col)
    fig.update_yaxes(title_text='PC2', row=1, col=col)

fig.update_layout(height=450, width=1000, template='plotly_white',
                  title='PCA — same samples, two pipeline versions')
fig.show()
"""

cell_code_cv = """# Per-site CV within condition (linear space) — distribution comparison

def per_site_cv_per_condition(imputed_wide, label):
    site_cols = [c for c in imputed_wide.columns if '~' in str(c)]
    rows = []
    for cond, sub in imputed_wide.groupby('condition'):
        if cond not in ('withEGF', 'woEGF'):
            continue
        block = sub[site_cols]
        lin = np.power(2.0, block)
        mean = lin.mean(axis=0, skipna=True)
        sd = lin.std(axis=0, skipna=True, ddof=1)
        cv = (sd / mean * 100).dropna()
        for v in cv.values:
            rows.append({'version': label, 'condition': cond, 'cv_pct': v})
    return pd.DataFrame(rows)


cv_aphos = per_site_cv_per_condition(aphos_imputed, 'aphos+sum+classI')
cv_sn    = per_site_cv_per_condition(sn_imputed, 'sn_classI')
cv_all = pd.concat([cv_aphos, cv_sn], ignore_index=True)

print('Per-site CV summary (%):')
print(cv_all.groupby(['version', 'condition'])['cv_pct'].describe(percentiles=[0.25, 0.5, 0.75]).round(1))

import plotly.express as px
y_top = float(np.nanpercentile(cv_all['cv_pct'], 99))
fig = px.box(
    cv_all, x='condition', y='cv_pct', color='version',
    points=False,
    category_orders={'condition': ['woEGF', 'withEGF'], 'version': ['aphos+sum+classI', 'sn_classI']},
    title='Per-site CV within condition (linear space)',
    color_discrete_map={'aphos+sum+classI': '#e60000', 'sn_classI': '#1f77b4'},
)
fig.update_traces(boxmean=True)
fig.update_yaxes(title='CV %', range=[0, y_top])
fig.update_layout(height=480, width=800, template='plotly_white')
fig.show()
"""

cell_code_summary = """# One-glance summary
print('=== Sample QC summary ===')
print()
print(f'aphos+sum+classI:  {sum(\"~\" in str(c) for c in aphos_imputed.columns):,} sites, '
      f'PC1 {aphos_var[0]:.1f}% / PC2 {aphos_var[1]:.1f}%, '
      f'median within-condition CV: {cv_aphos[\"cv_pct\"].median():.1f}%')
print(f'sn_classI:         {sum(\"~\" in str(c) for c in sn_imputed.columns):,} sites, '
      f'PC1 {sn_var[0]:.1f}% / PC2 {sn_var[1]:.1f}%, '
      f'median within-condition CV: {cv_sn[\"cv_pct\"].median():.1f}%')
print()
print('If PC1 explains a similar % of variance and median CV is similar,')
print('technical reproducibility is equivalent across both versions.')
"""

new_cells = [
    nbf.v4.new_markdown_cell(cell_md),
    nbf.v4.new_code_cell(cell_code_pca),
    nbf.v4.new_code_cell(cell_code_cv),
    nbf.v4.new_code_cell(cell_code_summary),
]
nb.cells.extend(new_cells)
nbf.write(nb, p)
print(f'Notebook now has {len(nb.cells)} cells (§9 appended).')
