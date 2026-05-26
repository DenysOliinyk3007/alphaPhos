"""Helper: add §7 cells — full downstream pipeline (filter, impute, limma)
on two versions (aphos_sum_classI vs sn_classI) and compare biology.
"""
import nbformat as nbf
from pathlib import Path

p = Path("D:/Projects/alphaPhos/docs/benchmark/benchmark.ipynb")
nb = nbf.read(p, as_version=4)


# ──────────────────────────────────────────────────────────────────────────
# §7 markdown
# ──────────────────────────────────────────────────────────────────────────
cell_md = """## §7 · Full downstream pipeline — `aphos_sum_classI` vs `sn_classI`

We have the two best-matching versions:

- **`aphos_only_MS2_sum_classI`** — alphaPhos MS2 + sum aggregation + condition-aware classI
- **`sn_classI`** — Spectronaut native PTM Class-I site report

Question: do they capture the same biology when we run them through the lab's
standard downstream pipeline?

The pipeline is:

1. **Valid-values filter** — keep sites with ≥70% non-NaN replicates in at
   least one condition (lab convention)
2. **KNN imputation** — fill the remaining sparse NaNs
3. **limma** — withEGF vs woEGF contrast (3 reps each), Rscript subprocess
4. **Compare** — significant-hit overlap, logFC Pearson r on shared features,
   top hits per version

If the two versions capture the same biology we expect Jaccard of significant
hits > 0.85 and Pearson r on logFC > 0.95.
"""


# ──────────────────────────────────────────────────────────────────────────
# Cell B: imports + reshape both versions to samples-as-rows + condition col
# ──────────────────────────────────────────────────────────────────────────
cell_code_setup = """import sys
sys.path.insert(0, r'D:/Projects/Dublin/testscripts/src')
import importlib
# Dublin core.py provides filter_phosphosites + impute_phosphosites
import core as dublin_core
importlib.reload(dublin_core)
from core import filter_phosphosites, impute_phosphosites

# Two versions to compare. Both are already site-level wide matrices.
APHOS_KEY = 'aphos_only_MS2_sum_classI'
SN_KEY    = 'sn_classI'

# Load the alphaphos version (it was generated in §6.5 with aggregation=sum)
aphos_sites = pd.read_parquet(OUT / f'aphos_only_MS2_sum_classI.parquet')
# sn_classI_sites is already in memory from §4

# Bring alphaphos to the same key format we used for comparison in §5
# But the downstream pipeline expects a wide DF with PTM_Collapse_key column,
# so we keep the original alphaphos shape for the filter/impute path.

print(f'aphos sites: {aphos_sites.shape} (rows=sites, cols=samples+meta)')
print(f'sn_classI:   {sn_classI_sites.shape}')

# For the pipeline we need: samples-as-rows, sites-as-cols + a 'condition' col.
# Build the standardized wide-by-sample form for each version.
sample_cols = condition_df['sample'].tolist()
s2c_map = dict(zip(condition_df['sample'], condition_df['condition']))


def to_samples_as_rows(sites_df, key_col=None):
    \"\"\"Wide DataFrame: rows=samples, cols=sites + 'condition'.\"\"\"
    if key_col is not None:
        # alphaphos: site identifier is in column 'PTM_Collapse_key'
        block = sites_df.set_index(key_col)[sample_cols].T
    else:
        # SN: site identifier is already the index (string)
        block = sites_df.reindex(columns=sample_cols).T
    block.index.name = 'sample'
    block['condition'] = block.index.map(s2c_map)
    return block


aphos_wide = to_samples_as_rows(aphos_sites, key_col='PTM_Collapse_key')
sn_wide    = to_samples_as_rows(sn_classI_sites, key_col=None)

print(f'\\naphos_wide: {aphos_wide.shape}   (rows=samples, cols=sites + condition)')
print(f'sn_wide:    {sn_wide.shape}')
print()
print(aphos_wide[['condition']].head())
"""


# ──────────────────────────────────────────────────────────────────────────
# Cell C: valid-values filter + KNN imputation
# ──────────────────────────────────────────────────────────────────────────
cell_code_filter_impute = """# 1) Valid-values filter (≥70% present in at least one condition)
# 2) KNN imputation
# Dublin's filter_phosphosites identifies sites by the presence of '~' in the
# column name (alphaphos format). The SN-classI columns are like 'P12345|S|10|1'
# — we need to make them filterable by either tagging them or adapting the call.

# Dublin's filter_phosphosites looks at columns whose name contains '~'.
# For SN we replace '|' with '~' in the column names so the filter recognises
# them as site columns. We restore later if needed.

def normalize_site_col_names(df):
    rename = {c: c.replace('|', '~') for c in df.columns if '|' in str(c)}
    return df.rename(columns=rename)

aphos_wide_n = aphos_wide  # already has '~' in keys
sn_wide_n    = normalize_site_col_names(sn_wide)

print('aphos: sample of site-col names:', [c for c in aphos_wide_n.columns if '~' in str(c)][:2])
print('sn:    sample of site-col names:', [c for c in sn_wide_n.columns if '~' in str(c)][:2])

print('\\nFiltering aphos ...')
aphos_filtered = filter_phosphosites(aphos_wide_n, how='condition', cutoff=0.7, condition_col='condition')
n_aphos_kept = sum('~' in str(c) for c in aphos_filtered.columns)
print(f'  aphos sites kept: {n_aphos_kept:,} (of {sum(\"~\" in str(c) for c in aphos_wide_n.columns):,})')

print('\\nFiltering sn ...')
sn_filtered = filter_phosphosites(sn_wide_n, how='condition', cutoff=0.7, condition_col='condition')
n_sn_kept = sum('~' in str(c) for c in sn_filtered.columns)
print(f'  sn sites kept:    {n_sn_kept:,} (of {sum(\"~\" in str(c) for c in sn_wide_n.columns):,})')

print('\\nImputing aphos ...')
aphos_imputed = impute_phosphosites(aphos_filtered)
print(f'  aphos_imputed: {aphos_imputed.shape}')

print('Imputing sn ...')
sn_imputed = impute_phosphosites(sn_filtered)
print(f'  sn_imputed:    {sn_imputed.shape}')
"""


# ──────────────────────────────────────────────────────────────────────────
# Cell D: limma via Rscript subprocess
# ──────────────────────────────────────────────────────────────────────────
cell_code_limma = """import subprocess

LIMMA_WRAPPER = Path(r'D:/Projects/alphaPhos/docs/benchmark/limma_wrapper.R')
RSCRIPT       = Path(r'C:/Program Files/R/R-4.5.2/bin/Rscript.exe')

LIMMA_OUT = OUT / 'limma'
LIMMA_OUT.mkdir(exist_ok=True)


def run_limma_via_rscript(imputed_wide, label):
    \"\"\"
    imputed_wide: samples-as-rows DataFrame with a 'condition' column.
    Builds a features x samples expression matrix and a meta TSV,
    invokes the Rscript limma wrapper, returns the parsed topTable DataFrame.
    \"\"\"
    site_cols = [c for c in imputed_wide.columns if '~' in str(c)]
    expr = imputed_wide[site_cols].T   # rows=features, cols=samples
    expr.index.name = 'feature'

    meta = pd.DataFrame({
        'sample_id': imputed_wide.index,
        'treatment': imputed_wide['condition'].values,
    })

    expr_path = LIMMA_OUT / f'{label}_expr.tsv'
    meta_path = LIMMA_OUT / f'{label}_meta.tsv'
    out_path  = LIMMA_OUT / f'{label}_topTable.tsv'

    expr.to_csv(expr_path, sep=chr(9), na_rep='NA')
    meta.to_csv(meta_path, sep=chr(9), index=False)

    cmd = [str(RSCRIPT), str(LIMMA_WRAPPER), str(expr_path), str(meta_path),
           'treatment', str(out_path)]
    print(f'  [{label}] running limma ...')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print('  STDERR:', result.stderr[-500:])
        raise RuntimeError(f'limma failed for {label}')

    res = pd.read_csv(out_path, sep=chr(9))
    P_CUT, LFC_CUT = 0.05, 0.585
    res['significant'] = (res['adj.P.Val'] < P_CUT) & (res['logFC'].abs() > LFC_CUT)
    return res


print('==== limma: aphos_sum_classI ====')
limma_aphos = run_limma_via_rscript(aphos_imputed, 'aphos_sum_classI')
print(f'  features: {len(limma_aphos):,}')
print(f'  significant (adj.P<0.05 & |logFC|>0.585): {int(limma_aphos[\"significant\"].sum())}')
print(f'  raw P<0.05: {int((limma_aphos[\"P.Value\"] < 0.05).sum())}')

print('\\n==== limma: sn_classI ====')
limma_sn = run_limma_via_rscript(sn_imputed, 'sn_classI')
print(f'  features: {len(limma_sn):,}')
print(f'  significant (adj.P<0.05 & |logFC|>0.585): {int(limma_sn[\"significant\"].sum())}')
print(f'  raw P<0.05: {int((limma_sn[\"P.Value\"] < 0.05).sum())}')
"""


# ──────────────────────────────────────────────────────────────────────────
# Cell E: biological-overlap comparison
# ──────────────────────────────────────────────────────────────────────────
cell_code_compare = """# Compare the two limma results: significant-hit overlap + logFC concordance

# To match cross-version, normalize feature IDs to a common scheme. alphaphos
# features are 'P12345;Q67890~GENE_S15_M1', SN features are 'P12345~S~15~1'.
# We collapse both to (first_protein, AA, position, mult) tuples.

def alpha_feat_to_matchkey(f):
    s = str(f)
    if '~' not in s:
        return None
    # alphaphos style: 'P12345;Q67890~GENE_S15_M1'
    head, tail = s.split('~', 1)
    first_prot = head.split(';')[0]
    # GENE_S15_M1
    m = re.match(r'^[^_]*_([A-Z])(\\d+)_M(\\d+)$', tail)
    if not m:
        return None
    return f'{first_prot}|{m.group(1)}|{int(m.group(2))}|{int(m.group(3))}'


def sn_feat_to_matchkey(f):
    s = str(f)
    if s.count('~') == 3:                      # SN style after our '|' -> '~' rename
        prot, aa, pos, mult = s.split('~')
        return f'{prot}|{aa}|{int(pos)}|{int(mult)}'
    if s.count('|') == 3:
        return s
    return None


limma_aphos['match_key'] = limma_aphos['feature'].map(alpha_feat_to_matchkey)
limma_sn['match_key']    = limma_sn['feature'].map(sn_feat_to_matchkey)

n_a_unmapped = int(limma_aphos['match_key'].isna().sum())
n_s_unmapped = int(limma_sn['match_key'].isna().sum())
print(f'unmapped feature ids: aphos={n_a_unmapped}, sn={n_s_unmapped}')

# Significant-hit overlap
sig_a = set(limma_aphos.loc[limma_aphos['significant'], 'match_key'].dropna())
sig_s = set(limma_sn.loc[limma_sn['significant'], 'match_key'].dropna())
shared_sig  = sig_a & sig_s
aphos_only  = sig_a - sig_s
sn_only     = sig_s - sig_a
jaccard_sig = len(shared_sig) / max(1, len(sig_a | sig_s))

print()
print(f'Significant hits (adj.P<0.05 & |logFC|>0.585):')
print(f'  aphos sig:      {len(sig_a):>5,}')
print(f'  sn sig:         {len(sig_s):>5,}')
print(f'  shared:         {len(shared_sig):>5,}')
print(f'  aphos-only:     {len(aphos_only):>5,}')
print(f'  sn-only:        {len(sn_only):>5,}')
print(f'  Jaccard:        {jaccard_sig:.3f}')

# logFC concordance on all features in BOTH limma tables
lfc = (limma_aphos[['match_key', 'logFC', 'P.Value', 'adj.P.Val']]
       .merge(limma_sn[['match_key', 'logFC', 'P.Value', 'adj.P.Val']],
              on='match_key', suffixes=('_aphos', '_sn'))
       .dropna(subset=['logFC_aphos', 'logFC_sn']))

pear = float(lfc['logFC_aphos'].corr(lfc['logFC_sn']))
spear = float(lfc['logFC_aphos'].corr(lfc['logFC_sn'], method='spearman'))
print()
print(f'logFC concordance on {len(lfc):,} shared features:')
print(f'  Pearson r:  {pear:.4f}')
print(f'  Spearman r: {spear:.4f}')
print(f'  median |diff|: {(lfc[\"logFC_aphos\"] - lfc[\"logFC_sn\"]).abs().median():.4f}')

# Top-15 hits per version, side by side
top_a = (limma_aphos[limma_aphos['significant']]
         .nsmallest(15, 'adj.P.Val')[['match_key', 'logFC', 'adj.P.Val']]
         .reset_index(drop=True))
top_a.columns = ['match_key', 'aphos_logFC', 'aphos_adjP']

top_s = (limma_sn[limma_sn['significant']]
         .nsmallest(15, 'adj.P.Val')[['match_key', 'logFC', 'adj.P.Val']]
         .reset_index(drop=True))
top_s.columns = ['match_key', 'sn_logFC', 'sn_adjP']

print()
print('Top 15 hits per version:')
print(pd.concat({'aphos': top_a, 'sn': top_s}, axis=1).to_string())
"""


# ──────────────────────────────────────────────────────────────────────────
# Cell F: visualization — logFC scatter + volcano
# ──────────────────────────────────────────────────────────────────────────
cell_code_viz = """import plotly.express as px
import plotly.graph_objects as go

# Mark each shared feature by joint significance class for color
def _class(row):
    is_a = row['adj.P.Val_aphos'] < 0.05 and abs(row['logFC_aphos']) > 0.585
    is_s = row['adj.P.Val_sn']    < 0.05 and abs(row['logFC_sn'])    > 0.585
    if is_a and is_s: return 'both_sig'
    if is_a:           return 'aphos_only'
    if is_s:           return 'sn_only'
    return 'not_sig'

lfc['class'] = lfc.apply(_class, axis=1)
print(lfc['class'].value_counts())

# 1) logFC vs logFC scatter
fig1 = px.scatter(
    lfc, x='logFC_sn', y='logFC_aphos', color='class',
    color_discrete_map={'both_sig': '#e60000', 'aphos_only': '#1f77b4',
                        'sn_only': '#ff7f0e',  'not_sig':   '#cccccc'},
    opacity=0.5,
    title=f'logFC concordance — Pearson r = {pear:.4f}',
    labels={'logFC_sn': 'sn_classI logFC (withEGF - woEGF)',
            'logFC_aphos': 'alphaphos+sum logFC (withEGF - woEGF)'},
    hover_data={'match_key': True, 'adj.P.Val_aphos': ':.3g', 'adj.P.Val_sn': ':.3g'},
)
mn, mx = float(lfc[['logFC_aphos','logFC_sn']].min().min()), float(lfc[['logFC_aphos','logFC_sn']].max().max())
fig1.add_shape(type='line', x0=mn, x1=mx, y0=mn, y1=mx, line=dict(dash='dash', color='black', width=1))
fig1.update_layout(height=550, width=750, template='plotly_white')
fig1.show()

# 2) Volcano for each version side-by-side
fig2 = go.Figure()
for label, df, color in [('aphos+sum', limma_aphos, '#e60000'), ('sn_classI', limma_sn, '#1f77b4')]:
    d = df.dropna(subset=['logFC', 'adj.P.Val']).copy()
    d['nlog10p'] = -np.log10(d['adj.P.Val'].clip(lower=1e-300))
    d['is_sig'] = d['significant']
    fig2.add_trace(go.Scatter(
        x=d['logFC'], y=d['nlog10p'],
        mode='markers', name=label,
        marker=dict(size=5, color=color,
                    opacity=np.where(d['is_sig'], 0.85, 0.18)),
        hovertext=d['feature'], hoverinfo='text+x+y',
    ))
fig2.add_hline(y=-np.log10(0.05), line_dash='dash', line_color='black')
fig2.add_vline(x=0.585,  line_dash='dot',  line_color='black')
fig2.add_vline(x=-0.585, line_dash='dot',  line_color='black')
fig2.update_layout(
    title='Volcano — withEGF vs woEGF', xaxis_title='logFC',
    yaxis_title='-log10(adj.P.Val)',
    height=500, width=900, template='plotly_white',
)
fig2.show()
"""


# Find end of notebook (or right before any future section)
insert_at = len(nb.cells)
new_cells = [
    nbf.v4.new_markdown_cell(cell_md),
    nbf.v4.new_code_cell(cell_code_setup),
    nbf.v4.new_code_cell(cell_code_filter_impute),
    nbf.v4.new_code_cell(cell_code_limma),
    nbf.v4.new_code_cell(cell_code_compare),
    nbf.v4.new_code_cell(cell_code_viz),
]
nb.cells.extend(new_cells)

nbf.write(nb, p)
print(f'Notebook now has {len(nb.cells)} cells.')
print(f'§7 starts at index {insert_at} (markdown) + 5 code cells.')
