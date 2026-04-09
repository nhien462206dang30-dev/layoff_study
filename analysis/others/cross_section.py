"""
Phase 5: Cross-Sectional Regression Analysis
=============================================
Regresses individual event CARs on firm/event characteristics to explain
*why* some layoff announcements produce larger abnormal returns than others.

Dependent variables (FF4 model):
  CAR[-1,+1]   short-window announcement effect
  CAR[0,+20]   medium-window drift
  CAR[-5,+60]  long-run market reaction

Independent variables:
  ai_mentioned    1 if layoff announcement cited AI/automation
  post_genai      1 if announcement date >= 2023
  ai_x_post       interaction: ai_mentioned × post_genai
  log_count       log(1 + layoff_count)
  layoff_pct_n    percentage of workforce laid off (numeric)
  intl            1 if non-US listed stock
  beta_mkt        market beta from estimation window

Specifications run per outcome:
  (1) Baseline:   ai_mentioned + post_genai
  (2) Interaction: + ai_x_post
  (3) Controls:   + log_count + layoff_pct_n + beta_mkt
  (4) Full:       + intl + industry fixed effects

Output:
  data/results/cross_section_results.csv   — full coefficient table
  data/results/figures/fig8_reg_coefs.png  — coefficient plot
  data/results/figures/fig9_reg_table.png  — formatted regression table image
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

warnings.filterwarnings('ignore')

BASE        = '/Users/irmina/Documents/Claude/layoff_study'
RESULTS_DIR = os.path.join(BASE, 'data/results')
FIGS_DIR    = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

BLUE  = '#2166ac'
RED   = '#b2182b'
GREEN = '#1a9850'
GREY  = '#636363'

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load and prepare data
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'car_by_event.csv'))
    df['announcement_date'] = pd.to_datetime(df['announcement_date'])

    # Convert CAR columns to percentages
    for col in ['CAR_1_1', 'CAR_0_1', 'CAR_0_5', 'CAR_0_20', 'CAR_5_60']:
        df[col] = df[col] * 100

    # Dummies
    df['post_genai'] = (df['announcement_date'].dt.year >= 2023).astype(int)
    df['intl']       = (df['listing_region'] == 'INTL').astype(int)
    df['ai_x_post']  = df['ai_mentioned'] * df['post_genai']

    # layoff_pct: strip %, convert to float
    df['layoff_pct_n'] = (
        df['layoff_pct'].astype(str)
          .str.replace('%', '', regex=False)
          .str.strip()
    )
    df['layoff_pct_n'] = pd.to_numeric(df['layoff_pct_n'], errors='coerce')

    # log(1 + layoff_count)  — handles zeros/NaN gracefully
    df['log_count'] = np.log1p(df['layoff_count'].fillna(0))
    df.loc[df['layoff_count'].isna(), 'log_count'] = np.nan

    # Industry dummies (drop 'Other' as reference)
    industry_dummies = pd.get_dummies(df['industry'], prefix='ind', drop_first=False)
    if 'ind_Other' in industry_dummies.columns:
        industry_dummies = industry_dummies.drop(columns=['ind_Other'])
    industry_dummies = industry_dummies.astype(int)
    df = pd.concat([df, industry_dummies], axis=1)

    print(f'Loaded {len(df)} events')
    print(f'  ai_mentioned=1 : {df["ai_mentioned"].sum()}')
    print(f'  post_genai=1   : {df["post_genai"].sum()}')
    print(f'  ai_x_post=1    : {df["ai_x_post"].sum()}')
    print(f'  intl=1         : {df["intl"].sum()}')
    print(f'  layoff_count N : {df["layoff_count"].notna().sum()}')
    print(f'  layoff_pct_n N : {df["layoff_pct_n"].notna().sum()}')

    return df, industry_dummies.columns.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 2. OLS with robust (HC3) standard errors
# ══════════════════════════════════════════════════════════════════════════════

def ols_robust(y, X, df):
    """Run OLS with HC3 robust SEs. Returns fitted model."""
    mask = y.notna() & X.notna().all(axis=1)
    y_c  = y[mask]
    X_c  = sm.add_constant(X[mask], has_constant='add')
    model = sm.OLS(y_c, X_c).fit(cov_type='HC3')
    return model, mask.sum()


def stars(p):
    if pd.isna(p):   return ''
    if p < 0.01:     return '***'
    if p < 0.05:     return '**'
    if p < 0.10:     return '*'
    return ''


# ══════════════════════════════════════════════════════════════════════════════
# 3. Run all specifications
# ══════════════════════════════════════════════════════════════════════════════

def run_specifications(df, ind_cols):
    outcomes = {
        'CAR[-1,+1]':  'CAR_1_1',
        'CAR[0,+20]':  'CAR_0_20',
        'CAR[-5,+60]': 'CAR_5_60',
    }

    base_vars    = ['ai_mentioned', 'post_genai']
    interact_vars = base_vars + ['ai_x_post']
    control_vars  = interact_vars + ['log_count', 'layoff_pct_n', 'beta_mkt_ff4']
    full_vars     = control_vars + ['intl'] + ind_cols

    spec_defs = [
        ('(1) Baseline',    base_vars),
        ('(2) Interaction', interact_vars),
        ('(3) Controls',    control_vars),
        ('(4) Full + IFE',  full_vars),
    ]

    all_results = {}   # (outcome, spec) -> model
    all_n       = {}

    for out_label, out_col in outcomes.items():
        y = df[out_col]
        for spec_label, spec_vars in spec_defs:
            # Drop vars missing from df (safety)
            valid_vars = [v for v in spec_vars if v in df.columns]
            X = df[valid_vars]
            model, n = ols_robust(y, X, df)
            all_results[(out_label, spec_label)] = model
            all_n[(out_label, spec_label)] = n
            print(f'  {out_label} {spec_label}: N={n}, R²={model.rsquared:.3f}')

    return all_results, all_n


# ══════════════════════════════════════════════════════════════════════════════
# 4. Build coefficient summary table
# ══════════════════════════════════════════════════════════════════════════════

KEY_VARS = ['ai_mentioned', 'post_genai', 'ai_x_post',
            'log_count', 'layoff_pct_n', 'beta_mkt_ff4', 'intl']

VAR_LABELS = {
    'ai_mentioned':  'AI Mentioned (0/1)',
    'post_genai':    'Post-GenAI (0/1)',
    'ai_x_post':     'AI × Post-GenAI',
    'log_count':     'log(1 + Layoff Count)',
    'layoff_pct_n':  'Layoff % of Workforce',
    'beta_mkt_ff4':  'Market Beta (β)',
    'intl':          'International Listing (0/1)',
    'const':         'Constant',
}


def build_table(all_results, all_n):
    """Build a tidy DataFrame of coef/se/stars for all specs."""
    rows = []
    for (out_label, spec_label), model in all_results.items():
        params = model.params
        pvals  = model.pvalues
        ses    = model.bse
        for var in KEY_VARS + ['const']:
            if var in params.index:
                rows.append({
                    'outcome':    out_label,
                    'spec':       spec_label,
                    'variable':   var,
                    'var_label':  VAR_LABELS.get(var, var),
                    'coef':       params[var],
                    'se':         ses[var],
                    'pval':       pvals[var],
                    'stars':      stars(pvals[var]),
                    'N':          all_n[(out_label, spec_label)],
                    'R2':         model.rsquared,
                    'R2_adj':     model.rsquared_adj,
                })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Print formatted table to console
# ══════════════════════════════════════════════════════════════════════════════

def print_table(coef_df, all_results, all_n):
    outcomes = coef_df['outcome'].unique()
    specs    = coef_df['spec'].unique()

    print('\n' + '=' * 100)
    print('PHASE 5: CROSS-SECTIONAL REGRESSION RESULTS')
    print('OLS with HC3 Heteroskedasticity-Robust Standard Errors')
    print('=' * 100)

    for out in outcomes:
        sub = coef_df[coef_df['outcome'] == out]
        col_w = 20

        header = f'\nDependent variable: {out} (%)\n'
        header += '-' * (col_w + len(specs) * 22)
        print(header)

        spec_headers = ''.join([f'{s:>22}' for s in specs])
        print(f'{"Variable":<{col_w}}{spec_headers}')
        print('-' * (col_w + len(specs) * 22))

        for var in KEY_VARS + ['const']:
            label = VAR_LABELS.get(var, var)
            row_coef = f'{label:<{col_w}}'
            row_se   = f'{" ":<{col_w}}'
            for spec in specs:
                cell = sub[(sub['spec'] == spec) & (sub['variable'] == var)]
                if cell.empty:
                    row_coef += f'{"":>22}'
                    row_se   += f'{"":>22}'
                else:
                    c = cell.iloc[0]
                    row_coef += f'{c["coef"]:>18.3f}{c["stars"]:>4}'
                    se_str = f'({c["se"]:.3f})'
                    row_se   += f'{se_str:>22}'
            print(row_coef)
            print(row_se)

        print('-' * (col_w + len(specs) * 22))

        # N and R2
        n_row  = f'{"N":<{col_w}}'
        r2_row = f'{"R²":<{col_w}}'
        ra_row = f'{"Adj. R²":<{col_w}}'
        for spec in specs:
            cell = sub[sub['spec'] == spec]
            if cell.empty:
                n_row  += f'{"":>22}'
                r2_row += f'{"":>22}'
                ra_row += f'{"":>22}'
            else:
                c = cell.iloc[0]
                n_row  += f'{int(c["N"]):>22}'
                r2_row += f'{c["R2"]:>22.3f}'
                ra_row += f'{c["R2_adj"]:>22.3f}'
        print(n_row)
        print(r2_row)
        print(ra_row)
        print('Industry FE' + ''.join(['No'.rjust(22) if '(4)' not in s else 'Yes'.rjust(22) for s in specs]))

    print('\n*** p<0.01  ** p<0.05  * p<0.10  |  Robust SEs in parentheses (HC3)')
    print('Reference category for industry FE: "Other"')


# ══════════════════════════════════════════════════════════════════════════════
# 6. FIG 8 — Coefficient plot (key vars, Spec 3 and 4, all outcomes)
# ══════════════════════════════════════════════════════════════════════════════

def fig8_coef_plot(coef_df):
    print('\nFig 8: Coefficient plot...')

    outcomes  = ['CAR[-1,+1]', 'CAR[0,+20]', 'CAR[-5,+60]']
    plot_vars = ['ai_mentioned', 'post_genai', 'ai_x_post',
                 'log_count', 'layoff_pct_n', 'beta_mkt_ff4', 'intl']
    specs_to_plot = ['(3) Controls', '(4) Full + IFE']

    colors = {
        '(3) Controls':   BLUE,
        '(4) Full + IFE': RED,
    }
    offsets = {'(3) Controls': -0.15, '(4) Full + IFE': 0.15}

    fig, axes = plt.subplots(1, len(outcomes), figsize=(16, 7), sharey=True)

    for ax, out in zip(axes, outcomes):
        sub = coef_df[coef_df['outcome'] == out]

        ys = np.arange(len(plot_vars))
        for spec in specs_to_plot:
            spec_sub = sub[sub['spec'] == spec]
            coefs, lower, upper, y_pos = [], [], [], []
            for i, var in enumerate(plot_vars):
                row = spec_sub[spec_sub['variable'] == var]
                if row.empty:
                    continue
                r = row.iloc[0]
                coefs.append(r['coef'])
                lower.append(r['coef'] - 1.96 * r['se'])
                upper.append(r['coef'] + 1.96 * r['se'])
                y_pos.append(i + offsets[spec])

            ax.errorbar(
                coefs, y_pos,
                xerr=[np.array(coefs) - np.array(lower),
                      np.array(upper) - np.array(coefs)],
                fmt='o', color=colors[spec], capsize=4,
                markersize=6, linewidth=1.5, label=spec,
            )

        ax.axvline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.6)
        ax.set_title(f'Outcome: {out}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Coefficient Estimate (%)', fontsize=10)
        ax.set_yticks(ys)

    axes[0].set_yticklabels([VAR_LABELS.get(v, v) for v in plot_vars], fontsize=9.5)
    axes[0].set_ylabel('Regressor', fontsize=11)

    handles = [
        plt.Line2D([0], [0], marker='o', color=colors[s], label=s,
                   markersize=7, linewidth=1.5)
        for s in specs_to_plot
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle('Figure 8: Cross-Sectional Regression Coefficients\n'
                 'OLS with HC3 Robust SEs | Error bars = 95% CI',
                 fontsize=12, y=1.01)

    fig.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig8_regression_coefs.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIG 9 — Visual regression table (Spec 3 full, all three outcomes)
# ══════════════════════════════════════════════════════════════════════════════

def fig9_visual_table(coef_df):
    print('\nFig 9: Visual regression table...')

    outcomes  = ['CAR[-1,+1]', 'CAR[0,+20]', 'CAR[-5,+60]']
    spec      = '(3) Controls'
    plot_vars = ['ai_mentioned', 'post_genai', 'ai_x_post',
                 'log_count', 'layoff_pct_n', 'beta_mkt_ff4', 'intl', 'const']

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')

    col_labels = ['Variable'] + [f'{o}\n{spec}' for o in outcomes]
    n_rows = len(plot_vars) * 2 + 4   # coef + se rows + dividers + footer
    n_cols = len(col_labels)

    # Build cell data
    cell_data = []
    for var in plot_vars:
        coef_row = [VAR_LABELS.get(var, var)]
        se_row   = ['']
        for out in outcomes:
            sub = coef_df[(coef_df['outcome'] == out) &
                          (coef_df['spec']    == spec) &
                          (coef_df['variable'] == var)]
            if sub.empty:
                coef_row.append('—')
                se_row.append('')
            else:
                r = sub.iloc[0]
                coef_row.append(f'{r["coef"]:+.3f}{r["stars"]}')
                se_row.append(f'({r["se"]:.3f})')
        cell_data.append(coef_row)
        cell_data.append(se_row)

    # N and R2 rows
    n_row   = ['N']
    r2_row  = ['R²']
    ra_row  = ['Adj. R²']
    ife_row = ['Industry FE']
    for out in outcomes:
        sub = coef_df[(coef_df['outcome'] == out) & (coef_df['spec'] == spec)]
        if sub.empty:
            n_row.append(''); r2_row.append(''); ra_row.append(''); ife_row.append('')
        else:
            r = sub.iloc[0]
            n_row.append(str(int(r['N'])))
            r2_row.append(f'{r["R2"]:.3f}')
            ra_row.append(f'{r["R2_adj"]:.3f}')
            ife_row.append('No')
    cell_data.append(['─' * 20, '─' * 14, '─' * 14, '─' * 14])
    cell_data.extend([n_row, r2_row, ra_row, ife_row])

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    # Style header
    for j in range(n_cols):
        table[0, j].set_facecolor('#2166ac')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row shading and bold coef rows
    for i in range(1, len(cell_data) + 1):
        for j in range(n_cols):
            if i % 2 == 1:   # coef row
                table[i, j].set_facecolor('#f7f7f7')
                if j > 0:
                    table[i, j].set_text_props(fontweight='bold')
            else:             # se row
                table[i, j].set_facecolor('white')
                table[i, j].set_text_props(color=GREY, fontstyle='italic')

    ax.set_title('Figure 9: Cross-Sectional OLS Regression — Specification (3): Controls\n'
                 '*** p<0.01  ** p<0.05  * p<0.10  |  HC3 robust SEs in parentheses',
                 fontsize=11, pad=16, fontweight='normal')

    fig.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig9_regression_table.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 8. FIG 10 — AI × Post-GenAI interaction: predicted CAR visualization
# ══════════════════════════════════════════════════════════════════════════════

def fig10_interaction(df):
    print('\nFig 10: AI × Post-GenAI interaction plot...')

    outcomes = {
        'CAR[-1,+1]':  'CAR_1_1',
        'CAR[0,+20]':  'CAR_0_20',
        'CAR[-5,+60]': 'CAR_5_60',
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    groups = [
        ('Pre-GenAI\nAI not mentioned',  0, 0, BLUE,   '-'),
        ('Pre-GenAI\nAI mentioned',      1, 0, BLUE,   '--'),
        ('Post-GenAI\nAI not mentioned', 0, 1, RED,    '-'),
        ('Post-GenAI\nAI mentioned',     1, 1, RED,    '--'),
    ]

    for ax, (out_label, out_col) in zip(axes, outcomes.items()):
        means, sems, labels, colors_g, hatch_g = [], [], [], [], []

        for glabel, ai, post, col, ls in groups:
            sub = df[(df['ai_mentioned'] == ai) & (df['post_genai'] == post)][out_col].dropna()
            if len(sub) < 3:
                continue
            means.append(sub.mean())
            sems.append(sub.sem())
            labels.append(glabel)
            colors_g.append(col)
            hatch_g.append('///' if ai == 1 else '')

        xs = np.arange(len(means))
        bars = ax.bar(xs, means, yerr=[1.96 * s for s in sems],
                      color=colors_g, alpha=0.75, width=0.55,
                      error_kw=dict(ecolor='grey', capsize=5, linewidth=1.2),
                      edgecolor='white')
        for bar, hatch in zip(bars, hatch_g):
            bar.set_hatch(hatch)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8, ha='center')
        ax.set_title(f'{out_label}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean CAR (%)', fontsize=10)

        # Add N labels
        for i, (glabel, ai, post, col, ls) in enumerate(groups):
            sub = df[(df['ai_mentioned'] == ai) & (df['post_genai'] == post)][out_col].dropna()
            if len(sub) < 3:
                continue
            ax.text(i, ax.get_ylim()[0] * 0.95, f'N={len(sub)}',
                    ha='center', fontsize=7.5, color=GREY)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=BLUE,  alpha=0.75, label='Pre-GenAI (≤ 2022)'),
        mpatches.Patch(facecolor=RED,   alpha=0.75, label='Post-GenAI (≥ 2023)'),
        mpatches.Patch(facecolor='grey', alpha=0.5, hatch='///', label='AI Mentioned'),
        mpatches.Patch(facecolor='grey', alpha=0.5, label='AI Not Mentioned'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=9.5, frameon=True, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle('Figure 10: Mean CAR by AI Mention × GenAI Period\n'
                 'Error bars = 95% CI  |  Hatching = AI mentioned in announcement',
                 fontsize=12, y=1.01)

    fig.tight_layout()
    path = os.path.join(FIGS_DIR, 'fig10_ai_post_interaction.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 72)
    print('PHASE 5: CROSS-SECTIONAL REGRESSION ANALYSIS')
    print('=' * 72)

    df, ind_cols = load_and_prepare()

    print('\n── Running OLS specifications ──')
    all_results, all_n = run_specifications(df, ind_cols)

    coef_df = build_table(all_results, all_n)
    coef_df.to_csv(os.path.join(RESULTS_DIR, 'cross_section_results.csv'), index=False)
    print(f'\nFull results saved → {os.path.join(RESULTS_DIR, "cross_section_results.csv")}')

    print_table(coef_df, all_results, all_n)

    print('\n── Generating figures ──')
    fig8_coef_plot(coef_df)
    fig9_visual_table(coef_df)
    fig10_interaction(df)

    print('\n' + '=' * 72)
    print(f'Phase 5 complete. Figures saved to: {FIGS_DIR}')
    print('=' * 72)


if __name__ == '__main__':
    main()
