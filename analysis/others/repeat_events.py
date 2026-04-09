"""
Repeat Events Analysis
======================

162 companies in our sample have 2 or more layoff events during 2020-2024.
Including all events inflates the sample and violates the cross-sectional
independence assumption (AR from the same firm are correlated over time).

This script:
  1. Flags each event as "first" (event_sequence=1) or "subsequent" (>1) per company.
  2. Compares mean CARs between first and subsequent events:
     - Hypothesis: first layoff announcement is genuine news → larger |CAR|.
     - Subsequent events partially anticipated → smaller reaction.
  3. Runs the main CAAR test on the first-event-only subsample (cleaner panel).
  4. Adds event_sequence as a control in cross-sectional OLS.
  5. Produces a figure comparing CAAR distributions for first vs. subsequent events.

Primary spec: US-only. CAR windows: [-1,+1] and [0,+20].

Outputs (data/results/robustness/):
  repeat_events_summary.csv    — mean CARs by event sequence
  repeat_events_cross_sec.csv  — cross-section OLS with event_sequence control
  fig_first_vs_repeat.png      — distribution comparison
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE        = '/Users/irmina/Documents/Claude/layoff_study'
CARS_PATH   = os.path.join(BASE, 'data/results/car_by_event.csv')   # original, has all event metadata
CARS_V2     = os.path.join(BASE, 'data/results/improved/final_labels_and_cars.csv')
MASTER_PATH = os.path.join(BASE, 'data/processed/master_events_final.csv')
OUT_DIR     = os.path.join(BASE, 'data/results/robustness')
os.makedirs(OUT_DIR, exist_ok=True)

BLUE  = '#2166ac'
RED   = '#b2182b'
GREY  = '#636363'
GREEN = '#1a9850'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})


def stars(p):
    if pd.isna(p): return ''
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data and assign event sequence
# ══════════════════════════════════════════════════════════════════════════════

def load_and_flag():
    # Use the enriched v2 file for CARs + tiered AI labels
    cars = pd.read_csv(CARS_V2)
    master = pd.read_csv(MASTER_PATH)

    cars['announcement_date'] = pd.to_datetime(cars['announcement_date'])
    master['announcement_date'] = pd.to_datetime(master['announcement_date'])

    # Merge in listing_region, layoff_count, layoff_pct, beta_mkt_ff4 from master + car_by_event
    # car_by_event.csv has beta_mkt_ff4 and other estimation stats
    try:
        car_orig = pd.read_csv(CARS_PATH)
        car_orig['announcement_date'] = pd.to_datetime(car_orig['announcement_date'])
        extra_cols = ['ticker', 'announcement_date', 'beta_mkt_ff4', 'listing_region',
                      'layoff_count', 'layoff_pct', 'industry']
        extra_cols = [c for c in extra_cols if c in car_orig.columns]
        cars = cars.merge(car_orig[extra_cols].drop_duplicates(subset=['ticker','announcement_date']),
                          on=['ticker','announcement_date'], how='left')
    except Exception as e:
        print(f'  Warning: could not merge car_by_event.csv: {e}')

    # Sort and assign event sequence per company
    cars = cars.sort_values(['ticker', 'announcement_date']).reset_index(drop=True)
    cars['event_sequence'] = cars.groupby('ticker').cumcount() + 1
    cars['is_first_event'] = (cars['event_sequence'] == 1).astype(int)

    print(f'Total events: {len(cars)}')
    print(f'  First events:      {cars["is_first_event"].sum()}  ({cars["is_first_event"].mean()*100:.1f}%)')
    print(f'  Subsequent events: {(1-cars["is_first_event"]).sum()}  ({(1-cars["is_first_event"]).mean()*100:.1f}%)')

    # Companies with repeat events
    repeat_cos = cars.groupby('ticker').size()
    print(f'\nCompanies with ≥2 events: {(repeat_cos >= 2).sum()}')
    print(f'  Max events per company: {repeat_cos.max()}')
    dist = repeat_cos.value_counts().sort_index()
    print(f'  Distribution: {dist.to_dict()}')

    return cars


# ══════════════════════════════════════════════════════════════════════════════
# 2. Compare mean CARs: first vs. subsequent events
# ══════════════════════════════════════════════════════════════════════════════

def compare_first_vs_subsequent(df):
    print('\n' + '=' * 65)
    print('ANALYSIS 1: First vs. Subsequent Events — Mean CARs')
    print('=' * 65)

    df_us = df[df['listing_region'] == 'US'].copy() if 'listing_region' in df.columns else df.copy()

    windows = {'CAR[-1,+1]': 'CAR_1_1', 'CAR[0,+20]': 'CAR_0_20'}
    rows = []

    print(f'\n  {"Group":<28} {"Window":<14} {"Mean CAR%":>10} {"Std":>8} {"N":>6} {"t-stat":>8} {"p":>8}')
    print(f'  {"─" * 80}')

    for wlabel, col in windows.items():
        if col not in df_us.columns:
            continue
        s_first = df_us[df_us['is_first_event'] == 1][col].dropna()
        s_subseq = df_us[df_us['is_first_event'] == 0][col].dropna()

        # Two-sample t-test
        if len(s_subseq) > 1:
            t, p = stats.ttest_ind(s_first, s_subseq, equal_var=False)
        else:
            t, p = np.nan, np.nan

        for label, s, seq in [('First events', s_first, 'first'),
                               ('Subsequent events', s_subseq, 'subsequent')]:
            print(f'  {label:<28} {wlabel:<14} {s.mean():>10.3f} {s.std():>8.3f} {len(s):>6}', end='')
            if label == 'Subsequent events':
                print(f' {t:>8.3f} {p:>8.4f} {stars(p)}')
            else:
                print()
            rows.append({'group': label, 'window': wlabel, 'mean_car': s.mean(),
                         'std_car': s.std(), 'N': len(s), 't_stat': t if label=='Subsequent events' else np.nan,
                         'p_val': p if label=='Subsequent events' else np.nan})

        # CAAR for first-only events
        caar_first = s_first.mean()
        se_first   = s_first.std() / np.sqrt(len(s_first))
        t_first    = caar_first / se_first if se_first > 0 else np.nan
        p_first    = 2 * (1 - stats.t.cdf(abs(t_first), df=len(s_first)-1)) if not np.isnan(t_first) else np.nan
        print(f'  {"  CAAR (first only, t-test)":<28} {wlabel:<14} {caar_first:>10.3f} '
              f'{se_first:>8.3f} {len(s_first):>6} {t_first:>8.3f} {p_first:>8.4f} {stars(p_first)}')
        print()

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CAAR: First-event-only vs. Full sample
# ══════════════════════════════════════════════════════════════════════════════

def caar_comparison(df):
    print('\n' + '=' * 65)
    print('ANALYSIS 2: CAAR — Full Sample vs. First-Event-Only (US)')
    print('=' * 65)

    df_us = df[df['listing_region'] == 'US'].copy() if 'listing_region' in df.columns else df.copy()
    df_first = df_us[df_us['is_first_event'] == 1].copy()

    windows = {'CAR[-1,+1]': 'CAR_1_1', 'CAR[0,+20]': 'CAR_0_20', 'CAR[-5,+30]': 'CAR_5_30'}
    rows = []

    print(f'\n  {"Sample":<25} {"Window":<14} {"CAAR%":>8} {"SE":>8} {"t":>8} {"p":>8} {"Sig":>4}')
    print(f'  {"─" * 75}')

    for sample_name, sample_df in [('Full US sample', df_us), ('First events only', df_first)]:
        for wlabel, col in windows.items():
            if col not in sample_df.columns:
                continue
            s = sample_df[col].dropna()
            if len(s) < 5:
                continue
            mean_c = s.mean()
            se_c   = s.std() / np.sqrt(len(s))
            t_c    = mean_c / se_c if se_c > 0 else np.nan
            p_c    = 2 * (1 - stats.t.cdf(abs(t_c), df=len(s)-1)) if not np.isnan(t_c) else np.nan
            print(f'  {sample_name:<25} {wlabel:<14} {mean_c:>8.3f} {se_c:>8.3f} '
                  f'{t_c:>8.3f} {p_c:>8.4f} {stars(p_c):>4}')
            rows.append({'sample': sample_name, 'window': wlabel, 'caar': mean_c,
                         'se': se_c, 't': t_c, 'p': p_c, 'N': len(s)})

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Cross-sectional OLS with event_sequence as control
# ══════════════════════════════════════════════════════════════════════════════

def cross_section_with_sequence(df):
    print('\n' + '=' * 65)
    print('ANALYSIS 3: Cross-Section OLS with Event Sequence Control')
    print('  H: later layoffs have smaller |CAR| — β(event_sequence) < 0')
    print('=' * 65)

    df_us = df[df['listing_region'] == 'US'].copy() if 'listing_region' in df.columns else df.copy()

    # Prepare controls
    df_us['post_chatgpt'] = df_us['post_chatgpt'].fillna(0).astype(int)
    df_us['ai_broad']     = df_us['ai_broad'].fillna(0).astype(int)
    df_us['ai_x_post']    = df_us['ai_broad'] * df_us['post_chatgpt']
    df_us['log_seq']      = np.log(df_us['event_sequence'])   # log-transform (diminishing returns)

    rows = []
    for wlabel, col in [('CAR[-1,+1]', 'CAR_1_1'), ('CAR[0,+20]', 'CAR_0_20')]:
        if col not in df_us.columns:
            continue
        y = df_us[col]

        # Spec A: just ai + post + interaction + event_sequence
        X_a = df_us[['ai_broad', 'post_chatgpt', 'ai_x_post', 'event_sequence']]
        mask = y.notna() & X_a.notna().all(axis=1)
        if mask.sum() < 30:
            continue
        Xa_c = sm.add_constant(X_a[mask], has_constant='add')
        m_a  = sm.OLS(y[mask], Xa_c).fit(cov_type='HC3')

        # Spec B: log(event_sequence)
        X_b = df_us[['ai_broad', 'post_chatgpt', 'ai_x_post', 'log_seq']]
        mask_b = y.notna() & X_b.notna().all(axis=1)
        Xb_c = sm.add_constant(X_b[mask_b], has_constant='add')
        m_b  = sm.OLS(y[mask_b], Xb_c).fit(cov_type='HC3')

        print(f'\n  Outcome: {wlabel}')
        print(f'  {"─" * 60}')
        print(f'  {"Variable":<25} {"Spec A (linear)":>18} {"Spec B (log)":>16}')
        print(f'  {"─" * 60}')

        for var, label in [('ai_broad','AI Broad'),('post_chatgpt','Post-ChatGPT'),
                            ('ai_x_post','AI × Post'), ('event_sequence','Event Sequence (linear)'),
                            ('log_seq','log(Event Sequence)'), ('const','Constant')]:
            ca = m_a.params.get(var, np.nan)
            pa = m_a.pvalues.get(var, np.nan)
            cb = m_b.params.get(var, np.nan)
            pb = m_b.pvalues.get(var, np.nan)
            sa = f'{ca:.3f}{stars(pa)}' if not np.isnan(ca) else '—'
            sb = f'{cb:.3f}{stars(pb)}' if not np.isnan(cb) else '—'
            print(f'  {label:<25} {sa:>18} {sb:>16}')
            rows.append({'outcome': wlabel, 'variable': var,
                         'coef_linear': ca, 'pval_linear': pa,
                         'coef_log': cb, 'pval_log': pb})

        print(f'  {"N":<25} {mask.sum():>18} {mask_b.sum():>16}')
        print(f'  {"R²":<25} {m_a.rsquared:>18.3f} {m_b.rsquared:>16.3f}')

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Figure: distribution comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_first_vs_repeat(df):
    df_us = df[df['listing_region'] == 'US'].copy() if 'listing_region' in df.columns else df.copy()
    first  = df_us[df_us['is_first_event'] == 1]['CAR_1_1'].dropna()
    subseq = df_us[df_us['is_first_event'] == 0]['CAR_1_1'].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Histogram
    ax = axes[0]
    bins = np.linspace(-15, 15, 40)
    ax.hist(first.clip(-15, 15),  bins=bins, alpha=0.6, color=BLUE,
            label=f'First events (N={len(first)})',  density=True)
    ax.hist(subseq.clip(-15, 15), bins=bins, alpha=0.6, color=RED,
            label=f'Subsequent (N={len(subseq)})', density=True)
    ax.axvline(first.mean(),  color=BLUE, linewidth=1.8, linestyle='--',
               label=f'Mean first = {first.mean():.2f}%')
    ax.axvline(subseq.mean(), color=RED,  linewidth=1.8, linestyle='--',
               label=f'Mean subseq = {subseq.mean():.2f}%')
    ax.set_xlabel('CAR[−1,+1] (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution of CAR[−1,+1]\nFirst vs. Subsequent Events', fontsize=11)
    ax.legend(fontsize=9)

    # Panel 2: Box plot by event sequence (cap at seq=5)
    ax2 = axes[1]
    max_seq = min(int(df_us['event_sequence'].max()), 5)
    data_by_seq = []
    labels_seq  = []
    for s in range(1, max_seq + 1):
        sub = df_us[df_us['event_sequence'] == s]['CAR_1_1'].dropna()
        if len(sub) >= 3:
            data_by_seq.append(sub.values)
            labels_seq.append(f'#{s}\n(N={len(sub)})')

    bp = ax2.boxplot(data_by_seq, patch_artist=True, notch=False, widths=0.55,
                     flierprops=dict(marker='.', markersize=4, alpha=0.4))
    colors = [BLUE] + [RED] * (len(data_by_seq) - 1)
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    ax2.set_xticklabels(labels_seq, fontsize=9.5)
    ax2.set_xlabel('Event Sequence (within company)', fontsize=11)
    ax2.set_ylabel('CAR[−1,+1] (%)', fontsize=11)
    ax2.set_title('CAR[−1,+1] by Event Sequence\n(Blue = first event, Red = subsequent)', fontsize=11)

    fig.suptitle('Repeat Events Analysis — US-listed Companies\n'
                 'Hypothesis: first layoff announcement contains more new information',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_first_vs_repeat.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\n  Figure saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 65)
    print('REPEAT EVENTS ANALYSIS')
    print('=' * 65)

    df = load_and_flag()

    summary_df  = compare_first_vs_subsequent(df)
    caar_df     = caar_comparison(df)
    xsec_df     = cross_section_with_sequence(df)
    plot_first_vs_repeat(df)

    # Save
    combined = pd.concat([summary_df, caar_df], ignore_index=True, sort=False)
    combined.to_csv(os.path.join(OUT_DIR, 'repeat_events_summary.csv'), index=False)
    xsec_df.to_csv(os.path.join(OUT_DIR, 'repeat_events_cross_sec.csv'), index=False)

    # Save first-event-only data for reuse
    df_us_first = df[(df.get('listing_region', pd.Series(['US']*len(df))) == 'US') &
                     (df['is_first_event'] == 1)]
    df_us_first.to_csv(os.path.join(OUT_DIR, 'cars_first_event_only.csv'), index=False)

    print(f'\nAll outputs saved to: {OUT_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
