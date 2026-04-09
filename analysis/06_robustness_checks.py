"""
Robustness Checks for Tech Layoff Event Study
==============================================

Two analyses addressing key threats to validity:

1. PLACEBO DID TEST (Section A)
   Re-estimate the DID regression at 5 fake breakpoints + 1 real one.
   If the ChatGPT launch is the true driver of a differential AI-layoff reaction,
   β3 (ai × post) should be meaningfully larger at the real breakpoint (2022-11-30)
   than at arbitrary fake dates.

2. PAYWALL ATTENUATION BOUND ANALYSIS (Section B)
   42% of layoff news articles are behind paywalls → returned empty text → coded
   ai_mentioned=0 by default (measurement error, systematic downward bias).
   We sweep over assumed AI-mention rates among paywalled events (0% → 50%) and
   re-run the DID each time, showing how β3 moves as the true rate rises.

Primary specification: US-listed stocks only (US FF4 factors are misspecified
for international stocks). Full sample shown in appendix rows.

Outputs (saved to data/results/robustness/):
  placebo_did_results.csv      — β3 for each breakpoint
  fig_placebo_did.png          — coefficient plot with 95% CI
  paywall_sensitivity.csv      — β3 for each assumed paywall AI rate
  fig_paywall_sensitivity.png  — sensitivity curve
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = '/Users/irmina/Documents/Claude/layoff_study'
CARS_PATH   = os.path.join(BASE, 'data/results/improved/final_labels_and_cars.csv')
MASTER_PATH = os.path.join(BASE, 'data/processed/master_events_final.csv')
OUT_DIR     = os.path.join(BASE, 'data/results/robustness')
os.makedirs(OUT_DIR, exist_ok=True)

BLUE  = '#2166ac'
RED   = '#b2182b'
GREEN = '#1a9850'
GREY  = '#636363'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})

REAL_BREAKPOINT = pd.Timestamp('2022-11-30')
PLACEBO_DATES = [
    pd.Timestamp('2021-06-01'),
    pd.Timestamp('2022-01-01'),
    pd.Timestamp('2022-06-01'),
    REAL_BREAKPOINT,            # the real one — included in same series for comparison
    pd.Timestamp('2023-06-01'),
    pd.Timestamp('2023-12-01'),
]
PLACEBO_LABELS = [
    '2021-06-01\n(Placebo)',
    '2022-01-01\n(Placebo)',
    '2022-06-01\n(Placebo)',
    '2022-11-30\n(ChatGPT ★)',
    '2023-06-01\n(Placebo)',
    '2023-12-01\n(Placebo)',
]


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def stars(p):
    if pd.isna(p): return ''
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''


def ols_robust(y, X_df):
    """OLS with HC3 robust SEs. Returns (model, N) or (None, 0)."""
    mask = y.notna() & X_df.notna().all(axis=1)
    if mask.sum() < 30:
        return None, 0
    y_c = y[mask]
    X_c = sm.add_constant(X_df[mask], has_constant='add')
    model = sm.OLS(y_c, X_c).fit(cov_type='HC3')
    return model, int(mask.sum())


def load_data():
    """Load and merge car data with master metadata."""
    cars   = pd.read_csv(CARS_PATH)
    master = pd.read_csv(MASTER_PATH)

    cars['announcement_date'] = pd.to_datetime(cars['announcement_date'])
    master['announcement_date'] = pd.to_datetime(master['announcement_date'])

    # Bring in listing_region and ai_evidence from master
    meta = master[['ticker', 'announcement_date', 'listing_region', 'ai_evidence']].drop_duplicates(
        subset=['ticker', 'announcement_date'], keep='first'
    )
    df = cars.merge(meta, on=['ticker', 'announcement_date'], how='left')

    # CARs in final_labels_and_cars.csv are already stored in percent form
    for col in ['CAR_1_1', 'CAR_0_20', 'CAR_5_30']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Controls
    df['log_count'] = np.nan   # not available in this file — omit from controls
    df['beta_mkt']  = np.nan   # not available here — omit

    print(f'Data loaded: {len(df)} events  |  US: {(df["listing_region"]=="US").sum()}  '
          f'|  INTL: {(df["listing_region"]!="US").sum()}')
    print(f'ai_broad=1: {df["ai_broad"].sum()}  |  ai_broad=0: {(df["ai_broad"]==0).sum()}')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Placebo DID Tests
# ══════════════════════════════════════════════════════════════════════════════

def run_did_at_breakpoint(df, breakpoint_date, ai_col='ai_broad', y_col='CAR_1_1'):
    """
    Run DID: y = α + β1·ai + β2·post + β3·(ai×post) + ε
    Returns (β3, SE, p-value, N).
    """
    d = df.copy()
    d['post']    = (d['announcement_date'] >= breakpoint_date).astype(int)
    d['ai']      = d[ai_col].fillna(0).astype(int)
    d['ai_post'] = d['ai'] * d['post']

    X = d[['ai', 'post', 'ai_post']]
    y = d[y_col]
    model, n = ols_robust(y, X)
    if model is None:
        return np.nan, np.nan, np.nan, 0

    coef = model.params.get('ai_post', np.nan)
    se   = model.bse.get('ai_post', np.nan)
    pval = model.pvalues.get('ai_post', np.nan)
    return coef, se, pval, n


def section_a_placebo_did(df):
    print('\n' + '=' * 65)
    print('SECTION A: PLACEBO DID TESTS')
    print('  DID breakpoint swept from 2021 to 2024.')
    print('  Primary: US-only, ai_broad, CAR[-1,+1]')
    print('=' * 65)

    df_us = df[df['listing_region'] == 'US'].copy()

    rows_us   = []
    rows_full = []

    print(f'\n  {"Breakpoint":<14} {"Sample":<12} {"β3 (DID)":>10} {"SE":>8} '
          f'{"p":>8} {"Sig":>4} {"N":>6}')
    print(f'  {"─" * 65}')

    for date, label in zip(PLACEBO_DATES, PLACEBO_LABELS):
        label_short = str(date.date())
        is_real = (date == REAL_BREAKPOINT)

        for sample_name, sample_df, row_list in [('US only', df_us, rows_us),
                                                  ('Full',    df,    rows_full)]:
            coef, se, pval, n = run_did_at_breakpoint(sample_df, date)
            sig = stars(pval)
            marker = ' ← REAL' if is_real else ''
            if sample_name == 'US only':
                print(f'  {label_short:<14} {sample_name:<12} {coef:>10.3f} {se:>8.3f} '
                      f'{pval:>8.4f} {sig:>4} {n:>6}{marker}')
            row_list.append({
                'breakpoint': label_short,
                'label':      label.replace('\n', ' '),
                'is_real':    is_real,
                'beta3':      coef,
                'se':         se,
                'pval':       pval,
                'stars':      sig,
                'N':          n,
                'sample':     sample_name,
            })

    # Combine and save
    all_rows = rows_us + rows_full
    results_df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, 'placebo_did_results.csv')
    results_df.to_csv(out_csv, index=False)
    print(f'\n  Saved → {out_csv}')

    # Plot (US-only)
    _plot_placebo(rows_us, 'US only (Primary)')
    return results_df


def _plot_placebo(rows, title_suffix):
    fig, ax = plt.subplots(figsize=(10, 5))

    xs     = np.arange(len(rows))
    coefs  = [r['beta3'] for r in rows]
    ses    = [r['se'] for r in rows]
    labels = [PLACEBO_LABELS[i] for i in range(len(rows))]
    colors = [RED if r['is_real'] else BLUE for r in rows]

    ci_lo = [c - 1.96 * s for c, s in zip(coefs, ses)]
    ci_hi = [c + 1.96 * s for c, s in zip(coefs, ses)]

    for i, (x, c, lo, hi, col) in enumerate(zip(xs, coefs, ci_lo, ci_hi, colors)):
        ax.errorbar(x, c, yerr=[[c - lo], [hi - c]],
                    fmt='o', color=col, capsize=5, markersize=8, linewidth=1.8,
                    label='Real breakpoint' if rows[i]['is_real'] else ('Placebo' if i == 0 else ''))

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    # Highlight real breakpoint column
    real_idx = next(i for i, r in enumerate(rows) if r['is_real'])
    ax.axvspan(real_idx - 0.4, real_idx + 0.4, alpha=0.08, color=RED)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel('β₃ Coefficient (DID Estimator) — CAR[−1,+1] %', fontsize=11)
    ax.set_title(
        f'Placebo DID Test: β₃ (AI × Post) at Different Breakpoints\n'
        f'{title_suffix} | Error bars = 95% CI | Red = ChatGPT (real)',
        fontsize=11
    )

    handles = [
        plt.Line2D([0], [0], marker='o', color=RED,  markersize=8, label='Real breakpoint (ChatGPT)'),
        plt.Line2D([0], [0], marker='o', color=BLUE, markersize=8, label='Placebo breakpoints'),
    ]
    ax.legend(handles=handles, fontsize=10, frameon=True)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_placebo_did.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Paywall Attenuation Bound Analysis
# ══════════════════════════════════════════════════════════════════════════════

def section_b_paywall_bounds(df, n_sim=50):
    """
    42% of articles returned empty text (paywall) → coded ai=0.
    We identify these as events where ai_evidence is NaN in the master file.
    For each assumed true AI rate among paywalled events, we randomly re-assign
    ai=1 to that fraction and re-run the DID. We repeat n_sim times per rate
    (with different random seeds) and report mean β3 ± std.
    """
    print('\n' + '=' * 65)
    print('SECTION B: PAYWALL ATTENUATION BOUND ANALYSIS')
    print('  Assumption: events with missing ai_evidence = paywalled.')
    print('  We sweep assumed AI rate among these events from 0% to 50%.')
    print('  Primary: US-only, ai_broad base, CAR[-1,+1]')
    print('=' * 65)

    df_us = df[df['listing_region'] == 'US'].copy()

    # Identify paywalled events: ai_evidence is NaN AND ai_broad = 0
    # (if ai_broad=1, the article text was accessible and AI was found)
    paywalled_mask = df_us['ai_evidence'].isna() & (df_us['ai_broad'] == 0)
    n_paywalled = paywalled_mask.sum()
    # Observed AI rate: among events where we DID retrieve text (ai_evidence not null),
    # what fraction had ai_broad=1? This is the rate we'd expect among paywalled events.
    readable_mask  = df_us['ai_evidence'].notna()
    n_readable     = readable_mask.sum()
    n_readable_ai  = (readable_mask & (df_us['ai_broad'] == 1)).sum()
    # Fallback: use overall ai_broad rate if readable sample is too small
    if n_readable < 10:
        observed_rate = df_us['ai_broad'].mean()
    else:
        observed_rate = min(n_readable_ai / n_readable, 0.99)

    print(f'\n  US sample: {len(df_us)} events')
    print(f'  Paywalled (ai_ev=NaN, ai_broad=0): {n_paywalled}  ({n_paywalled/len(df_us)*100:.1f}%)')
    print(f'  Observed AI rate (readable articles): {n_readable_ai}/{n_readable} = {observed_rate*100:.1f}%')
    print(f'\n  Sweeping assumed paywall AI rate from 0% to 50% ({n_sim} simulations each)...')

    assumed_rates = [0.0, 0.05, 0.10, observed_rate, 0.20, 0.27, 0.30, 0.40, 0.50]
    rows = []

    print(f'\n  {"Rate":>6} {"Mean β3":>10} {"Std β3":>9} {"Mean p":>9} {"Sig @5%":>8}')
    print(f'  {"─" * 48}')

    for rate in assumed_rates:
        betas = []
        pvals = []
        for seed in range(n_sim):
            rng = np.random.default_rng(seed)
            d = df_us.copy()
            # Randomly flip ai_broad=1 for (rate × n_paywalled) paywalled events
            paywalled_idx = d[paywalled_mask].index.tolist()
            n_flip = min(int(round(rate * n_paywalled)), len(paywalled_idx))
            if n_flip > 0:
                flip_idx = rng.choice(paywalled_idx, size=n_flip, replace=False)
                d.loc[flip_idx, 'ai_broad'] = 1

            coef, se, pval, n = run_did_at_breakpoint(d, REAL_BREAKPOINT, ai_col='ai_broad')
            if not np.isnan(coef):
                betas.append(coef)
                pvals.append(pval)

        mean_b = np.mean(betas) if betas else np.nan
        std_b  = np.std(betas)  if betas else np.nan
        mean_p = np.mean(pvals) if pvals else np.nan
        sig_frac = np.mean([p < 0.05 for p in pvals]) if pvals else np.nan
        marker = ' ← observed rate' if abs(rate - observed_rate) < 0.005 else ''

        print(f'  {rate*100:>5.1f}% {mean_b:>10.3f} {std_b:>9.3f} {mean_p:>9.4f} '
              f'{sig_frac*100:>7.1f}%{marker}')

        rows.append({
            'assumed_paywall_ai_rate': rate,
            'mean_beta3': mean_b,
            'std_beta3':  std_b,
            'mean_pval':  mean_p,
            'pct_sig_5pct': sig_frac * 100,
            'n_sims':     len(betas),
        })

    results_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'paywall_sensitivity.csv')
    results_df.to_csv(out_csv, index=False)
    print(f'\n  Saved → {out_csv}')

    _plot_paywall(results_df, observed_rate)
    return results_df


def _plot_paywall(df, observed_rate):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    rates = df['assumed_paywall_ai_rate'] * 100
    mean_b = df['mean_beta3']
    std_b  = df['std_beta3']
    sig    = df['pct_sig_5pct']

    # Panel 1: β3 sensitivity curve
    ax = axes[0]
    ax.fill_between(rates, mean_b - std_b, mean_b + std_b, alpha=0.18, color=BLUE,
                    label='±1 SD across simulations')
    ax.plot(rates, mean_b, color=BLUE, linewidth=2, marker='o', markersize=6, label='Mean β₃')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    obs_pct = observed_rate * 100
    ax.axvline(obs_pct, color=GREEN, linewidth=1.5, linestyle=':', label=f'Observed rate ({obs_pct:.1f}%)')
    ax.set_xlabel('Assumed AI-mention rate among\npaywalled events (%)', fontsize=11)
    ax.set_ylabel('β₃ (DID Estimator) — CAR[−1,+1] %', fontsize=11)
    ax.set_title('Sensitivity of DID Estimate\nto Paywall Measurement Error', fontsize=11)
    ax.legend(fontsize=9.5, frameon=True)

    # Panel 2: % of simulations significant at 5%
    ax2 = axes[1]
    ax2.plot(rates, sig, color=RED, linewidth=2, marker='s', markersize=6)
    ax2.axhline(5, color='grey', linewidth=1, linestyle='--', alpha=0.6, label='5% level (nominal)')
    ax2.axvline(obs_pct, color=GREEN, linewidth=1.5, linestyle=':', label=f'Observed rate ({obs_pct:.1f}%)')
    ax2.set_xlabel('Assumed AI-mention rate among\npaywalled events (%)', fontsize=11)
    ax2.set_ylabel('% Simulations with p < 0.05 (%)', fontsize=11)
    ax2.set_title('Statistical Power vs.\nPaywall AI Rate', fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9.5, frameon=True)

    fig.suptitle(
        'Paywall Attenuation Bound Analysis (US-only, ai_broad, CAR[−1,+1])\n'
        f'Each point = mean over 50 random simulations of paywall re-assignment',
        fontsize=11, y=1.02
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_paywall_sensitivity.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Parallel Trends Test
# ══════════════════════════════════════════════════════════════════════════════

def section_c_parallel_trends():
    """
    DID validity check: parallel trends assumption.

    If the AI=1 and AI=0 groups had different pre-existing CAR trends before
    the ChatGPT launch, the DID estimate is invalid. We test this by plotting
    monthly average CAR[−1,+1] for each group across calendar time in the
    pre-ChatGPT period (2020–2022).

    Under parallel trends: the two lines should move up and down together
    (similar calendar-time fluctuations), even if their levels differ.

    Data: ar_panel_daily.csv → compute CAR[−1,+1] per event → assign to
    calendar month → plot monthly group averages with CI bands.
    """
    print('\n' + '=' * 65)
    print('SECTION C: PARALLEL TRENDS TEST (DID Validity)')
    print('  Pre-ChatGPT period: 2020-01 to 2022-11')
    print('  Groups: AI-mentioned (ai_mentioned=1) vs. Not (ai_mentioned=0)')
    print('=' * 65)

    AR_PATH  = os.path.join(BASE, 'data/results/improved/ar_panel_daily.csv')
    CARS_PATH_LOCAL = os.path.join(BASE, 'data/results/improved/final_labels_and_cars.csv')
    MASTER_PATH_LOCAL = os.path.join(BASE, 'data/processed/master_events_final.csv')

    panel = pd.read_csv(AR_PATH)
    panel['announcement_date'] = pd.to_datetime(panel['announcement_date'])

    # Merge listing_region
    master = pd.read_csv(MASTER_PATH_LOCAL)
    master['announcement_date'] = pd.to_datetime(master['announcement_date'])
    meta = master[['ticker','announcement_date','listing_region']].drop_duplicates(
        subset=['ticker','announcement_date'], keep='first')
    panel = panel.merge(meta, on=['ticker','announcement_date'], how='left')

    # Merge tiered ai_broad labels
    cars = pd.read_csv(CARS_PATH_LOCAL)
    cars['announcement_date'] = pd.to_datetime(cars['announcement_date'])
    if 'ai_broad' in cars.columns:
        panel = panel.merge(
            cars[['ticker','announcement_date','ai_broad']].drop_duplicates(
                subset=['ticker','announcement_date']),
            on=['ticker','announcement_date'], how='left')
        panel['ai_broad'] = panel['ai_broad'].fillna(panel['ai_mentioned']).fillna(0).astype(int)
    else:
        panel['ai_broad'] = panel['ai_mentioned'].fillna(0).astype(int)

    # Restrict to US, pre-ChatGPT
    panel_pre = panel[
        (panel['listing_region'] == 'US') &
        (panel['post_chatgpt'] == 0)
    ].copy()

    # Compute CAR[−1,+1] per event from the daily AR panel
    event_cars = []
    for ev_id, grp in panel_pre.groupby('event_id'):
        window = grp[(grp['t'] >= -1) & (grp['t'] <= 1)]
        if len(window) < 2:
            continue
        car = window['AR'].sum() * 100   # to percent
        ann_date = grp['announcement_date'].iloc[0]
        ai_val   = grp['ai_broad'].iloc[0]
        event_cars.append({
            'event_id':          ev_id,
            'announcement_date': ann_date,
            'year_month':        ann_date.to_period('M'),
            'CAR_1_1':           car,
            'ai':                ai_val,
        })

    ev_df = pd.DataFrame(event_cars)
    n_ai  = (ev_df['ai'] == 1).sum()
    n_nai = (ev_df['ai'] == 0).sum()
    print(f'\n  Pre-ChatGPT events used: {len(ev_df)}  (AI=1: {n_ai}, AI=0: {n_nai})')

    # Monthly average CAR by group
    def monthly_stats(sub):
        return sub.groupby('year_month')['CAR_1_1'].agg(['mean','std','count']).reset_index()

    monthly_ai  = monthly_stats(ev_df[ev_df['ai'] == 1])
    monthly_nai = monthly_stats(ev_df[ev_df['ai'] == 0])

    # Cumulative sum of monthly averages (to show trend direction over time)
    monthly_ai['cum_mean']  = monthly_ai['mean'].cumsum()
    monthly_nai['cum_mean'] = monthly_nai['mean'].cumsum()

    # Statistical test: are the two groups' monthly CARs correlated?
    # Merge on year_month and compute correlation
    merged_monthly = monthly_ai[['year_month','mean']].merge(
        monthly_nai[['year_month','mean']], on='year_month', suffixes=('_ai','_nai'))
    if len(merged_monthly) > 3:
        from scipy.stats import pearsonr
        r, p_r = pearsonr(merged_monthly['mean_ai'], merged_monthly['mean_nai'])
        print(f'\n  Monthly correlation between AI and non-AI group CARs:')
        print(f'  r = {r:.3f}  p = {p_r:.4f}  {"✓ Correlated (consistent with parallel trends)" if r > 0 and p_r < 0.1 else "⚠ Weak/no correlation — trends may diverge"}')
    else:
        r, p_r = np.nan, np.nan

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Monthly mean CAR (bar chart)
    ax = axes[0]
    months_ai  = [str(m) for m in monthly_ai['year_month']]
    months_nai = [str(m) for m in monthly_nai['year_month']]

    # Use only months present in both groups for clean comparison
    common = merged_monthly['year_month'].astype(str).tolist()
    ma_vals  = merged_monthly['mean_ai'].values
    nai_vals = merged_monthly['mean_nai'].values
    xs = np.arange(len(common))

    ax.bar(xs - 0.2, ma_vals,  width=0.38, color=RED,  alpha=0.65,
           label=f'AI mentioned (N={n_ai})')
    ax.bar(xs + 0.2, nai_vals, width=0.38, color=BLUE, alpha=0.65,
           label=f'AI not mentioned (N={n_nai})')
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--')

    # Only show every 3rd month label to avoid crowding
    tick_every = max(1, len(common) // 10)
    ax.set_xticks(xs[::tick_every])
    ax.set_xticklabels(common[::tick_every], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Monthly Mean CAR[−1,+1] (%)', fontsize=11)
    ax.set_title('Monthly Mean CAR: AI vs Non-AI\n(Pre-ChatGPT, US-only)', fontsize=10)
    ax.legend(fontsize=9.5)

    # Panel 2: Cumulative trend
    ax2 = axes[1]
    # Convert Period to timestamp for plotting
    ai_dates  = monthly_ai['year_month'].dt.to_timestamp()
    nai_dates = monthly_nai['year_month'].dt.to_timestamp()

    ax2.plot(ai_dates,  monthly_ai['cum_mean'],  color=RED,  linewidth=2,
             marker='o', markersize=4, label=f'AI mentioned (N={n_ai})')
    ax2.plot(nai_dates, monthly_nai['cum_mean'], color=BLUE, linewidth=2,
             marker='o', markersize=4, label=f'AI not mentioned (N={n_nai})')
    ax2.axhline(0, color='grey', linewidth=0.6, linestyle='--')
    ax2.set_ylabel('Cumulative Mean CAR (%)', fontsize=11)
    ax2.set_title(
        f'Cumulative CAR Trend (Pre-ChatGPT)\nr = {r:.2f}, p = {p_r:.3f} '
        f'— {"parallel" if not np.isnan(r) and r > 0.2 else "non-parallel"} trends',
        fontsize=10
    )
    ax2.legend(fontsize=9.5)

    n_months = len(merged_monthly)
    if not np.isnan(r) and r > 0.4:
        note = f'✓ r={r:.2f} — strong co-movement (N={n_months} months; parallel trends plausible)'
    elif not np.isnan(r) and r > 0:
        note = f'r={r:.2f} — modest co-movement (N={n_months} months; inconclusive)'
    else:
        note = f'⚠ r={r:.2f} — diverging trends (N={n_months} months; parallel trends questionable)'
    fig.text(0.5, -0.02, note, ha='center', fontsize=10,
             color=GREEN if '✓' in note else RED)

    fig.suptitle('Parallel Trends Test — DID Validity Check\n'
                 'Pre-ChatGPT period (2020–2022): AI vs Non-AI monthly CAR[−1,+1]',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_parallel_trends.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Figure saved → {path}')

    # Save monthly data
    merged_monthly.to_csv(os.path.join(OUT_DIR, 'parallel_trends_monthly.csv'), index=False)
    print(f'  Data saved → parallel_trends_monthly.csv')

    return {'r': r, 'p': p_r, 'n_months': len(merged_monthly)}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 65)
    print('ROBUSTNESS CHECKS: PLACEBO DID + PAYWALL + PARALLEL TRENDS')
    print('=' * 65)

    df = load_data()

    placebo_df  = section_a_placebo_did(df)
    paywall_df  = section_b_paywall_bounds(df)
    pt_result   = section_c_parallel_trends()

    print('\n' + '=' * 65)
    print('ROBUSTNESS SUMMARY')
    print('=' * 65)

    # Placebo summary
    us_rows = placebo_df[placebo_df['sample'] == 'US only']
    real_row    = us_rows[us_rows['is_real'] == True].iloc[0]
    placebo_rows = us_rows[us_rows['is_real'] == False]
    print(f'\nA. Placebo DID (US-only, ai_broad, CAR[-1,+1]):')
    print(f'   Real breakpoint (2022-11-30): β3 = {real_row["beta3"]:.3f}  '
          f'p = {real_row["pval"]:.4f}  {real_row["stars"]}')
    print(f'   Placebo β3 range: [{placebo_rows["beta3"].min():.3f}, {placebo_rows["beta3"].max():.3f}]')
    print(f'   Placebo p-values: [{placebo_rows["pval"].min():.3f}, {placebo_rows["pval"].max():.3f}]')
    if abs(real_row['beta3']) > placebo_rows['beta3'].abs().max():
        print(f'   ✓ Real β3 is LARGER than all placebo β3 — consistent with ChatGPT as driver.')
    else:
        print(f'   ⚠ Some placebo β3 values exceed real β3 — interpret DID result with caution.')

    # Paywall summary
    obs_row = paywall_df.iloc[(paywall_df['assumed_paywall_ai_rate'] - 0.158).abs().argsort()[:1]]
    max_row = paywall_df.iloc[-1]
    print(f'\nB. Paywall Attenuation (US-only, ai_broad, CAR[-1,+1]):')
    print(f'   At observed rate (~15.8%): mean β3 = {obs_row["mean_beta3"].values[0]:.3f}')
    print(f'   At 50% paywall AI rate:   mean β3 = {max_row["mean_beta3"]:.3f}')
    print(f'   Conclusion: {"β3 moves in consistent direction" if max_row["mean_beta3"] * obs_row["mean_beta3"].values[0] > 0 else "β3 changes sign — fragile result"}')

    # Parallel trends summary
    r_val = pt_result.get('r', np.nan)
    p_val = pt_result.get('p', np.nan)
    print(f'\nC. Parallel Trends (pre-ChatGPT, AI vs non-AI monthly CAR):')
    print(f'   Monthly correlation: r = {r_val:.3f}  p = {p_val:.4f}')
    if not np.isnan(r_val) and r_val > 0.2:
        print(f'   ✓ Positive correlation — AI and non-AI groups move together pre-ChatGPT.')
        print(f'     Parallel trends assumption is PLAUSIBLE.')
    else:
        print(f'   ⚠ Weak/no correlation — two groups may have diverged pre-ChatGPT.')
        print(f'     Parallel trends assumption is NOT strongly supported.')

    print(f'\nAll outputs saved to: {OUT_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
