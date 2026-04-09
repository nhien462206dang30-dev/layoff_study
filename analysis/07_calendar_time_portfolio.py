"""
Calendar-Time Portfolio Analysis
==================================

Standard event-study tests (Patell, BMP, Corrado) assume cross-sectional
independence of abnormal returns. When layoff announcements cluster in the
same calendar period (e.g., January 2023: Google, Microsoft, Amazon, Salesforce
all announced within days of each other), this assumption is violated and
test statistics are biased upward.

The calendar-time portfolio approach (Jaffe 1974, Fama 1998) corrects for
this by building monthly equal-weight portfolios of firms with active event
windows, then regressing the time series of portfolio excess returns on FF4
factors. The intercept α is the average monthly abnormal return; its t-statistic
automatically accounts for cross-event correlation within months.

Methodology:
  For each calendar month m:
    - Include all firms that have an active event window during month m.
    - Event window = [announcement_date − 5, announcement_date + 60].
    - Compute equal-weight portfolio return for month m as the average
      daily return across included firms, aggregated to monthly.
  Regress: R_portfolio,m − RF_m = α + β1·MKT_RF + β2·SMB + β3·HML + β4·MOM + ε
  The annualized abnormal return = α × 12.

Run for:
  (A) All events (full sample, US-only)
  (B) Pre-ChatGPT (announcement ≤ 2022-11-29)
  (C) Post-ChatGPT (announcement ≥ 2022-11-30)
  (D) Post-ChatGPT, AI-mentioned
  (E) Post-ChatGPT, AI-not-mentioned

Outputs (data/results/calendar_time/):
  ct_results.csv            — α, t-stat, p-val for each subsample
  fig_ct_portfolio.png      — monthly portfolio abnormal returns
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
MASTER_PATH = os.path.join(BASE, 'data/processed/master_events_final.csv')
FF_PATH     = os.path.join(BASE, 'data/processed/ff_factors.csv')
RETURNS_DIR = os.path.join(BASE, 'data/processed/stock_returns')
CARS_PATH   = os.path.join(BASE, 'data/results/improved/final_labels_and_cars.csv')
OUT_DIR     = os.path.join(BASE, 'data/results/calendar_time')
os.makedirs(OUT_DIR, exist_ok=True)

CHATGPT_DATE = pd.Timestamp('2022-11-30')
# Event window: [−5, +60] trading days ≈ calendar window [−7, +84] calendar days
EVENT_WINDOW_DAYS = (-7, 84)   # calendar days around announcement

BLUE  = '#2166ac'
RED   = '#b2182b'
GREEN = '#1a9850'
GREY  = '#636363'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})


def stars(p):
    if pd.isna(p): return ''
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load events and identify active firms per calendar day
# ══════════════════════════════════════════════════════════════════════════════

def load_events():
    master = pd.read_csv(MASTER_PATH)
    master['announcement_date'] = pd.to_datetime(master['announcement_date'])
    cars   = pd.read_csv(CARS_PATH)
    cars['announcement_date'] = pd.to_datetime(cars['announcement_date'])

    # Filter to US, event_study_usable
    events = master[
        (master['listing_region'] == 'US') &
        (master['event_study_usable'] == True)
    ].copy()

    # Merge tiered AI labels
    ai_cols = ['ticker', 'announcement_date', 'ai_broad', 'post_chatgpt']
    ai_cols = [c for c in ai_cols if c in cars.columns]
    events = events.merge(cars[ai_cols].drop_duplicates(subset=['ticker','announcement_date']),
                          on=['ticker','announcement_date'], how='left')
    events['ai_broad']    = events['ai_broad'].fillna(0).astype(int)
    events['post_chatgpt'] = (events['announcement_date'] >= CHATGPT_DATE).astype(int)

    print(f'Events loaded: {len(events)} US usable events')
    print(f'  Pre-ChatGPT:  {(events["post_chatgpt"]==0).sum()}')
    print(f'  Post-ChatGPT: {(events["post_chatgpt"]==1).sum()}')
    print(f'  Post-ChatGPT AI:     {((events["post_chatgpt"]==1) & (events["ai_broad"]==1)).sum()}')
    print(f'  Post-ChatGPT Non-AI: {((events["post_chatgpt"]==1) & (events["ai_broad"]==0)).sum()}')
    return events


def load_ff():
    ff = pd.read_csv(FF_PATH, parse_dates=['Date'], index_col='Date')
    ff = ff.rename(columns={'Mkt-RF': 'MKT_RF', 'Mom   ': 'MOM', 'Mom': 'MOM'})
    for col in ff.columns:
        if 'mom' in col.lower() and 'MOM' not in ff.columns:
            ff = ff.rename(columns={col: 'MOM'})
    ff.index = pd.to_datetime(ff.index)
    ff = ff.sort_index()
    # Convert from % to decimal if needed (Ken French files are in %)
    for col in ['MKT_RF', 'SMB', 'HML', 'MOM', 'RF']:
        if col in ff.columns and ff[col].abs().max() > 1:
            ff[col] = ff[col] / 100
    return ff


def load_daily_returns(ticker):
    path = os.path.join(RETURNS_DIR, f'{ticker}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    if ticker in df.columns:
        df = df.rename(columns={ticker: 'ret'})
    elif len(df.columns) == 1:
        df.columns = ['ret']
    if 'ret' not in df.columns:
        return None
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    return df[['ret']]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Build calendar-time portfolio monthly returns
# ══════════════════════════════════════════════════════════════════════════════

def build_monthly_portfolio(events_subset, ff):
    """
    For each calendar month in the sample, compute the equal-weight portfolio
    excess return of all firms with an active event window in that month.

    Returns a DataFrame with monthly portfolio returns and FF4 factors.
    """
    if len(events_subset) == 0:
        return pd.DataFrame()

    # Determine date range
    date_min = events_subset['announcement_date'].min() + pd.Timedelta(days=EVENT_WINDOW_DAYS[0]) - pd.Timedelta(days=30)
    date_max = events_subset['announcement_date'].max() + pd.Timedelta(days=EVENT_WINDOW_DAYS[1]) + pd.Timedelta(days=30)

    # Load returns for all tickers in this subset
    print(f'  Loading returns for {events_subset["ticker"].nunique()} tickers...')
    returns_cache = {}
    for ticker in events_subset['ticker'].unique():
        ret = load_daily_returns(ticker)
        if ret is not None:
            returns_cache[ticker] = ret

    if not returns_cache:
        print('  No returns loaded.')
        return pd.DataFrame()

    # Build daily panel: which firms are active on each date
    all_dates = ff.index[(ff.index >= date_min) & (ff.index <= date_max)]

    daily_portfolio = []
    for date in all_dates:
        # Find active firms: announcement_date + window_start <= date <= announcement_date + window_end
        active_mask = (
            (events_subset['announcement_date'] + pd.Timedelta(days=EVENT_WINDOW_DAYS[0]) <= date) &
            (events_subset['announcement_date'] + pd.Timedelta(days=EVENT_WINDOW_DAYS[1]) >= date)
        )
        active_events = events_subset[active_mask]

        if len(active_events) == 0:
            continue

        # Average daily return across active firms (equal-weight)
        daily_rets = []
        for _, ev in active_events.iterrows():
            ticker = ev['ticker']
            if ticker in returns_cache:
                ret_df = returns_cache[ticker]
                if date in ret_df.index:
                    ret = ret_df.loc[date, 'ret']
                    if not np.isnan(ret):
                        daily_rets.append(ret)

        if len(daily_rets) == 0:
            continue

        port_ret  = np.mean(daily_rets)
        rf_day    = ff.loc[date, 'RF'] if date in ff.index else 0
        daily_portfolio.append({
            'date':     date,
            'port_ret': port_ret,
            'rf':       rf_day,
            'n_firms':  len(daily_rets),
        })

    if not daily_portfolio:
        return pd.DataFrame()

    daily_df = pd.DataFrame(daily_portfolio).set_index('date')
    daily_df['excess_ret'] = daily_df['port_ret'] - daily_df['rf']

    # Aggregate to monthly
    monthly = daily_df.resample('ME').agg(
        port_ret  = ('port_ret',  lambda x: (1 + x).prod() - 1),
        excess_ret= ('excess_ret', 'sum'),   # approximate monthly excess return
        n_firms   = ('n_firms',   'mean'),
    )
    # Merge monthly FF4 factors
    ff_monthly = ff.resample('ME').agg(
        MKT_RF = ('MKT_RF', 'sum'),
        SMB    = ('SMB',    'sum'),
        HML    = ('HML',    'sum'),
        MOM    = ('MOM',    'sum'),
        RF     = ('RF',     'sum'),
    ) if all(c in ff.columns for c in ['MKT_RF','SMB','HML','MOM']) else pd.DataFrame()

    if ff_monthly.empty:
        return pd.DataFrame()

    merged = monthly.join(ff_monthly, how='inner')
    merged = merged.dropna(subset=['excess_ret', 'MKT_RF'])
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 3. Run calendar-time FF4 regression for one subsample
# ══════════════════════════════════════════════════════════════════════════════

def run_ct_regression(monthly_df, label):
    """FF4 OLS on monthly portfolio excess returns."""
    if len(monthly_df) < 12:
        print(f'  {label}: Insufficient months ({len(monthly_df)}). Skipping.')
        return None

    y = monthly_df['excess_ret']
    X = monthly_df[['MKT_RF', 'SMB', 'HML', 'MOM']]
    X_const = sm.add_constant(X, has_constant='add')

    mask = y.notna() & X_const.notna().all(axis=1)
    model = sm.OLS(y[mask], X_const[mask]).fit()  # OLS, no HC correction (time series)

    alpha = model.params['const']
    t_alpha = model.tvalues['const']
    p_alpha = model.pvalues['const']
    n_months = mask.sum()
    annualized = alpha * 12 * 100  # annualized percent

    print(f'\n  {label}')
    print(f'  {"─" * 55}')
    print(f'  Months: {n_months}  |  Avg firms/month: {monthly_df["n_firms"].mean():.1f}')
    print(f'  α (monthly) = {alpha*100:.3f}%  |  t = {t_alpha:.3f}  |  p = {p_alpha:.4f}  {stars(p_alpha)}')
    print(f'  α (annualized) = {annualized:.2f}%')
    print(f'  R² = {model.rsquared:.3f}')

    return {
        'label':          label,
        'n_months':       n_months,
        'avg_firms_month': monthly_df['n_firms'].mean(),
        'alpha_monthly_pct': alpha * 100,
        'alpha_annual_pct':  annualized,
        't_alpha':        t_alpha,
        'p_alpha':        p_alpha,
        'stars':          stars(p_alpha),
        'R2':             model.rsquared,
        'beta_mkt':       model.params.get('MKT_RF', np.nan),
        'model':          model,
        'monthly_df':     monthly_df,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_ct_results(ct_results):
    # Filter out results with model objects
    valid = [r for r in ct_results if r is not None and 'monthly_df' in r]
    if not valid:
        return

    n = len(valid)
    fig, axes = plt.subplots(1, min(n, 3), figsize=(5 * min(n, 3), 5))
    if n == 1:
        axes = [axes]

    colors = [BLUE, RED, GREEN, GREY, '#d94801', '#7b2d8b']

    for i, (res, ax) in enumerate(zip(valid[:3], axes)):
        mdf = res['monthly_df']
        # Compute monthly abnormal return: excess_ret - fitted (excl. alpha)
        X_noconst = sm.add_constant(mdf[['MKT_RF','SMB','HML','MOM']], has_constant='add')
        mask = mdf['excess_ret'].notna() & X_noconst.notna().all(axis=1)
        predicted_no_alpha = X_noconst[mask].drop(columns='const') @ res['model'].params[1:]
        monthly_ar = (mdf['excess_ret'][mask] - predicted_no_alpha) * 100

        ax.bar(range(len(monthly_ar)), monthly_ar.values,
               color=[RED if v < 0 else BLUE for v in monthly_ar.values],
               alpha=0.65, width=0.8)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axhline(res['alpha_monthly_pct'], color=GREEN, linewidth=1.5, linestyle='-',
                   label=f'α = {res["alpha_monthly_pct"]:.2f}% {res["stars"]}')

        ax.set_title(f'{res["label"]}\nα={res["alpha_monthly_pct"]:.2f}%/mo  '
                     f't={res["t_alpha"]:.2f}  {res["stars"]}', fontsize=9.5)
        ax.set_xlabel('Month (index)', fontsize=10)
        ax.set_ylabel('Monthly Abnormal Return (%)', fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle('Calendar-Time Portfolio Abnormal Returns\n(Corrects for cross-event clustering)',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_ct_portfolio.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\n  Figure saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 65)
    print('CALENDAR-TIME PORTFOLIO ANALYSIS')
    print('  Corrects for event clustering (e.g. Jan 2023 layoff wave)')
    print('  Method: Jaffe (1974) / Fama (1998) calendar-time portfolios')
    print('=' * 65)

    events = load_events()
    ff = load_ff()

    # Define subsamples
    subsamples = {
        'All events (US)':           events,
        'Pre-ChatGPT':               events[events['post_chatgpt'] == 0],
        'Post-ChatGPT':              events[events['post_chatgpt'] == 1],
        'Post-ChatGPT, AI':          events[(events['post_chatgpt'] == 1) & (events['ai_broad'] == 1)],
        'Post-ChatGPT, Non-AI':      events[(events['post_chatgpt'] == 1) & (events['ai_broad'] == 0)],
    }

    ct_results = []
    rows = []

    print('\n' + '=' * 65)
    print('CALENDAR-TIME REGRESSION RESULTS')
    print('=' * 65)

    for label, ev_sub in subsamples.items():
        print(f'\n  Building portfolio: {label} ({len(ev_sub)} events)...')
        if len(ev_sub) < 5:
            print('  Too few events. Skipping.')
            continue
        monthly_df = build_monthly_portfolio(ev_sub, ff)
        if monthly_df.empty:
            print(f'  No monthly data for {label}. Skipping.')
            continue
        res = run_ct_regression(monthly_df, label)
        ct_results.append(res)
        if res is not None:
            rows.append({k: v for k, v in res.items() if k not in ('model', 'monthly_df')})

    # Save results table
    results_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'ct_results.csv')
    results_df.to_csv(out_csv, index=False)
    print(f'\n  Results saved → {out_csv}')

    # Plot
    plot_ct_results(ct_results)

    # Summary
    print('\n' + '=' * 65)
    print('CALENDAR-TIME SUMMARY')
    print('=' * 65)
    print(f'\n  {"Subsample":<35} {"α/month":>9} {"t":>8} {"p":>8} {"Sig":>4} {"Ann. α":>9}')
    print(f'  {"─" * 80}')
    for r in rows:
        print(f'  {r["label"]:<35} {r["alpha_monthly_pct"]:>9.3f}% '
              f'{r["t_alpha"]:>8.3f} {r["p_alpha"]:>8.4f} {r["stars"]:>4} '
              f'{r["alpha_annual_pct"]:>8.2f}%')

    print(f'\n  Interpretation note:')
    print(f'  If α is consistent with event-study CAAR direction,')
    print(f'  the clustering-corrected result confirms the main finding.')
    print(f'  All outputs saved to: {OUT_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
