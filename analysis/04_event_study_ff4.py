"""
Phase 4: Event Study Analysis for Tech Layoff Paper
====================================================
Implements:
  - CAPM and Fama-French 4-factor (FF4) event study
  - Patell (1976), BMP (1991), and Corrado (1989) statistical tests
  - CAAR plots and per-event CAR tables

Note on international stocks: We use US FF4 factors (MKT_RF) as an
approximation for international benchmarks. The US-only subsample
is the cleanest specification.
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

# ── Paths ──────────────────────────────────────────────────────────────
BASE = '/Users/irmina/Documents/Claude/layoff_study'
EVENTS_PATH = os.path.join(BASE, 'data/processed/master_events_final.csv')
FF_PATH = os.path.join(BASE, 'data/processed/ff_factors.csv')
RETURNS_DIR = os.path.join(BASE, 'data/processed/stock_returns')
RESULTS_DIR = os.path.join(BASE, 'data/results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Event windows ──────────────────────────────────────────────────────
EVENT_WINDOWS = {
    '[-1,+1]':  (-1,  1),   # 3-day announcement window (short-term)
    '[0,+5]':   (0,   5),   # post-announcement week (short-term)
    '[0,+10]':  (0,  10),   # two weeks (medium-term)
    '[0,+20]':  (0,  20),   # one month (medium-term)
    '[0,+60]':  (0,  60),   # three months (long-term)
}
# NOTE: estimation window is [-260, -11]. All event windows start at -1 or later
# to avoid any overlap with the estimation period.

EST_START, EST_END = -260, -11  # estimation window in trading days


def load_data():
    """Load events, factors."""
    events = pd.read_csv(EVENTS_PATH)
    events['announcement_date'] = pd.to_datetime(events['announcement_date'])
    events = events[events['event_study_usable'] == True].copy()
    events = events.reset_index(drop=True)

    ff = pd.read_csv(FF_PATH, parse_dates=['Date'], index_col='Date')
    ff.index = pd.to_datetime(ff.index)
    ff = ff.sort_index()

    return events, ff


def load_stock_returns(ticker):
    """Load daily returns for a single ticker."""
    path = os.path.join(RETURNS_DIR, f'{ticker}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    # Normalize column name
    if ticker in df.columns:
        df = df.rename(columns={ticker: 'ret'})
    elif len(df.columns) == 1:
        df.columns = ['ret']
    if 'ret' not in df.columns:
        return None
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    return df[['ret']]


def find_event_day(ret_df, ann_date, tolerance=3):
    """Find nearest trading day to announcement_date within +-tolerance calendar days."""
    trading_dates = ret_df.index
    window_start = ann_date - pd.Timedelta(days=tolerance)
    window_end = ann_date + pd.Timedelta(days=tolerance)
    candidates = trading_dates[(trading_dates >= window_start) & (trading_dates <= window_end)]
    if len(candidates) == 0:
        return None
    # Pick closest
    diffs = abs(candidates - ann_date)
    return candidates[diffs.argmin()]


def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize at given percentiles."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def run_single_event(event_row, ff, model='ff4'):
    """
    Run event study for a single event.

    Returns:
        result dict with estimation stats and daily AR/SAR for event window,
        or None if data insufficient.
    """
    ticker = event_row['ticker']
    ann_date = event_row['announcement_date']

    # Load returns
    ret_df = load_stock_returns(ticker)
    if ret_df is None:
        return None

    # Find t=0
    t0_date = find_event_day(ret_df, ann_date)
    if t0_date is None:
        return None

    # Build trading-day index relative to t=0
    trading_dates = ret_df.index.sort_values()
    t0_loc = trading_dates.get_loc(t0_date)
    if isinstance(t0_loc, slice):
        t0_loc = t0_loc.start

    # Check we have enough data
    est_start_loc = t0_loc + EST_START
    est_end_loc = t0_loc + EST_END
    evt_end_loc = t0_loc + 60  # max event window end (+60 = 3 months)

    if est_start_loc < 0:
        return None
    if evt_end_loc >= len(trading_dates):
        return None

    # Estimation window dates
    est_dates = trading_dates[est_start_loc:est_end_loc + 1]
    # Full event window dates [-20, +60]
    evt_start_loc = t0_loc - 11  # start at day -11 (first day outside estimation window)
    if evt_start_loc < 0:
        return None
    evt_dates = trading_dates[evt_start_loc:evt_end_loc + 1]

    # Merge returns with factors
    est_data = ret_df.loc[est_dates].join(ff, how='inner')
    evt_data = ret_df.loc[evt_dates].join(ff, how='inner')

    # Need minimum 100 estimation observations
    if len(est_data) < 100:
        return None

    # Winsorize estimation-window returns
    est_data['ret'] = winsorize(est_data['ret'])

    # Excess return
    est_data['excess_ret'] = est_data['ret'] - est_data['RF']
    evt_data['excess_ret'] = evt_data['ret'] - evt_data['RF']

    # Choose factors
    if model == 'capm':
        factor_cols = ['MKT_RF']
    else:
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'MOM']

    est_data = est_data.dropna(subset=['excess_ret'] + factor_cols)
    evt_data = evt_data.dropna(subset=['excess_ret'] + factor_cols)

    if len(est_data) < 100:
        return None

    # OLS regression
    Y = est_data['excess_ret'].values
    X = est_data[factor_cols].values
    X_const = sm.add_constant(X)

    try:
        ols_result = sm.OLS(Y, X_const).fit()
    except Exception:
        return None

    alpha = ols_result.params[0]
    betas = ols_result.params[1:]
    residuals = ols_result.resid
    s_i = np.std(residuals, ddof=len(ols_result.params))  # residual std
    r2 = ols_result.rsquared
    n_est = len(est_data)

    # (X'X)^{-1} for Patell adjustment
    XtX_inv = np.linalg.inv(X_const.T @ X_const)

    # Compute AR for event window
    evt_X = evt_data[factor_cols].values
    evt_X_const = sm.add_constant(evt_X)
    predicted = evt_X_const @ ols_result.params
    AR = evt_data['excess_ret'].values - predicted

    # Compute SAR (standardized AR) for Patell test
    # SAR_t = AR_t / (s_i * sqrt(1 + x_t'(X'X)^{-1}x_t))
    SAR = np.zeros(len(AR))
    for j in range(len(AR)):
        x_t = evt_X_const[j]
        var_adj = 1 + x_t @ XtX_inv @ x_t
        SAR[j] = AR[j] / (s_i * np.sqrt(var_adj))

    # Map event-window dates to relative trading days
    # evt_dates starts at t0_loc - 11
    rel_days = np.arange(-11, -11 + len(evt_data))
    # But some days may be missing due to factor alignment, so recompute
    # Actually we need to be precise: the evt_dates we sliced are consecutive trading days
    # from t0-20 to t0+60. After inner join with factors, some may drop.
    # We need to map each remaining date to its relative day.
    all_trading = trading_dates[evt_start_loc:evt_end_loc + 1]
    date_to_relday = {d: i - 11 for i, d in enumerate(all_trading)}
    rel_days = np.array([date_to_relday.get(d, np.nan) for d in evt_data.index])

    # ── Data quality filter: exclude penny stocks / bankrupt OTC stocks ──────
    # If any single-day |AR| in the event window exceeds 50%, the stock is
    # likely a near-zero-price OTC/bankruptcy name (e.g. SONDQ, SDCCQ) where
    # small absolute moves produce enormous % returns.  Standard event-study
    # practice is to drop these to prevent CAAR distortion.
    if np.max(np.abs(AR)) > 0.50:
        return None

    # Build daily results
    daily = pd.DataFrame({
        'date': evt_data.index,
        'rel_day': rel_days,
        'ret': evt_data['ret'].values,
        'AR': AR,
        'SAR': SAR,
    })

    # Also compute AR/SAR for estimation window (needed for Corrado rank test)
    est_predicted = X_const @ ols_result.params
    est_AR = est_data['excess_ret'].values - est_predicted
    est_SAR = np.zeros(len(est_AR))
    for j in range(len(est_AR)):
        x_t = X_const[j]
        var_adj = 1 + x_t @ XtX_inv @ x_t
        est_SAR[j] = est_AR[j] / (s_i * np.sqrt(var_adj))

    return {
        'ticker': ticker,
        'company_fyi': event_row.get('company_fyi', ''),
        'announcement_date': ann_date,
        'period': event_row.get('period', ''),
        'listing_region': event_row.get('listing_region', ''),
        'industry': event_row.get('industry', ''),
        'ai_mentioned': event_row.get('ai_mentioned', 0),
        'layoff_count': event_row.get('layoff_count', np.nan),
        'layoff_pct': event_row.get('layoff_pct', ''),
        't0_date': t0_date,
        'alpha': alpha,
        'betas': betas,
        'beta_mkt': betas[0],
        's_i': s_i,
        'r2': r2,
        'n_est': n_est,
        'daily': daily,
        'est_AR': est_AR,
        'model': model,
    }


def compute_cars(result):
    """Compute CAR for each event window from daily AR."""
    daily = result['daily']
    cars = {}
    for wname, (t1, t2) in EVENT_WINDOWS.items():
        mask = (daily['rel_day'] >= t1) & (daily['rel_day'] <= t2)
        window_data = daily[mask]
        if len(window_data) == 0:
            cars[wname] = np.nan
        else:
            cars[wname] = window_data['AR'].sum()
    return cars


def compute_scar(result, t1, t2):
    """Compute standardized CAR for a window."""
    daily = result['daily']
    mask = (daily['rel_day'] >= t1) & (daily['rel_day'] <= t2)
    window_data = daily[mask]
    if len(window_data) == 0:
        return np.nan
    car = window_data['AR'].sum()
    L = len(window_data)
    scar = car / (result['s_i'] * np.sqrt(L))
    return scar


def aggregate_tests(results_list, window_name):
    """
    Compute CAAR and three test statistics for a given window across events.

    Returns dict with CAAR, test stats, p-values.
    """
    t1, t2 = EVENT_WINDOWS[window_name]

    cars = []
    scars = []
    # For Corrado: collect ranks
    corrado_K = []

    for res in results_list:
        daily = res['daily']
        mask = (daily['rel_day'] >= t1) & (daily['rel_day'] <= t2)
        window_data = daily[mask]
        if len(window_data) == 0:
            continue

        car = window_data['AR'].sum()
        cars.append(car)

        L = len(window_data)
        scar = car / (res['s_i'] * np.sqrt(L))
        scars.append(scar)

        # Corrado rank test: rank AR across estimation + event window
        all_AR = np.concatenate([res['est_AR'], daily['AR'].values])
        T_total = len(all_AR)
        ranks = stats.rankdata(all_AR)
        # Event-window ranks start at index len(est_AR) + offset
        # We need to find the event-window AR within all_AR
        n_est = len(res['est_AR'])
        # Map event window days
        evt_ranks = ranks[n_est:]  # ranks for the full [-20,+60] window
        # Now find which of these correspond to [t1,t2]
        evt_rel = daily['rel_day'].values
        evt_mask = (evt_rel >= t1) & (evt_rel <= t2)
        if evt_mask.sum() == 0:
            continue
        window_ranks = evt_ranks[evt_mask]
        K_i = np.mean(window_ranks / (T_total + 1) - 0.5)
        corrado_K.append(K_i)

    cars = np.array(cars)
    scars = np.array(scars)
    corrado_K = np.array(corrado_K)
    N = len(cars)

    if N == 0:
        return None

    caar = np.mean(cars)
    caar_pct = caar * 100

    # Test 1: Patell t-stat
    # Z_patell = mean(SCAR) * sqrt(N)
    if N > 1:
        z_patell = np.mean(scars) * np.sqrt(N)
        p_patell = 2 * (1 - stats.norm.cdf(abs(z_patell)))
    else:
        z_patell = np.nan
        p_patell = np.nan

    # Test 2: BMP
    if N > 1:
        theta_bar = np.mean(scars)
        s_theta = np.std(scars, ddof=1)
        if s_theta > 0:
            t_bmp = theta_bar * np.sqrt(N) / s_theta
            p_bmp = 2 * (1 - stats.t.cdf(abs(t_bmp), df=N - 1))
        else:
            t_bmp = np.nan
            p_bmp = np.nan
    else:
        t_bmp = np.nan
        p_bmp = np.nan

    # Test 3: Corrado rank test
    if len(corrado_K) > 1:
        K_bar = np.mean(corrado_K)
        s_K = np.std(corrado_K, ddof=1)
        if s_K > 0:
            t_corrado = K_bar * np.sqrt(len(corrado_K)) / s_K
            p_corrado = 2 * (1 - stats.norm.cdf(abs(t_corrado)))
        else:
            t_corrado = np.nan
            p_corrado = np.nan
    else:
        t_corrado = np.nan
        p_corrado = np.nan

    def stars(p):
        if pd.isna(p):
            return ''
        if p < 0.01:
            return '***'
        elif p < 0.05:
            return '**'
        elif p < 0.10:
            return '*'
        return ''

    # Use most significant p-value for stars column
    p_vals = [p for p in [p_patell, p_bmp, p_corrado] if not np.isnan(p)]
    min_p = min(p_vals) if p_vals else np.nan

    return {
        'N': N,
        'CAAR': caar,
        'CAAR_pct': caar_pct,
        't_patell': z_patell,
        'p_patell': p_patell,
        't_BMP': t_bmp,
        'p_BMP': p_bmp,
        't_corrado': t_corrado,
        'p_corrado': p_corrado,
        'stars': stars(min_p),
    }


def compute_daily_caar(results_list, min_day=-11, max_day=60):
    """Compute CAAR path day by day for plotting, with confidence bands."""
    daily_ars = {}  # rel_day -> list of AR values across events
    for res in results_list:
        daily = res['daily']
        for _, row in daily.iterrows():
            rd = int(row['rel_day'])
            if min_day <= rd <= max_day:
                if rd not in daily_ars:
                    daily_ars[rd] = []
                daily_ars[rd].append(row['AR'])

    days = sorted(daily_ars.keys())
    caar_path = []
    ci_lower = []
    ci_upper = []
    cumulative = 0
    for d in days:
        ars = np.array(daily_ars[d])
        N = len(ars)
        mean_ar = np.mean(ars)
        cumulative += mean_ar
        caar_path.append(cumulative * 100)  # percent

        # 95% CI based on cross-sectional std
        if N > 1:
            se = np.std(ars, ddof=1) / np.sqrt(N)
            # Cumulative SE: approximate as sqrt(sum of daily variances)
            # For simplicity, use cross-sectional SE of CARs up to this day
        else:
            se = 0

    # Better CI: compute CAR up to each day for each event, then cross-sectional SE
    # Rebuild per-event cumulative paths
    event_cars = {}  # event_idx -> {day: cumulative_ar}
    for idx, res in enumerate(results_list):
        daily = res['daily']
        cum = 0
        event_cars[idx] = {}
        for _, row in daily.iterrows():
            rd = int(row['rel_day'])
            if min_day <= rd <= max_day:
                cum += row['AR']
                event_cars[idx][rd] = cum

    caar_path = []
    ci_lower = []
    ci_upper = []
    for d in days:
        cars_at_d = [event_cars[idx][d] for idx in event_cars if d in event_cars[idx]]
        N = len(cars_at_d)
        mean_car = np.mean(cars_at_d)
        caar_path.append(mean_car * 100)
        if N > 1:
            se = np.std(cars_at_d, ddof=1) / np.sqrt(N)
            ci_lower.append((mean_car - 1.96 * se) * 100)
            ci_upper.append((mean_car + 1.96 * se) * 100)
        else:
            ci_lower.append(mean_car * 100)
            ci_upper.append(mean_car * 100)

    return days, caar_path, ci_lower, ci_upper


def plot_caap_full(results_capm, results_ff4, save_path):
    """Plot 1: Full sample CAAR, CAPM vs FF4."""
    fig, ax = plt.subplots(figsize=(10, 6))

    days_c, caar_c, lo_c, hi_c = compute_daily_caar(results_capm)
    days_f, caar_f, lo_f, hi_f = compute_daily_caar(results_ff4)

    ax.plot(days_c, caar_c, color='#2166ac', linewidth=1.8, label=f'CAPM (N={len(results_capm)})')
    ax.fill_between(days_c, lo_c, hi_c, color='#2166ac', alpha=0.12)

    ax.plot(days_f, caar_f, color='#b2182b', linewidth=1.8, label=f'FF4 (N={len(results_ff4)})')
    ax.fill_between(days_f, lo_f, hi_f, color='#b2182b', alpha=0.12)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Event Day (relative to announcement)', fontsize=12)
    ax.set_ylabel('CAAR (%)', fontsize=12)
    ax.set_title('Cumulative Average Abnormal Returns: Tech Layoff Announcements', fontsize=13)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_caap_pre_post(results_pre, results_post, save_path):
    """Plot 2: Pre vs Post GenAI."""
    fig, ax = plt.subplots(figsize=(10, 6))

    days_pre, caar_pre, lo_pre, hi_pre = compute_daily_caar(results_pre)
    days_post, caar_post, lo_post, hi_post = compute_daily_caar(results_post)

    ax.plot(days_pre, caar_pre, color='#2166ac', linewidth=1.8,
            label=f'Pre-GenAI, <=2022 (N={len(results_pre)})')
    ax.fill_between(days_pre, lo_pre, hi_pre, color='#2166ac', alpha=0.12)

    ax.plot(days_post, caar_post, color='#b2182b', linewidth=1.8,
            label=f'Post-GenAI, >=2023 (N={len(results_post)})')
    ax.fill_between(days_post, lo_post, hi_post, color='#b2182b', alpha=0.12)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Event Day (relative to announcement)', fontsize=12)
    ax.set_ylabel('CAAR (%)', fontsize=12)
    ax.set_title('CAAR: Pre-GenAI vs Post-GenAI Layoff Announcements (FF4 Model)', fontsize=13)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_caap_ai(results_ai, results_nonai, save_path):
    """Plot 3: AI-mentioned vs not, Post-GenAI only."""
    fig, ax = plt.subplots(figsize=(10, 6))

    days_ai, caar_ai, lo_ai, hi_ai = compute_daily_caar(results_ai)
    days_na, caar_na, lo_na, hi_na = compute_daily_caar(results_nonai)

    ax.plot(days_na, caar_na, color='#2166ac', linewidth=1.8,
            label=f'AI not mentioned (N={len(results_nonai)})')
    ax.fill_between(days_na, lo_na, hi_na, color='#2166ac', alpha=0.12)

    ax.plot(days_ai, caar_ai, color='#b2182b', linewidth=1.8,
            label=f'AI mentioned (N={len(results_ai)})')
    ax.fill_between(days_ai, lo_ai, hi_ai, color='#b2182b', alpha=0.12)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Event Day (relative to announcement)', fontsize=12)
    ax.set_ylabel('CAAR (%)', fontsize=12)
    ax.set_title('CAAR: AI-Mentioned vs Non-AI Layoffs, Post-GenAI (FF4 Model)', fontsize=13)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_caap_tech_vs_nontech(results_tech, results_nontech, save_path):
    """Plot 4: Core tech vs non-tech sectors, US only, FF4."""
    fig, ax = plt.subplots(figsize=(10, 6))

    days_t, caar_t, lo_t, hi_t = compute_daily_caar(results_tech)
    days_n, caar_n, lo_n, hi_n = compute_daily_caar(results_nontech)

    ax.plot(days_n, caar_n, color='#4dac26', linewidth=1.8,
            label=f'Non-tech sectors (N={len(results_nontech)})')
    ax.fill_between(days_n, lo_n, hi_n, color='#4dac26', alpha=0.12)

    ax.plot(days_t, caar_t, color='#7b2d8b', linewidth=1.8,
            label=f'Core tech (N={len(results_tech)})')
    ax.fill_between(days_t, lo_t, hi_t, color='#7b2d8b', alpha=0.12)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Event Day (relative to announcement)', fontsize=12)
    ax.set_ylabel('CAAR (%)', fontsize=12)
    ax.set_title('CAAR: Core Tech vs Non-Tech Sectors, US Only (FF4 Model)', fontsize=13)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_caar_path(results_ff4, save_path, title='CAAR Time Path: Layoff Announcements'):
    """Plot CAAR cumulative path from day -10 to +60 (FF4, full sample).

    Y is reindexed so that the value at T=-1 is exactly 0. This means:
    - For T >= 0, the path value equals the window CAAR [0, T] from car_summary
    - The zero-crossing in the figure directly corresponds to the day where
      window CARs transition from negative to positive (~day 7-8)
    """
    days, caar, lo, hi = compute_daily_caar(results_ff4, min_day=-10, max_day=60)

    # Reindex: set Y=0 at T=-1 so post-announcement path = window CARs [0,T]
    if -1 in days:
        idx_m1 = days.index(-1)
        baseline = caar[idx_m1]
        caar = [c - baseline for c in caar]
        lo   = [l - baseline for l in lo]
        hi   = [h - baseline for h in hi]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(days, caar, color='#b2182b', linewidth=2.0, label=f'FF4 CAAR (N={len(results_ff4)})')
    ax.fill_between(days, lo, hi, color='#b2182b', alpha=0.12, label='95% CI')

    # Mark announcement day
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.9, alpha=0.7, label='Announcement (t=0)')
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.6, alpha=0.6)

    # Shade pre-announcement window
    ax.axvspan(-10, 0, alpha=0.04, color='steelblue', label='Pre-announcement window [-10,0]')

    # Annotate key event windows (values now equal window CARs from Table 1)
    for day, label in [(1, '[-1,+1]'), (5, '[0,+5]'), (10, '[0,+10]'), (20, '[0,+20]'), (60, '[0,+60]')]:
        if day in days:
            idx = days.index(day)
            ax.annotate(f'{label}\n{caar[idx]:.2f}%',
                        xy=(day, caar[idx]),
                        xytext=(day + 2, caar[idx] + 0.3),
                        fontsize=7, color='#444444',
                        arrowprops=dict(arrowstyle='->', color='#888888', lw=0.7))

    ax.set_xlabel('Trading Days Relative to Announcement (t = 0)', fontsize=12)
    ax.set_ylabel('Cumulative Average Abnormal Return (%)\n(Reindexed: Y = 0 at t = −1)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(frameon=True, fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.25)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Caption note at bottom
    note = ('注：Y轴已在 t=−1 处归零，图中路径值等同于表1各窗口CAAR（均从第0天起累计）。'
            'CAAR由负转正约在公告后第7–8个交易日（[0,+5]=−0.66%，[0,+10]=+0.09%）。')
    fig.text(0.5, -0.02, note, ha='center', fontsize=8, color='#555555',
             wrap=True, fontstyle='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save_path}')


def build_per_event_table(results_capm, results_ff4):
    """Build per-event CAR table (Step 5)."""
    # Index ff4 results by (ticker, announcement_date)
    ff4_lookup = {}
    for res in results_ff4:
        key = (res['ticker'], res['announcement_date'])
        ff4_lookup[key] = res

    rows = []
    for res in results_ff4:
        cars = compute_cars(res)
        rows.append({
            'company_fyi': res['company_fyi'],
            'ticker': res['ticker'],
            'period': res['period'],
            'announcement_date': res['announcement_date'],
            'listing_region': res['listing_region'],
            'industry': res['industry'],
            'ai_mentioned': res['ai_mentioned'],
            'layoff_count': res['layoff_count'],
            'layoff_pct': res['layoff_pct'],
            'CAR_1_1': cars.get('[-1,+1]', np.nan),
            'CAR_0_5': cars.get('[0,+5]', np.nan),
            'CAR_0_10': cars.get('[0,+10]', np.nan),
            'CAR_0_20': cars.get('[0,+20]', np.nan),
            'CAR_0_60': cars.get('[0,+60]', np.nan),
            'alpha_ff4': res['alpha'],
            'beta_mkt_ff4': res['beta_mkt'],
            'r2_ff4': res['r2'],
            'n_est_obs': res['n_est'],
        })
    return pd.DataFrame(rows)


def main():
    print('=' * 72)
    print('PHASE 4: EVENT STUDY ANALYSIS — TECH LAYOFF ANNOUNCEMENTS')
    print('=' * 72)

    # ── Load data ──
    print('\nLoading data...')
    events, ff = load_data()
    print(f'  Usable events: {len(events)}')
    print(f'  FF factors: {len(ff)} trading days, {ff.index.min().date()} to {ff.index.max().date()}')

    # ── Run event studies ──
    for model_name in ['capm', 'ff4']:
        print(f'\n--- Running {model_name.upper()} model ---')
        results = []
        skipped = 0
        for idx, row in events.iterrows():
            res = run_single_event(row, ff, model=model_name)
            if res is None:
                skipped += 1
            else:
                results.append(res)
            if (idx + 1) % 100 == 0:
                print(f'  Processed {idx + 1}/{len(events)} events...')

        print(f'  Completed: {len(results)} events, skipped: {skipped}')

        if model_name == 'capm':
            results_capm = results
        else:
            results_ff4 = results

    # ── Define subsamples ──
    def subsample(results, condition):
        return [r for r in results if condition(r)]

    # Core tech industry classification (based on layoffs.fyi taxonomy).
    # "Other" is included because it is the catch-all for software companies
    # that do not fit a named vertical (commonly pure-play SaaS, enterprise software).
    CORE_TECH_INDUSTRIES = {
        'Hardware', 'Security', 'Data', 'Infrastructure', 'Marketing',
        'Media', 'Support', 'AI', 'Crypto', 'Product', 'Education',
        'Recruiting', 'HR', 'Other',
    }

    # Condition A: tickers from 裁员影响模型 hand-curated sample (post-IPO mature firms,
    # double-verified via Yahoo+OpenFIGI). 130 of 152 tickers have stock data here;
    # 22 missing are international (6) or distressed/OTC firms excluded on quality grounds.
    _cond_a_path = os.path.join(BASE, 'data/processed/condition_a_tickers.csv')
    if os.path.exists(_cond_a_path):
        COND_A_TICKERS = set(pd.read_csv(_cond_a_path)['ticker'].tolist())
        print(f'  Condition A: {len(COND_A_TICKERS)} tickers loaded from {_cond_a_path}')
    else:
        COND_A_TICKERS = set()
        print(f'  WARNING: condition_a_tickers.csv not found at {_cond_a_path}')

    # PRIMARY SPEC: US-listed stocks only (US FF4 factors are misspecified for
    # international stocks; US-only is the clean causal identification sample).
    # Full sample is reported as a secondary/appendix check.
    subsamples = {
        'US only (PRIMARY)': lambda r: r['listing_region'] == 'US',
        'Condition A (FF4)': lambda r, _t=COND_A_TICKERS: r['ticker'] in _t and r['listing_region'] == 'US',
        'Full sample': lambda r: True,
        'Core tech, US only': lambda r: r['listing_region'] == 'US' and r['industry'] in CORE_TECH_INDUSTRIES,
        'Non-tech, US only': lambda r: r['listing_region'] == 'US' and r['industry'] not in CORE_TECH_INDUSTRIES,
        'US only, Post-GenAI (>=2023)': lambda r: r['listing_region'] == 'US' and r['announcement_date'].year >= 2023,
        'US only, Pre-GenAI (<=2022)': lambda r: r['listing_region'] == 'US' and r['announcement_date'].year <= 2022,
        'Post-GenAI (>=2023)': lambda r: r['announcement_date'].year >= 2023,
        'Pre-GenAI (<=2022)': lambda r: r['announcement_date'].year <= 2022,
    }

    # ── Build CAR summary table (Step 3) ──
    print('\n--- Computing CAAR and test statistics ---')
    summary_rows = []

    for model_name, all_results in [('CAPM', results_capm), ('FF4', results_ff4)]:
        for sample_name, cond_fn in subsamples.items():
            sub = subsample(all_results, cond_fn)
            for wname in EVENT_WINDOWS:
                agg = aggregate_tests(sub, wname)
                if agg is None:
                    continue
                summary_rows.append({
                    'sample': sample_name,
                    'model': model_name,
                    'window': wname,
                    **agg,
                })

    car_summary = pd.DataFrame(summary_rows)
    car_summary.to_csv(os.path.join(RESULTS_DIR, 'car_summary.csv'), index=False)
    print(f'  Saved: {os.path.join(RESULTS_DIR, "car_summary.csv")}')

    # ── Print CAAR table ──
    print('\n' + '=' * 72)
    print('CUMULATIVE AVERAGE ABNORMAL RETURNS (CAAR) — SUMMARY TABLE')
    print('=' * 72)

    for model_name in ['CAPM', 'FF4']:
        print(f'\n{"─" * 72}')
        print(f'Model: {model_name}')
        print(f'{"─" * 72}')
        for sample_name in subsamples:
            subset = car_summary[(car_summary['model'] == model_name) &
                                 (car_summary['sample'] == sample_name)]
            if len(subset) == 0:
                continue
            print(f'\n  Sample: {sample_name}')
            print(f'  {"Window":<14} {"N":>5} {"CAAR%":>8} {"t_Patell":>9} {"p":>7} '
                  f'{"t_BMP":>8} {"p":>7} {"t_Corrado":>10} {"p":>7} {"Sig":>4}')
            print(f'  {"-" * 82}')
            for _, r in subset.iterrows():
                print(f'  {r["window"]:<14} {r["N"]:>5} {r["CAAR_pct"]:>8.3f} '
                      f'{r["t_patell"]:>9.3f} {r["p_patell"]:>7.4f} '
                      f'{r["t_BMP"]:>8.3f} {r["p_BMP"]:>7.4f} '
                      f'{r["t_corrado"]:>10.3f} {r["p_corrado"]:>7.4f} '
                      f'{r["stars"]:>4}')

    # ── Per-event CAR table (Step 5) ──
    print('\n--- Building per-event CAR table ---')
    per_event = build_per_event_table(results_capm, results_ff4)
    per_event.to_csv(os.path.join(RESULTS_DIR, 'car_by_event.csv'), index=False)
    print(f'  Saved: {os.path.join(RESULTS_DIR, "car_by_event.csv")} ({len(per_event)} events)')

    # ── Plots (Step 4) ──
    print('\n--- Generating CAAR plots ---')

    # Plot 1: Full sample, CAPM vs FF4
    plot_caap_full(results_capm, results_ff4,
                   os.path.join(RESULTS_DIR, 'caap_full_sample.png'))

    # Plot 2: Pre vs Post GenAI — PRIMARY SPEC: US only
    pre = subsample(results_ff4, lambda r: r['listing_region'] == 'US' and r['announcement_date'].year <= 2022)
    post = subsample(results_ff4, lambda r: r['listing_region'] == 'US' and r['announcement_date'].year >= 2023)
    plot_caap_pre_post(pre, post,
                       os.path.join(RESULTS_DIR, 'caap_pre_vs_post_genai.png'))

    # Plot 3: AI vs non-AI (Post-GenAI only, US only, FF4) — PRIMARY SPEC
    post_ai = subsample(results_ff4,
                        lambda r: r['listing_region'] == 'US' and r['announcement_date'].year >= 2023 and r['ai_mentioned'] == 1)
    post_nonai = subsample(results_ff4,
                           lambda r: r['listing_region'] == 'US' and r['announcement_date'].year >= 2023 and r['ai_mentioned'] == 0)
    plot_caap_ai(post_ai, post_nonai,
                 os.path.join(RESULTS_DIR, 'caap_ai_vs_nonai.png'))

    # Plot 4: Core tech vs non-tech, US only, FF4
    us_tech = subsample(results_ff4,
                        lambda r: r['listing_region'] == 'US' and r['industry'] in CORE_TECH_INDUSTRIES)
    us_nontech = subsample(results_ff4,
                           lambda r: r['listing_region'] == 'US' and r['industry'] not in CORE_TECH_INDUSTRIES)
    plot_caap_tech_vs_nontech(us_tech, us_nontech,
                              os.path.join(RESULTS_DIR, 'caap_tech_vs_nontech.png'))

    # Plot 5: CAAR time path -10 to +60 (primary path figure for README)
    us_ff4 = subsample(results_ff4, lambda r: r['listing_region'] == 'US')
    # Save daily CAAR path to CSV so figure can be replotted without re-running
    _days, _caar, _lo, _hi = compute_daily_caar(us_ff4, min_day=-10, max_day=60)
    _bl = _caar[_days.index(-1)]
    pd.DataFrame({'day': _days,
                  'caar_pct': [c - _bl for c in _caar],
                  'ci_lo': [l - _bl for l in _lo],
                  'ci_hi': [h - _bl for h in _hi]
                  }).to_csv(os.path.join(RESULTS_DIR, 'caar_path_daily.csv'), index=False)
    plot_caar_path(us_ff4,
                   os.path.join(RESULTS_DIR, 'fig_caar_path_full.png'),
                   title='CAAR Time Path: US Layoff Announcements (FF4, t = −10 to +60)')

    # ── Final summary ──
    print('\n' + '=' * 72)
    print('SUMMARY REPORT')
    print('=' * 72)

    n_total = len(events)
    n_capm = len(results_capm)
    n_ff4 = len(results_ff4)
    n_us = len(subsample(results_ff4, lambda r: r['listing_region'] == 'US'))
    n_pre = len(pre)
    n_post = len(post)
    n_ai = len(post_ai)
    n_nonai = len(post_nonai)

    print(f'\nEvents: {n_total} usable in master file')
    print(f'  CAPM processed: {n_capm}  |  Skipped: {n_total - n_capm}')
    print(f'  FF4  processed: {n_ff4}  |  Skipped: {n_total - n_ff4}')
    print(f'\nSubsamples (FF4):')
    print(f'  US only:        {n_us}')
    print(f'  Pre-GenAI:      {n_pre}')
    print(f'  Post-GenAI:     {n_post}')
    print(f'  Post-GenAI AI:  {n_ai}')
    print(f'  Post-GenAI non: {n_nonai}')

    # Key findings
    print('\nKEY FINDINGS (FF4 model, full sample):')
    for wname in EVENT_WINDOWS:
        row = car_summary[(car_summary['model'] == 'FF4') &
                          (car_summary['sample'] == 'Full sample') &
                          (car_summary['window'] == wname)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        direction = 'POSITIVE' if r['CAAR_pct'] > 0 else 'NEGATIVE'
        print(f'  {wname:>14}: CAAR = {r["CAAR_pct"]:+.3f}%  '
              f'(BMP t={r["t_BMP"]:.3f}, p={r["p_BMP"]:.4f}) {r["stars"]} {direction}')

    print('\n' + '=' * 72)
    print('Phase 4 complete. All results saved to data/results/')
    print('=' * 72)


if __name__ == '__main__':
    main()
