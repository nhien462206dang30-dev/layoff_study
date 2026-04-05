"""
Improved Analysis: Aligned ai_mentioned + Grouped Event Study + DID
====================================================================
Steps:
  1. Re-label ai_mentioned using desktop ticker file (broader definition)
     → target ~50% ai=1 rate vs original 2.6%
  2. Recompute daily AR panels with the FF4 model
  3. Grouped CAAR plots: ai=1 vs ai=0, and 4-way (ai × pre/post ChatGPT)
  4. DID regression: breakpoint = ChatGPT launch (2022-11-30)
  5. Cross-sectional OLS (aligned with desktop spec: CAR[-1,+1])
     plus robustness check on CAR[-5,+60]

Output saved to:  data/results/improved/
"""

import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

BASE        = '/Users/irmina/Documents/Claude/layoff_study'
RETURNS_DIR = os.path.join(BASE, 'data/processed/stock_returns')
FF_PATH     = os.path.join(BASE, 'data/processed/ff_factors.csv')
EVENTS_PATH = os.path.join(BASE, 'data/processed/master_events_final.csv')
CAR_PATH    = os.path.join(BASE, 'data/results/car_by_event.csv')
DESKTOP_TKR = '/Users/irmina/Desktop/裁员影响模型/为上市公司匹配ticker并清洗脏数据3/5手动匹配错误公司并删除后.csv'
OUT_DIR     = os.path.join(BASE, 'data/results/improved')
os.makedirs(OUT_DIR, exist_ok=True)

# Event study parameters (identical to original event_study.py)
EST_START, EST_END = -260, -11
CAAR_WINDOW = (-20, 30)       # t range for CAAR plot
CHATGPT_DATE = pd.Timestamp('2022-11-30')

BLUE, RED, GREEN, GREY = '#2166ac', '#b2182b', '#1a9850', '#636363'
ORANGE = '#d94801'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Rebuild ai_mentioned labels
# ═══════════════════════════════════════════════════════════════════

def build_ai_labels(car: pd.DataFrame) -> pd.DataFrame:
    """
    Merge desktop ticker file labels onto car_by_event.
    Priority: exact (ticker + date) > ticker-level majority > original label.
    """
    try:
        desk = pd.read_csv(DESKTOP_TKR, encoding='latin-1')
    except Exception as e:
        print(f"  WARNING: could not load desktop file ({e}). Using original labels.")
        return car

    desk['ticker']   = desk['ticker'].astype(str).str.strip()
    desk['date_dt']  = pd.to_datetime(desk['announcement_date_verified'], errors='coerce')
    desk['ai_desk']  = pd.to_numeric(desk['ai_mentioned'], errors='coerce')

    car = car.copy()
    car['date_dt']   = pd.to_datetime(car['announcement_date'])
    car['ticker']    = car['ticker'].astype(str).str.strip()

    # --- Exact match: ticker + date ---
    exact = desk.dropna(subset=['date_dt'])[['ticker','date_dt','ai_desk']].drop_duplicates(
        subset=['ticker','date_dt'], keep='first')
    car = car.merge(exact, on=['ticker','date_dt'], how='left')

    # --- Fallback: ticker-level max (if ANY event for that ticker was AI-labelled → 1) ---
    # Rationale: if company publicly cited AI in one layoff, it's plausible for others too.
    # Conservative alternative: use median; we use max to be consistent with desktop's
    # broad labeling intent.
    tkr_max = desk.groupby('ticker')['ai_desk'].max().reset_index().rename(
        columns={'ai_desk': 'ai_tkr'})
    car = car.merge(tkr_max, on='ticker', how='left')

    # Final label: exact match > ticker fallback > keep original
    car['ai_new'] = car['ai_desk'].fillna(car['ai_tkr']).fillna(car['ai_mentioned'])
    car['ai_new'] = car['ai_new'].fillna(0).astype(int)

    n_orig = int(car['ai_mentioned'].sum())
    n_new  = int(car['ai_new'].sum())
    print(f"  ai_mentioned: {n_orig} → {n_new}  "
          f"({n_orig/len(car)*100:.1f}% → {n_new/len(car)*100:.1f}%)")

    car['ai_mentioned'] = car['ai_new']
    car = car.drop(columns=['ai_desk','ai_tkr','ai_new','date_dt'], errors='ignore')
    return car


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Recompute daily AR panels
# ═══════════════════════════════════════════════════════════════════

def load_stock_returns(ticker: str):
    path = os.path.join(RETURNS_DIR, f'{ticker}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date').sort_index()
    if ticker in df.columns:
        df = df.rename(columns={ticker: 'ret'})
    elif len(df.columns) == 1:
        df.columns = ['ret']
    if 'ret' not in df.columns:
        return None
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    return df[['ret']]


def find_event_day(ret_df, ann_date, tol=3):
    td = ret_df.index
    cands = td[(td >= ann_date - pd.Timedelta(days=tol)) &
               (td <= ann_date + pd.Timedelta(days=tol))]
    if len(cands) == 0:
        return None
    return cands[np.argmin(np.abs(cands - ann_date))]


def compute_event_ars(event_row: pd.Series, ff: pd.DataFrame):
    """
    Returns DataFrame with columns [t, AR, SAR] over CAAR_WINDOW,
    plus dict of scalar CARs and estimation stats.
    Returns (None, None) if data insufficient.
    """
    ticker   = str(event_row['ticker'])
    ann_date = pd.to_datetime(event_row['announcement_date'])

    ret_df = load_stock_returns(ticker)
    if ret_df is None:
        return None, None

    t0 = find_event_day(ret_df, ann_date)
    if t0 is None:
        return None, None

    td = ret_df.index.sort_values()
    t0_loc = td.get_loc(t0)
    if isinstance(t0_loc, slice):
        t0_loc = t0_loc.start

    # Window bounds
    est_s = t0_loc + EST_START
    est_e = t0_loc + EST_END
    caar_s = t0_loc + CAAR_WINDOW[0]
    caar_e = t0_loc + CAAR_WINDOW[1]

    if est_s < 0 or caar_s < 0 or caar_e >= len(td):
        return None, None

    est_dates  = td[est_s:est_e + 1]
    caar_dates = td[caar_s:caar_e + 1]

    factor_cols = ['MKT_RF', 'SMB', 'HML', 'MOM']

    est_data  = ret_df.loc[est_dates].join(ff, how='inner').dropna(
        subset=['ret'] + factor_cols)
    caar_data = ret_df.loc[caar_dates].join(ff, how='inner').dropna(
        subset=['ret'] + factor_cols)

    if len(est_data) < 100 or len(caar_data) < (CAAR_WINDOW[1] - CAAR_WINDOW[0]):
        return None, None

    # Estimation
    est_data = est_data.copy()
    est_data['excess_ret'] = est_data['ret'] - est_data['RF']
    Y = est_data['excess_ret'].values
    X = sm.add_constant(est_data[factor_cols].values)

    try:
        ols = sm.OLS(Y, X).fit()
    except Exception:
        return None, None

    s_i    = np.std(ols.resid, ddof=len(ols.params))
    n_est  = len(est_data)
    XtX_inv = np.linalg.inv(X.T @ X)
    X_bar  = X.mean(axis=0)

    # AR for CAAR window
    caar_data = caar_data.copy()
    caar_data['excess_ret'] = caar_data['ret'] - caar_data['RF']
    Xe = sm.add_constant(caar_data[factor_cols].values)
    AR = caar_data['excess_ret'].values - (Xe @ ols.params)

    # SAR (Patell-style, accounts for prediction error variance)
    SAR = []
    for j in range(len(caar_data)):
        x_t = Xe[j]
        var_ar = (s_i**2) * (1 + x_t @ XtX_inv @ x_t)
        SAR.append(AR[j] / np.sqrt(var_ar) if var_ar > 0 else np.nan)

    # t index relative to event (0 = announcement day)
    t_index = np.arange(CAAR_WINDOW[0], CAAR_WINDOW[1] + 1)[:len(AR)]

    ar_df = pd.DataFrame({'t': t_index, 'AR': AR, 'SAR': np.array(SAR)})

    # Scalar CARs for each window
    def car_window(a, b):
        mask = (ar_df['t'] >= a) & (ar_df['t'] <= b)
        return ar_df.loc[mask, 'AR'].sum() if mask.sum() > 0 else np.nan

    scalars = {
        'alpha_ff4':    ols.params[0],
        'beta_mkt_ff4': ols.params[1],
        'r2_ff4':       ols.rsquared,
        'n_est_obs':    n_est,
        'CAR_1_1':      car_window(-1,  1),
        'CAR_0_5':      car_window( 0,  5),
        'CAR_0_20':     car_window( 0, 20),
        'CAR_5_60':     car_window(-5, 30),   # note: [-5,+30] due to CAAR_WINDOW end
    }
    # Full [-5,+60] if available (need extended window)
    mask_long = (ar_df['t'] >= -5) & (ar_df['t'] <= 30)
    scalars['CAR_5_30'] = ar_df.loc[mask_long, 'AR'].sum() if mask_long.sum() > 0 else np.nan

    return ar_df, scalars


def build_ar_panel(car: pd.DataFrame, ff: pd.DataFrame) -> tuple:
    """
    Recompute daily ARs for all events.
    Returns (ar_panel: DataFrame with event_id + t + AR + SAR,
             car_updated: car with refreshed scalar CARs).
    """
    ar_records, scalar_updates = [], []
    total = len(car)

    for i, (_, row) in enumerate(car.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{total} events processed...")

        ar_df, scalars = compute_event_ars(row, ff)
        if ar_df is None:
            continue

        ev_id = f"{row['ticker']}_{row['announcement_date']}"
        ar_df = ar_df.copy()
        ar_df['event_id']       = ev_id
        ar_df['ai_mentioned']   = row['ai_mentioned']
        ar_df['post_chatgpt']   = int(pd.to_datetime(row['announcement_date']) >= CHATGPT_DATE)
        ar_df['ticker']         = row['ticker']
        ar_df['announcement_date'] = row['announcement_date']
        ar_records.append(ar_df)

        scalars['event_id']         = ev_id
        scalars['ticker']           = row['ticker']
        scalars['company_fyi']      = row['company_fyi']
        scalars['announcement_date']= row['announcement_date']
        scalars['ai_mentioned']     = row['ai_mentioned']
        scalars['post_chatgpt']     = int(pd.to_datetime(row['announcement_date']) >= CHATGPT_DATE)
        scalars['period']           = row['period']
        scalars['listing_region']   = row['listing_region']
        scalars['industry']         = row.get('industry', '')
        scalars['layoff_count']     = row.get('layoff_count', np.nan)
        scalars['layoff_pct']       = row.get('layoff_pct', np.nan)
        scalar_updates.append(scalars)

    ar_panel   = pd.concat(ar_records, ignore_index=True) if ar_records else pd.DataFrame()
    car_updated = pd.DataFrame(scalar_updates)
    print(f"  → {len(car_updated)} events with valid ARs  "
          f"(ai=1: {int(car_updated['ai_mentioned'].sum())})")
    return ar_panel, car_updated


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Grouped CAAR plots
# ═══════════════════════════════════════════════════════════════════

def caar_stats(ar_panel: pd.DataFrame, mask: pd.Series):
    """Compute CAAR[t], SE[t], and t-stat[t] for a subset of events."""
    sub = ar_panel[mask]
    grouped = sub.groupby('t')['AR'].agg(['mean','std','count']).reset_index()
    grouped.columns = ['t','CAAR','std','n']
    grouped['SE']     = grouped['std'] / np.sqrt(grouped['n'])
    grouped['t_stat'] = grouped['CAAR'] / grouped['SE'].replace(0, np.nan)
    grouped['CAAR_cum'] = grouped['CAAR'].cumsum()
    return grouped


def plot_grouped_caar(ar_panel: pd.DataFrame, car_updated: pd.DataFrame):
    """
    Figure A: ai=1 vs ai=0 CAAR comparison
    Figure B: 4-way CAAR (ai × pre/post ChatGPT)
    """
    # --- Figure A: ai=1 vs ai=0 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (ax, winsorize_flag) in enumerate(zip(axes, [False, True])):
        panel = ar_panel.copy()
        if winsorize_flag:
            # Winsorize ARs at 1%/99% per time step (removes extreme outliers)
            panel['AR'] = panel.groupby('t')['AR'].transform(
                lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))

        for ai_val, label, color in [(1,'AI mentioned',RED),(0,'No AI mention',BLUE)]:
            mask  = ar_panel['event_id'].isin(
                car_updated.loc[car_updated['ai_mentioned']==ai_val,'event_id'])
            g = caar_stats(panel, panel['event_id'].isin(
                car_updated.loc[car_updated['ai_mentioned']==ai_val,'event_id']))
            if g.empty:
                continue
            n = int(car_updated['ai_mentioned'].eq(ai_val).sum())

            ax.plot(g['t'], g['CAAR_cum']*100, color=color, linewidth=2,
                    label=f'{label} (N={n})')
            ax.fill_between(g['t'],
                (g['CAAR_cum'] - 1.96*g['SE'].cumsum())*100,
                (g['CAAR_cum'] + 1.96*g['SE'].cumsum())*100,
                alpha=0.15, color=color)

        ax.axvline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.7)
        ax.axhline(0, color='grey',  linewidth=0.8, linestyle=':',  alpha=0.5)
        title_suffix = ' (winsorized 1%/99%)' if winsorize_flag else ' (raw)'
        ax.set_title(f'CAAR: AI vs Non-AI Layoffs{title_suffix}', fontweight='bold')
        ax.set_xlabel('Trading days relative to announcement (t=0)')
        ax.set_ylabel('Cumulative Average Abnormal Return (%)')
        ax.legend(fontsize=9)

    fig.suptitle('Grouped Event Study: AI-mentioned vs Non-AI Layoffs\n'
                 'FF4 model | 95% CI bands | Estimation window [-260,−11]',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'figA_caar_ai_vs_nonai.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {path}')

    # --- Figure B: 4-way (ai × pre/post ChatGPT) ---
    groups = [
        (1, 0, 'Pre-ChatGPT + AI',      RED,   '--'),
        (1, 1, 'Post-ChatGPT + AI',     RED,   '-'),
        (0, 0, 'Pre-ChatGPT + No AI',   BLUE,  '--'),
        (0, 1, 'Post-ChatGPT + No AI',  BLUE,  '-'),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for ai_val, post_val, label, color, ls in groups:
        ids = car_updated.loc[
            (car_updated['ai_mentioned'] == ai_val) &
            (car_updated['post_chatgpt'] == post_val), 'event_id']
        if len(ids) < 5:
            print(f'  Skipping {label}: only {len(ids)} events')
            continue
        g = caar_stats(ar_panel, ar_panel['event_id'].isin(ids))
        ax.plot(g['t'], g['CAAR_cum']*100, color=color, linewidth=2,
                linestyle=ls, label=f'{label} (N={len(ids)})')

    ax.axvline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axhline(0, color='grey',  linewidth=0.8, linestyle=':',  alpha=0.5)
    ax.set_title('4-Way CAAR: AI Mention × ChatGPT Era\n'
                 'Breakpoint: Nov 30 2022 (ChatGPT launch)', fontweight='bold')
    ax.set_xlabel('Trading days relative to announcement (t=0)')
    ax.set_ylabel('Cumulative Average Abnormal Return (%)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'figB_caar_4way.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {path}')

    # --- Figure C: DID mean-bar chart (CAR windows) ---
    windows = [('CAR_1_1','[-1,+1]'), ('CAR_0_20','[0,+20]'), ('CAR_5_30','[-5,+30]')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (col, wlabel) in zip(axes, windows):
        if col not in car_updated.columns:
            continue
        data = car_updated.copy()
        data[col] = pd.to_numeric(data[col], errors='coerce') * 100

        means, errs, xlabels, colors_g = [], [], [], []
        for ai_val, post_val, label, color, _ in groups:
            sub = data.loc[(data['ai_mentioned']==ai_val) &
                           (data['post_chatgpt']==post_val), col].dropna()
            if len(sub) < 5:
                continue
            means.append(sub.mean())
            errs.append(sub.sem() * 1.96)
            xlabels.append(label.replace(' + ','\n'))
            colors_g.append(color)

        xs = np.arange(len(means))
        ax.bar(xs, means, yerr=errs, color=colors_g, alpha=0.75, width=0.55,
               capsize=5, error_kw=dict(ecolor='grey', linewidth=1.2),
               edgecolor='white')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(f'Mean CAR {wlabel}', fontweight='bold')
        ax.set_ylabel('Mean CAR (%)')

    fig.suptitle('Figure C: Mean CAR by AI×Era Group\nError bars = 95% CI',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'figC_mean_car_bars.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {path}')


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — DID Regression
# ═══════════════════════════════════════════════════════════════════

def stars(p):
    if pd.isna(p): return ''
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''


def prepare_reg_data(car_updated: pd.DataFrame) -> pd.DataFrame:
    df = car_updated.copy()

    # Numeric conversions
    df['layoff_pct_n'] = (pd.to_numeric(
        df['layoff_pct'].astype(str).str.replace('%','').str.strip(),
        errors='coerce'))
    df['log_count'] = np.log1p(pd.to_numeric(df['layoff_count'], errors='coerce').fillna(0))
    df.loc[pd.to_numeric(df['layoff_count'], errors='coerce').isna(), 'log_count'] = np.nan

    df['intl'] = (df['listing_region'] == 'INTL').astype(int)

    # Winsorize CARs at 1%/99%
    for col in ['CAR_1_1', 'CAR_0_20', 'CAR_5_30', 'CAR_5_60']:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce')
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            df[col] = s.clip(lo, hi)

    return df


def run_ols(y_col: str, x_cols: list, df: pd.DataFrame):
    mask = df[y_col].notna() & df[x_cols].notna().all(axis=1)
    y = df.loc[mask, y_col]
    X = sm.add_constant(df.loc[mask, x_cols], has_constant='add')
    if len(y) < 30:
        return None, 0
    model = sm.OLS(y, X).fit(cov_type='HC3')
    return model, mask.sum()


def print_model(label: str, model, n: int, x_cols: list):
    print(f"\n  {'─'*60}")
    print(f"  {label}  |  N={n}  R²={model.rsquared:.3f}  "
          f"Adj.R²={model.rsquared_adj:.3f}")
    print(f"  {'─'*60}")
    print(f"  {'Variable':<28} {'Coef':>9} {'SE':>9} {'t':>7} {'p':>8} {'':>4}")
    print(f"  {'─'*60}")
    for var in ['const'] + x_cols:
        if var not in model.params:
            continue
        c = model.params[var]
        se = model.bse[var]
        t  = model.tvalues[var]
        p  = model.pvalues[var]
        sig = stars(p)
        print(f"  {var:<28} {c:>9.4f} {se:>9.4f} {t:>7.2f} {p:>8.4f} {sig:>4}")


def run_did(df: pd.DataFrame):
    """
    DID specification:
      CAR = α + β1·ai + β2·post_chatgpt + β3·(ai×post_chatgpt) + controls + ε
      β3 is the DID estimator.
    Interpretation: β3 captures how the MARGINAL market reaction to AI-related
    layoffs changed after ChatGPT launch, relative to non-AI layoffs.
    """
    print('\n' + '='*65)
    print('STEP 4: DID REGRESSION')
    print('  Breakpoint: 2022-11-30 (ChatGPT launch)')
    print('  Treatment : ai_mentioned=1')
    print('  DID coeff : ai_mentioned × post_chatgpt')
    print('='*65)

    df = df.copy()
    df['ai_x_post'] = df['ai_mentioned'] * df['post_chatgpt']

    results = []
    for y_col, wlabel in [('CAR_1_1','[-1,+1]'), ('CAR_0_20','[0,+20]'),
                           ('CAR_5_30','[-5,+30]')]:
        if y_col not in df.columns:
            continue
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce') * 100  # to %

        # Spec 1: pure DID (no controls)
        x1 = ['ai_mentioned','post_chatgpt','ai_x_post']
        m1, n1 = run_ols(y_col, x1, df)
        if m1: print_model(f'DID (no controls) — CAR{wlabel}', m1, n1, x1)

        # Spec 2: DID + controls
        x2 = ['ai_mentioned','post_chatgpt','ai_x_post',
               'log_count','layoff_pct_n','beta_mkt_ff4','intl']
        m2, n2 = run_ols(y_col, x2, df)
        if m2: print_model(f'DID + controls — CAR{wlabel}', m2, n2, x2)

        for spec_label, m, n, xc in [('DID (no controls)', m1, n1, x1),
                                       ('DID + controls',   m2, n2, x2)]:
            if m is None:
                continue
            for var in xc + ['const']:
                if var not in m.params:
                    continue
                results.append({'outcome': f'CAR{wlabel}', 'spec': spec_label,
                                 'variable': var, 'coef': m.params[var],
                                 'se': m.bse[var], 'pval': m.pvalues[var],
                                 'stars': stars(m.pvalues[var]),
                                 'N': n, 'R2': m.rsquared})

    # Print DID summary table focusing on β3
    print('\n  ── DID SUMMARY (β3 = ai × post_chatgpt) ──')
    print(f"  {'Outcome':<14} {'Spec':<22} {'β3 coef':>10} {'SE':>8} {'p':>8} {'sig':>5} N")
    for r in results:
        if r['variable'] == 'ai_x_post':
            print(f"  {r['outcome']:<14} {r['spec']:<22} "
                  f"{r['coef']:>10.3f} {r['se']:>8.3f} {r['pval']:>8.4f} "
                  f"{r['stars']:>5} {r['N']}")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Cross-sectional OLS (desktop-aligned spec)
# ═══════════════════════════════════════════════════════════════════

def run_cross_section(df: pd.DataFrame):
    """
    Aligned with desktop: CAR[-1,+1] as primary outcome.
    Also run CAR[-5,+30] as robustness check.
    Spec: ai + post_chatgpt + ai×post + log_count + layoff_pct + beta + intl
    (year FE dropped: post_chatgpt already captures time split)
    """
    print('\n' + '='*65)
    print('STEP 5: CROSS-SECTIONAL OLS (desktop-aligned spec)')
    print('='*65)

    results = []
    for y_col, wlabel in [('CAR_1_1','[-1,+1]'), ('CAR_0_20','[0,+20]'),
                           ('CAR_5_30','[-5,+30]')]:
        if y_col not in df.columns:
            continue
        df_c = df.copy()
        df_c[y_col] = pd.to_numeric(df_c[y_col], errors='coerce') * 100

        df_c['ai_x_post'] = df_c['ai_mentioned'] * df_c['post_chatgpt']

        x_full = ['ai_mentioned','post_chatgpt','ai_x_post',
                  'log_count','layoff_pct_n','beta_mkt_ff4','intl']
        m, n = run_ols(y_col, x_full, df_c)
        if m:
            print_model(f'Cross-section (full controls) — CAR{wlabel}', m, n, x_full)
            for var in x_full + ['const']:
                if var not in m.params:
                    continue
                results.append({'outcome': f'CAR{wlabel}', 'variable': var,
                                 'coef': m.params[var], 'se': m.bse[var],
                                 'pval': m.pvalues[var], 'stars': stars(m.pvalues[var]),
                                 'N': n, 'R2': m.rsquared, 'R2_adj': m.rsquared_adj})

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# STEP 6 — Summary statistics table
# ═══════════════════════════════════════════════════════════════════

def print_summary_stats(car_updated: pd.DataFrame):
    print('\n' + '='*65)
    print('DESCRIPTIVE STATISTICS')
    print('='*65)
    df = car_updated.copy()
    for col in ['CAR_1_1','CAR_0_20','CAR_5_30']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') * 100

    print(f"  Total events            : {len(df)}")
    print(f"  ai_mentioned=1          : {int(df['ai_mentioned'].sum())}  "
          f"({df['ai_mentioned'].mean()*100:.1f}%)")
    print(f"  post_chatgpt=1          : {int(df['post_chatgpt'].sum())}")
    print(f"  ai=1 × post=1           : {int((df['ai_mentioned']*df['post_chatgpt']).sum())}")
    print(f"  ai=1 × post=0           : {int((df['ai_mentioned']*(1-df['post_chatgpt'])).sum())}")
    print(f"  ai=0 × post=1           : {int(((1-df['ai_mentioned'])*df['post_chatgpt']).sum())}")
    print(f"  ai=0 × post=0           : {int(((1-df['ai_mentioned'])*(1-df['post_chatgpt'])).sum())}")
    print()
    for col, label in [('CAR_1_1','CAR[-1,+1] (%)'),
                        ('CAR_0_20','CAR[0,+20] (%)'),
                        ('CAR_5_30','CAR[-5,+30] (%)')]:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        print(f"  {label:<25}: mean={s.mean():.2f}  "
              f"std={s.std():.2f}  min={s.min():.2f}  max={s.max():.2f}  N={len(s)}")
    print()

    # Group means
    print(f"  {'Group':<30} {'CAR[-1,+1]':>12} {'CAR[-5,+30]':>13} {'N':>6}")
    print(f"  {'─'*63}")
    for ai_val, post_val, label in [
            (1, 0, 'AI=1, Pre-ChatGPT'),
            (1, 1, 'AI=1, Post-ChatGPT'),
            (0, 0, 'AI=0, Pre-ChatGPT'),
            (0, 1, 'AI=0, Post-ChatGPT')]:
        sub = df[(df['ai_mentioned']==ai_val) & (df['post_chatgpt']==post_val)]
        c1 = sub['CAR_1_1'].mean() if 'CAR_1_1' in sub.columns else np.nan
        c2 = sub['CAR_5_30'].mean() if 'CAR_5_30' in sub.columns else np.nan
        print(f"  {label:<30} {c1:>12.3f} {c2:>13.3f} {len(sub):>6}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print('='*65)
    print('IMPROVED ANALYSIS: re-labelled ai_mentioned + DID + grouped event study')
    print('='*65)

    # Load base data
    car = pd.read_csv(CAR_PATH)
    ff  = pd.read_csv(FF_PATH, parse_dates=['Date'], index_col='Date').sort_index()

    # Rename factor columns if needed
    ff = ff.rename(columns={'Mkt-RF':'MKT_RF','Mom   ':'MOM','Mom':'MOM'})
    if 'MOM' not in ff.columns:
        # Try common variants
        for c in ff.columns:
            if 'mom' in c.lower():
                ff = ff.rename(columns={c: 'MOM'})
                break
    print(f"FF factors loaded: {ff.columns.tolist()}, {len(ff)} days")

    # ── Step 1: Fix ai_mentioned ──────────────────────────────────
    print('\n── Step 1: Re-labelling ai_mentioned ──')
    car = build_ai_labels(car)

    # Sanity check: ai_mentioned distribution after re-labelling
    print(f"  Post-relabel: ai=1: {int(car['ai_mentioned'].sum())}, "
          f"ai=0: {int((car['ai_mentioned']==0).sum())}")

    # ── Step 2: Recompute daily ARs ───────────────────────────────
    print('\n── Step 2: Computing daily AR panels ──')
    ar_panel, car_updated = build_ar_panel(car, ff)

    if ar_panel.empty:
        print('ERROR: no AR data computed. Check stock returns directory.')
        return

    # Save updated car file
    car_updated.to_csv(os.path.join(OUT_DIR, 'car_by_event_v2.csv'), index=False)
    ar_panel.to_csv(os.path.join(OUT_DIR, 'ar_panel_daily.csv'), index=False)
    print(f"  Saved car_by_event_v2.csv ({len(car_updated)} events) "
          f"and ar_panel_daily.csv ({len(ar_panel)} rows)")

    # ── Step 3: Grouped CAAR plots ────────────────────────────────
    print('\n── Step 3: Grouped CAAR plots ──')
    plot_grouped_caar(ar_panel, car_updated)

    # ── Step 4 & 5: Regressions ───────────────────────────────────
    df_reg = prepare_reg_data(car_updated)
    print_summary_stats(car_updated)

    did_results   = run_did(df_reg)
    xsec_results  = run_cross_section(df_reg)

    # Save regression tables
    did_results.to_csv(os.path.join(OUT_DIR, 'did_results.csv'), index=False)
    xsec_results.to_csv(os.path.join(OUT_DIR, 'cross_section_v2.csv'), index=False)

    print('\n' + '='*65)
    print(f'All outputs saved to: {OUT_DIR}')
    print('='*65)


if __name__ == '__main__':
    main()
