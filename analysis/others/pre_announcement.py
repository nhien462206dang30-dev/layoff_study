"""
Pre-Announcement Drift Analysis
================================

If markets anticipate layoff announcements before t=0 (e.g., through news leaks,
insider trading, or gradual information diffusion), we should see significant
cumulative abnormal returns in the pre-announcement window [−20, −1].

This script:
  1. Computes the CAAR path from t=−20 to t=+30 with daily confidence bands.
  2. Tests whether pre-announcement drift (CAR[−20,−2] and CAR[−20,−1]) is
     significantly different from zero using a t-test.
  3. Examines whether the pre-announcement drift predicts the announcement-day
     reaction (price discovery test): does leakage reduce the t=0 surprise?
  4. Runs these analyses separately for AI-mentioned vs. non-AI layoffs to
     check if information leakage differs by type.

Primary spec: US-only. AR data is in decimal form (0.01 = 1%).

Outputs (data/results/robustness/):
  fig_pre_announcement.png     — full CAAR path [−20, +30] with pre/post split
  pre_announcement_stats.csv   — pre-announcement CAR statistics
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

BASE        = '/Users/irmina/Documents/Claude/layoff_study'
AR_PATH     = os.path.join(BASE, 'data/results/improved/ar_panel_daily.csv')
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


def stars(p):
    if pd.isna(p): return ''
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    panel = pd.read_csv(AR_PATH)
    panel['announcement_date'] = pd.to_datetime(panel['announcement_date'])

    # Bring in listing_region from master
    master = pd.read_csv(MASTER_PATH)
    master['announcement_date'] = pd.to_datetime(master['announcement_date'])
    meta = master[['ticker', 'announcement_date', 'listing_region']].drop_duplicates(
        subset=['ticker', 'announcement_date'], keep='first'
    )
    panel = panel.merge(meta, on=['ticker', 'announcement_date'], how='left')

    # Bring in tiered AI labels from final_labels_and_cars
    cars = pd.read_csv(CARS_PATH)
    cars['announcement_date'] = pd.to_datetime(cars['announcement_date'])
    ai_cols = ['ticker', 'announcement_date', 'ai_causal', 'ai_primary', 'ai_broad']
    ai_cols = [c for c in ai_cols if c in cars.columns]
    panel = panel.merge(cars[ai_cols].drop_duplicates(subset=['ticker','announcement_date']),
                        on=['ticker', 'announcement_date'], how='left')

    panel_us = panel[panel['listing_region'] == 'US'].copy()
    print(f'AR panel loaded: {len(panel)} rows  |  US: {len(panel_us)} rows')
    print(f'  Events (US): {panel_us["event_id"].nunique()}')
    print(f'  t range: {panel["t"].min()} to {panel["t"].max()}')
    return panel_us


# ══════════════════════════════════════════════════════════════════════════════
# 2. Compute cumulative CAR per event up to each day
# ══════════════════════════════════════════════════════════════════════════════

def build_cumulative_cars(panel, t_min=-20, t_max=30):
    """
    For each event, compute cumulative AR from t_min to each day t.
    Returns a wide DataFrame: rows = events, columns = t values.
    """
    events = panel['event_id'].unique()
    cum_cars = {}

    for ev_id in events:
        ev_data = panel[panel['event_id'] == ev_id].sort_values('t')
        ev_data = ev_data[(ev_data['t'] >= t_min) & (ev_data['t'] <= t_max)]
        cumulative = 0.0
        day_dict = {}
        for _, row in ev_data.iterrows():
            cumulative += row['AR']
            day_dict[int(row['t'])] = cumulative
        cum_cars[ev_id] = day_dict

    # Convert to DataFrame (events × days)
    all_days = range(t_min, t_max + 1)
    df_cum = pd.DataFrame(cum_cars).T  # events × days
    # Ensure all day columns exist
    for d in all_days:
        if d not in df_cum.columns:
            df_cum[d] = np.nan

    return df_cum[[d for d in all_days if d in df_cum.columns]]


def caar_with_ci(cum_df, days):
    """Compute CAAR and 95% CI across events for each day."""
    caar = []
    ci_lo = []
    ci_hi = []
    n_obs = []
    for d in days:
        if d not in cum_df.columns:
            caar.append(np.nan); ci_lo.append(np.nan); ci_hi.append(np.nan); n_obs.append(0)
            continue
        vals = cum_df[d].dropna()
        n = len(vals)
        mean_c = vals.mean() * 100   # to percent
        if n > 1:
            se = vals.std() / np.sqrt(n) * 100
            ci_lo.append(mean_c - 1.96 * se)
            ci_hi.append(mean_c + 1.96 * se)
        else:
            ci_lo.append(mean_c); ci_hi.append(mean_c)
        caar.append(mean_c)
        n_obs.append(n)
    return np.array(caar), np.array(ci_lo), np.array(ci_hi), np.array(n_obs)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Test pre-announcement drift significance
# ══════════════════════════════════════════════════════════════════════════════

def test_pre_drift(panel, cum_df):
    print('\n' + '=' * 65)
    print('PRE-ANNOUNCEMENT DRIFT TESTS (US-only, FF4 abnormal returns)')
    print('=' * 65)

    rows = []
    # Test CAR[−20, −2] and CAR[−20, −1]
    for window_end, wlabel in [(-2, 'CAR[−20, −2]'), (-1, 'CAR[−20, −1]'), (0, 'CAR[−20, 0]')]:
        col = window_end
        if col not in cum_df.columns:
            continue
        vals = cum_df[col].dropna() * 100  # to percent
        n = len(vals)
        mean_c = vals.mean()
        se_c   = vals.std() / np.sqrt(n)
        t_c    = mean_c / se_c if se_c > 0 else np.nan
        p_c    = 2 * (1 - stats.t.cdf(abs(t_c), df=n-1)) if not np.isnan(t_c) else np.nan
        sig    = stars(p_c)
        print(f'  {wlabel:<20}: mean = {mean_c:>7.3f}%  SE = {se_c:.3f}  '
              f't = {t_c:.3f}  p = {p_c:.4f}  {sig}  N = {n}')
        rows.append({'window': wlabel, 'mean_car_pct': mean_c, 'se': se_c,
                     't': t_c, 'p': p_c, 'stars': sig, 'N': n})

    # Test by group: AI vs non-AI (post-ChatGPT)
    print(f'\n  Subgroup: Post-ChatGPT only')
    ev_meta = panel.drop_duplicates('event_id').set_index('event_id')
    for ai_val, label in [(1, 'AI mentioned'), (0, 'AI not mentioned')]:
        post_ids = ev_meta[(ev_meta['post_chatgpt'] == ai_val) |
                           (ev_meta.get('ai_broad', 0) == ai_val)].index
        # Use ai_mentioned from panel as proxy
        ai_ids  = ev_meta[ev_meta['ai_mentioned'] == ai_val].index
        post_ai_ids = ev_meta[(ev_meta['post_chatgpt'] == 1) &
                              (ev_meta['ai_mentioned'] == ai_val)].index
        sub_df  = cum_df[cum_df.index.isin(post_ai_ids)]
        if len(sub_df) < 5:
            continue
        for window_end, wlabel in [(-1, 'CAR[−20, −1]'), (0, 'CAR[−20, 0]')]:
            if window_end not in sub_df.columns:
                continue
            vals = sub_df[window_end].dropna() * 100
            n = len(vals)
            mean_c = vals.mean()
            se_c   = vals.std() / np.sqrt(n)
            t_c    = mean_c / se_c if se_c > 0 else np.nan
            p_c    = 2*(1-stats.t.cdf(abs(t_c),df=n-1)) if not np.isnan(t_c) else np.nan
            print(f'  Post-ChatGPT, {label:<22} {wlabel}: mean={mean_c:.3f}%  '
                  f't={t_c:.2f}  p={p_c:.4f}  {stars(p_c)}  N={n}')
            rows.append({'window': f'Post-ChatGPT, {label}, {wlabel}',
                         'mean_car_pct': mean_c, 'se': se_c, 't': t_c, 'p': p_c,
                         'stars': stars(p_c), 'N': n})

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Price discovery test: does pre-drift predict announcement reaction?
# ══════════════════════════════════════════════════════════════════════════════

def price_discovery_test(cum_df):
    """
    Regress CAR[0,+1] on pre-announcement drift CAR[-20,-2].
    H: negative β (if market anticipates, announcement surprise is smaller).
    """
    import statsmodels.api as sm

    print('\n' + '=' * 65)
    print('PRICE DISCOVERY TEST')
    print('  Reg: CAR[0,+1] = α + β · CAR[−20,−2] + ε')
    print('  H (leakage): β < 0 — pre-drift absorbs announcement surprise')
    print('  H (momentum): β > 0 — pre-drift continues into announcement')
    print('=' * 65)

    # CAR[0,+1] = cumulative AR at t=1 minus cumulative AR at t=-1
    if 1 not in cum_df.columns or -2 not in cum_df.columns:
        print('  Required columns not available.')
        return pd.DataFrame()

    # CAR[0,+1]: from t=0 to t=+1 = cum_df[1] - cum_df[-1]
    if -1 in cum_df.columns:
        car_announce = (cum_df[1] - cum_df[-1]) * 100
    else:
        car_announce = cum_df[1] * 100

    car_predrift = cum_df[-2] * 100  # CAR[-20,-2]

    valid = car_announce.notna() & car_predrift.notna()
    y = car_announce[valid]
    X = sm.add_constant(car_predrift[valid])
    model = sm.OLS(y, X).fit(cov_type='HC3')

    b = model.params.get('key', model.params.iloc[-1])
    b_name = 'CAR[-20,-2]'
    b_val  = model.params.get(b_name, model.params.iloc[-1])
    p_val  = model.pvalues.get(b_name, model.pvalues.iloc[-1])
    se_val = model.bse.get(b_name, model.bse.iloc[-1])

    print(f'  N = {valid.sum()}  R² = {model.rsquared:.3f}')
    print(f'  β (CAR pre-drift) = {b_val:.3f}  SE = {se_val:.3f}  '
          f'p = {p_val:.4f}  {stars(p_val)}')
    if b_val < 0 and p_val < 0.1:
        print('  → Evidence of price discovery (leakage): pre-drift negatively predicts announcement CAR.')
    elif b_val > 0 and p_val < 0.1:
        print('  → Evidence of momentum: pre-drift positively continues into announcement.')
    else:
        print('  → No significant relation (consistent with efficient price discovery or no leakage).')

    return pd.DataFrame([{'variable': b_name, 'coef': b_val, 'se': se_val,
                           'pval': p_val, 'stars': stars(p_val), 'R2': model.rsquared,
                           'N': valid.sum()}])


# ══════════════════════════════════════════════════════════════════════════════
# 5. Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_pre_announcement(panel, cum_df_all, cum_df_ai, cum_df_nonai):
    days = list(range(-20, 31))

    caar_all, lo_all, hi_all, _ = caar_with_ci(cum_df_all, days)
    caar_ai,  lo_ai,  hi_ai,  _ = caar_with_ci(cum_df_ai,  days)
    caar_na,  lo_na,  hi_na,  _ = caar_with_ci(cum_df_nonai, days)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Full sample CAAR with pre/post shading
    ax = axes[0]
    pre_days  = [d for d in days if d <= -1]
    post_days = [d for d in days if d >= 0]
    pre_idx   = [i for i, d in enumerate(days) if d <= -1]
    post_idx  = [i for i, d in enumerate(days) if d >= 0]

    ax.fill_between(pre_days,
                    [lo_all[i] for i in pre_idx],
                    [hi_all[i] for i in pre_idx],
                    alpha=0.15, color=GREY, label='Pre-announcement 95% CI')
    ax.fill_between(post_days,
                    [lo_all[i] for i in post_idx],
                    [hi_all[i] for i in post_idx],
                    alpha=0.15, color=BLUE)
    ax.plot(days, caar_all, color=BLUE, linewidth=2,
            label=f'CAAR (N={cum_df_all.shape[0]})')
    ax.axvline(0,  color='black', linewidth=1, linestyle='--', alpha=0.7, label='Announcement (t=0)')
    ax.axhline(0,  color='grey',  linewidth=0.6, alpha=0.5)
    ax.axvspan(-20, -0.5, alpha=0.04, color=GREY)

    # Label pre-announcement CAR
    pre_caar_val = caar_all[days.index(-1)] if -1 in days else np.nan
    ax.annotate(f'Pre-drift\n({pre_caar_val:.2f}%)',
                xy=(-1, pre_caar_val), xytext=(-15, pre_caar_val + 0.4),
                arrowprops=dict(arrowstyle='->', color=GREY),
                fontsize=9, color=GREY)

    ax.set_xlabel('Event Day (relative to announcement)', fontsize=11)
    ax.set_ylabel('CAAR (%)', fontsize=11)
    ax.set_title('Full Sample CAAR: t=−20 to t=+30\n(US-only, FF4, shaded = pre-announcement window)',
                 fontsize=10)
    ax.legend(fontsize=9.5, frameon=True)

    # Panel 2: AI vs non-AI pre-announcement comparison
    ax2 = axes[1]
    ax2.fill_between(days, lo_ai,  hi_ai,  alpha=0.12, color=RED)
    ax2.fill_between(days, lo_na,  hi_na,  alpha=0.12, color=BLUE)
    ax2.plot(days, caar_ai, color=RED,  linewidth=1.8,
             label=f'AI mentioned (N={cum_df_ai.shape[0]})')
    ax2.plot(days, caar_na, color=BLUE, linewidth=1.8,
             label=f'AI not mentioned (N={cum_df_nonai.shape[0]})')
    ax2.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
    ax2.axhline(0, color='grey',  linewidth=0.6, alpha=0.5)
    ax2.axvspan(-20, -0.5, alpha=0.04, color=GREY)

    ax2.set_xlabel('Event Day (relative to announcement)', fontsize=11)
    ax2.set_ylabel('CAAR (%)', fontsize=11)
    ax2.set_title('CAAR by AI Mention: Pre-announcement Comparison\n(Post-ChatGPT events, US-only)',
                  fontsize=10)
    ax2.legend(fontsize=9.5, frameon=True)

    fig.suptitle('Pre-Announcement Drift Analysis\n'
                 'Significant pre-t=0 drift would indicate anticipation / information leakage',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_pre_announcement.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\n  Figure saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 65)
    print('PRE-ANNOUNCEMENT DRIFT ANALYSIS')
    print('=' * 65)

    panel = load_data()

    # Subgroups for Panel 2
    post_ai_events  = panel[(panel['post_chatgpt'] == 1) & (panel['ai_mentioned'] == 1)]['event_id'].unique()
    post_nonai_events = panel[(panel['post_chatgpt'] == 1) & (panel['ai_mentioned'] == 0)]['event_id'].unique()

    cum_all    = build_cumulative_cars(panel)
    cum_ai     = build_cumulative_cars(panel[panel['event_id'].isin(post_ai_events)])
    cum_nonai  = build_cumulative_cars(panel[panel['event_id'].isin(post_nonai_events)])

    print(f'\n  Built cumulative CAR matrices:')
    print(f'  Full sample:   {cum_all.shape[0]} events × {cum_all.shape[1]} days')
    print(f'  Post-ChatGPT AI:     {cum_ai.shape[0]} events')
    print(f'  Post-ChatGPT Non-AI: {cum_nonai.shape[0]} events')

    stats_df = test_pre_drift(panel, cum_all)
    pd_df    = price_discovery_test(cum_all)
    plot_pre_announcement(panel, cum_all, cum_ai, cum_nonai)

    # Save
    stats_df.to_csv(os.path.join(OUT_DIR, 'pre_announcement_stats.csv'), index=False)
    print(f'\n  Stats saved → {OUT_DIR}/pre_announcement_stats.csv')

    print(f'\nAll outputs saved to: {OUT_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
