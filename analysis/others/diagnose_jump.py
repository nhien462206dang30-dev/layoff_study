"""
Diagnose the CAAR jump around day +45 in the Post-GenAI subsample.
Checks: sample drop-off, outlier ARs, and composition change.
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from event_study import load_data, run_single_event

BASE       = '/Users/irmina/Documents/Claude/layoff_study'
FIGS_DIR   = os.path.join(BASE, 'data/results/figures')
RESULTS_DIR = os.path.join(BASE, 'data/results')

def main():
    print('Loading data and running FF4 for Post-GenAI events...')
    events, ff = load_data()
    post_events = events[events['announcement_date'].dt.year >= 2023]
    print(f'  Post-GenAI events: {len(post_events)}')

    results = []
    for _, row in post_events.iterrows():
        r = run_single_event(row, ff, model='ff4')
        if r:
            results.append(r)
    print(f'  Processed: {len(results)}')

    # ── 1. Count how many events have data at each relative day ──────────────
    day_counts = {}
    day_mean_ar = {}
    for r in results:
        daily = r['daily']
        for _, drow in daily.iterrows():
            rd = int(drow['rel_day'])
            if rd not in day_counts:
                day_counts[rd] = 0
                day_mean_ar[rd] = []
            day_counts[rd] += 1
            day_mean_ar[rd].append(drow['AR'])

    days_sorted = sorted(day_counts.keys())
    counts = [day_counts[d] for d in days_sorted]
    mean_ars = [np.mean(day_mean_ar[d]) for d in days_sorted]

    print('\nSample size and mean AR by relative day (days 30-60):')
    print(f'  {"Day":>5}  {"N events":>10}  {"Mean AR%":>10}  {"Max AR%":>10}  {"Min AR%":>10}')
    for d in range(30, 61):
        if d in day_counts:
            ars = day_mean_ar[d]
            print(f'  {d:>5}  {len(ars):>10}  {np.mean(ars)*100:>10.4f}  '
                  f'{np.max(ars)*100:>10.4f}  {np.min(ars)*100:>10.4f}')

    # ── 2. Find events that drop out between day 40 and 50 ──────────────────
    events_at_40 = set()
    events_at_50 = set()
    for r in results:
        daily = r['daily']
        if (daily['rel_day'] == 40).any():
            events_at_40.add(r['ticker'])
        if (daily['rel_day'] == 50).any():
            events_at_50.add(r['ticker'])

    dropped = events_at_40 - events_at_50
    print(f'\nEvents present at day +40 but NOT at day +50: {len(dropped)}')
    if dropped:
        print('  Dropped tickers:', sorted(dropped))

    # ── 3. Check if dropped events have extreme CARs ─────────────────────────
    print('\nCAR [-5,+60] for events that drop out early:')
    from event_study import compute_cars
    for r in results:
        if r['ticker'] in dropped:
            cars = compute_cars(r)
            last_day = r['daily']['rel_day'].max()
            print(f"  {r['ticker']:>12}  last_day={last_day:>4}  "
                  f"CAR[-5+60]={cars.get('[-5,+60]', float('nan'))*100:>8.3f}%  "
                  f"date={r['announcement_date'].date()}")

    # ── 4. Identify large AR outliers around days 40-50 ─────────────────────
    print('\nTop 10 largest |AR| on any day between +40 and +50 (Post-GenAI):')
    outlier_rows = []
    for r in results:
        daily = r['daily']
        mask = (daily['rel_day'] >= 40) & (daily['rel_day'] <= 50)
        for _, drow in daily[mask].iterrows():
            outlier_rows.append({
                'ticker': r['ticker'],
                'company': r['company_fyi'],
                'rel_day': int(drow['rel_day']),
                'AR_pct': drow['AR'] * 100,
                'abs_AR': abs(drow['AR']),
                'date': drow['date'],
            })
    outlier_df = pd.DataFrame(outlier_rows).sort_values('abs_AR', ascending=False)
    print(outlier_df.head(10).to_string(index=False))

    # ── 5. Plot: N events per day + mean AR per day ──────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(days_sorted, counts, color='#2166ac', linewidth=2)
    ax1.axvspan(40, 50, color='orange', alpha=0.15, label='Days 40–50 (suspect zone)')
    ax1.axvline(45, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Day +45')
    ax1.set_ylabel('Number of Events with Data', fontsize=11)
    ax1.set_title('Post-GenAI Subsample: Diagnostic — Sample Drop-off and Mean AR by Day', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.bar(days_sorted, [v * 100 for v in mean_ars],
            color=['red' if v < 0 else '#2166ac' for v in mean_ars],
            alpha=0.6, width=0.8)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.axvspan(40, 50, color='orange', alpha=0.15)
    ax2.axvline(45, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Mean Abnormal Return (%) per Day', fontsize=11)
    ax2.set_xlabel('Relative Trading Day', fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, 'diag_jump_analysis.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nDiagnostic plot saved → {out}')

    # ── 6. CAAR paths with and without the outlier events ───────────────────
    from event_study import compute_daily_caar

    # Find tickers with extreme CAR in the +40 to +50 range
    threshold = outlier_df['abs_AR'].quantile(0.95)
    extreme_tickers = set(outlier_df[outlier_df['abs_AR'] > threshold]['ticker'].unique())
    print(f'\nTickers with |AR| > {threshold*100:.2f}% on any day +40 to +50: {extreme_tickers}')

    clean_results = [r for r in results if r['ticker'] not in extreme_tickers]
    print(f'Post-GenAI sample: {len(results)} total → {len(clean_results)} after removing extreme outliers')

    fig2, ax = plt.subplots(figsize=(11, 5))
    for res, color, label in [
        (results,       '#b2182b', f'Original Post-GenAI (N={len(results)})'),
        (clean_results, '#2166ac', f'Outliers removed (N={len(clean_results)})'),
    ]:
        days, caar, lo, hi = compute_daily_caar(res)
        ax.plot(days, caar, color=color, linewidth=2, label=label)
        ax.fill_between(days, lo, hi, color=color, alpha=0.10)

    ax.axvline(0,  color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(0,  color='grey',  linestyle='-',  linewidth=0.5, alpha=0.5)
    ax.axvspan(40, 50, color='orange', alpha=0.12, label='Days 40–50 (suspect zone)')
    ax.set_xlabel('Trading Days Relative to Announcement', fontsize=12)
    ax.set_ylabel('CAAR (%)', fontsize=12)
    ax.set_title('Post-GenAI CAAR: Original vs. Outlier-Removed', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(FIGS_DIR, 'diag_caar_outlier_comparison.png')
    fig2.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f'Comparison plot saved → {out2}')

if __name__ == '__main__':
    main()
