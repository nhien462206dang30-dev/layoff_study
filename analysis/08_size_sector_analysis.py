"""
Size × Sector Heterogeneity Analysis
=====================================
Tests whether the large-firm (positive CAAR) vs small-firm (negative CAAR)
pattern documented in the main analysis interacts with tech vs non-tech sector
classification.

3×2 matrix:
  Rows:    Mega-cap / Mid-cap / Small-cap  (proxy: R² of FF4 estimation window)
  Columns: Core tech vs Non-tech           (layoffs.fyi industry taxonomy)

Firm size proxy: R² of the FF4 factor model estimated over the pre-event
  estimation window [-260, -11].
  - Rationale: higher R² means the stock's return is better explained by
    systematic market-wide factors, which is characteristic of large, liquid,
    analyst-covered stocks. Low R² = high idiosyncratic risk = smaller or
    more financially stressed firm. This data is already in car_by_event.csv
    and requires no external API calls.
  - This is a standard liquidity/coverage proxy in the event study literature.

Output:
  data/results/size_sector/
    size_sector_caar.csv        — CAAR summary for each group
    fig_size_sector_2x2.png     — bar chart comparison
    fig_size_sector_box.png     — CAR distribution box plots
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE    = '/Users/irmina/Documents/Claude/layoff_study'
RES     = os.path.join(BASE, 'data/results')
OUT_DIR = os.path.join(RES, 'size_sector')
os.makedirs(OUT_DIR, exist_ok=True)

CORE_TECH_INDUSTRIES = {
    'Hardware', 'Security', 'Data', 'Infrastructure', 'Marketing',
    'Media', 'Support', 'AI', 'Crypto', 'Product', 'Education',
    'Recruiting', 'HR', 'Other',
}


# ── 1. CAAR stats via simple cross-sectional t-test ──────────────────
def caar_stats(cars: np.ndarray, label: str = '') -> dict:
    """
    For subsample CAAR comparisons a simple cross-sectional t-test is used.
    This is standard in heterogeneity analysis sections of event study papers.
    Input cars are in DECIMAL form (e.g. -0.0096 = -0.96%).
    Output CAAR_pct is in PERCENT form (multiplied by 100).
    """
    cars = cars[~np.isnan(cars)]
    N = len(cars)
    if N < 5:
        return {'label': label, 'N': N, 'CAAR_pct': np.nan, 't': np.nan, 'p': np.nan, 'sig': ''}
    mean_pct = np.mean(cars) * 100   # convert to percent
    se_pct   = np.std(cars, ddof=1) / np.sqrt(N) * 100
    t    = mean_pct / se_pct if se_pct > 0 else np.nan
    p    = 2 * stats.t.sf(abs(t), df=N-1) if not np.isnan(t) else np.nan
    sig  = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
    return {'label': label, 'N': N, 'CAAR_pct': mean_pct, 't': t, 'p': p, 'sig': sig}


def stars(p):
    if pd.isna(p): return ''
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))


# ── 3. Main ──────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('SIZE × SECTOR HETEROGENEITY ANALYSIS')
    print('=' * 65)

    # Load per-event CARs
    cars_path = os.path.join(RES, 'car_by_event.csv')
    df = pd.read_csv(cars_path)
    df = df[df['listing_region'] == 'US'].copy()
    df['announcement_date'] = pd.to_datetime(df['announcement_date'])
    print(f'\nUS-only events: {len(df)}')

    # Sector classification
    df['is_tech'] = df['industry'].isin(CORE_TECH_INDUSTRIES)
    df['sector']  = df['is_tech'].map({True: 'Core tech', False: 'Non-tech'})

    # ── Firm size proxy: R² of FF4 estimation-window regression ──
    # Already in car_by_event.csv — no external API needed.
    # Higher R² = stock explained by systematic factors = large, liquid, covered.
    # Three tiers by quartile of R².
    q25 = df['r2_ff4'].quantile(0.25)
    q75 = df['r2_ff4'].quantile(0.75)
    print(f'  R² quartiles: Q25={q25:.3f}  Q75={q75:.3f}  median={df["r2_ff4"].median():.3f}')

    def size_tier(r2):
        if r2 >= q75: return 'High R² (top 25%)\n≈ large/liquid'
        elif r2 >= q25: return 'Mid R² (25–75%)'
        else: return 'Low R² (bot 25%)\n≈ small/idiosyncratic'
    df['size_group'] = df['r2_ff4'].apply(size_tier)
    size_proxy_label = 'FF4 estimation-window R² (quartile split; high R² ≈ large/liquid firm)'

    print(f'  Size proxy: {size_proxy_label}')
    print(f'\n  Group sizes:')
    ct = df.groupby(['size_group', 'sector']).size().unstack(fill_value=0)
    print(ct.to_string())

    # ── CAAR analysis for each 2×2 cell ──
    windows = {
        'CAR[-1,+1]': 'CAR_1_1',
        'CAR[0,+20]': 'CAR_0_20',
        'CAR[-5,+60]': 'CAR_5_60',
    }

    print(f'\n{"─"*65}')
    print(f'CAAR by Size × Sector Group')
    print(f'{"─"*65}')

    result_rows = []
    for wname, wcol in windows.items():
        print(f'\n  Window: {wname}')
        print(f'  {"Group":<42} {"N":>4} {"CAAR%":>8} {"t":>6} {"p":>7} {"Sig":>4}')
        print(f'  {"─"*68}')

        # ── 3-tier × sector ──
        tiers = ['High R² (top 25%)\n≈ large/liquid', 'Mid R² (25–75%)', 'Low R² (bot 25%)\n≈ small/idiosyncratic']
        for size in tiers:
            for sector in ['Core tech', 'Non-tech']:
                mask = (df['size_group'] == size) & (df['sector'] == sector)
                cars = df.loc[mask, wcol].values.astype(float)
                label = f'{size} × {sector}'
                res = caar_stats(cars, label=label)
                res.update({'window': wname, 'size_group': size, 'sector': sector})
                result_rows.append(res)
                caar_str = f"{res['CAAR_pct']:.3f}%" if not np.isnan(res.get('CAAR_pct', np.nan)) else '—'
                t_str    = f"{res['t']:.2f}"          if not np.isnan(res.get('t', np.nan))        else '—'
                p_str    = f"{res['p']:.4f}"           if not np.isnan(res.get('p', np.nan))        else '—'
                print(f'  {label:<42} {res["N"]:>4} {caar_str:>9} {t_str:>6} {p_str:>7} {res["sig"]:>4}')
            # tier marginal
            mask = df['size_group'] == size
            cars = df.loc[mask, wcol].values.astype(float)
            res = caar_stats(cars, label=f'  {size} [all sectors]')
            caar_str = f"{res['CAAR_pct']:.3f}%" if not np.isnan(res.get('CAAR_pct', np.nan)) else '—'
            t_str    = f"{res['t']:.2f}"          if not np.isnan(res.get('t', np.nan))        else '—'
            p_str    = f"{res['p']:.4f}"           if not np.isnan(res.get('p', np.nan))        else '—'
            print(f'  {res["label"]:<42} {res["N"]:>4} {caar_str:>9} {t_str:>6} {p_str:>7} {res["sig"]:>4}')
            print()

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(os.path.join(OUT_DIR, 'size_sector_caar.csv'), index=False)
    print(f'\n  Saved: {os.path.join(OUT_DIR, "size_sector_caar.csv")}')

    # ── Plot 1: Bar chart — 3 windows × 3-tier × sector ──
    tiers = ['High R² (top 25%)\n≈ large/liquid', 'Mid R² (25–75%)', 'Low R² (bot 25%)\n≈ small/idiosyncratic']
    tier_short = ['High R²\n(top 25%)', 'Mid R²\n(25–75%)', 'Low R²\n(bot 25%)']
    colors_tech    = ['#1a6faf', '#5aafd4', '#99d4ec']  # blue family = core tech
    colors_nontech = ['#b2182b', '#e6614d', '#fdbf6f']  # red/orange family = non-tech

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (wname, wcol) in zip(axes, windows.items()):
        sub = results_df[results_df['window'] == wname]
        sub = sub[sub['sector'].isin(['Core tech', 'Non-tech'])]

        x = np.arange(len(tiers))
        w = 0.35
        for j, (sector, col_set, offset) in enumerate([('Core tech', colors_tech, -w/2),
                                                        ('Non-tech', colors_nontech, w/2)]):
            grp = sub[sub['sector'] == sector].set_index('size_group')
            caars = [grp.loc[t, 'CAAR_pct'] if t in grp.index else np.nan for t in tiers]
            ns    = [int(grp.loc[t, 'N'])    if t in grp.index else 0        for t in tiers]
            sigs  = [grp.loc[t, 'sig']       if t in grp.index else ''        for t in tiers]
            bars  = ax.bar(x + offset, caars, w, color=col_set, alpha=0.85,
                           edgecolor='white', linewidth=0.5, label=sector)
            for i, (c, s, n) in enumerate(zip(caars, sigs, ns)):
                if not np.isnan(c) and s:
                    ax.text(x[i]+offset, c + (0.1 if c >= 0 else -0.4),
                            s, ha='center', fontsize=9)
                ax.text(x[i]+offset, min([v for v in caars if not np.isnan(v)] or [0])-1.5,
                        f'n={n}', ha='center', fontsize=7, color='#555')

        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(tier_short, fontsize=8)
        ax.set_ylabel('CAAR (%)', fontsize=11)
        ax.set_title(f'CAAR {wname}', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('white')

    fig.suptitle('CAAR by Firm Size (Market Cap Tercile) × Sector\nUS-only, FF4 Model',
                 fontsize=12, fontweight='bold')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    save1 = os.path.join(OUT_DIR, 'fig_size_sector_2x2.png')
    plt.savefig(save1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save1}')

    # ── Plot 2: Box plot by 3-tier × sector ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, sector in zip(axes, ['Core tech', 'Non-tech']):
        group_data, group_labels, group_colors = [], [], []
        palette = ['#1a6faf', '#5aafd4', '#99d4ec'] if sector == 'Core tech' else ['#b2182b', '#e6614d', '#fdbf6f']
        for tier, short, color in zip(tiers, tier_short, palette):
            mask = (df['size_group'] == tier) & (df['sector'] == sector)
            cars = df.loc[mask, 'CAR_1_1'].dropna().values * 100
            group_data.append(cars)
            group_labels.append(f'{short}\n(N={len(cars)})')
            group_colors.append(color)

        bp = ax.boxplot(group_data, labels=group_labels, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1.5},
                        whiskerprops={'linewidth': 1.0},
                        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4})
        for patch, color in zip(bp['boxes'], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_title(f'{sector}', fontsize=13)
        ax.set_ylabel('CAR[−1,+1] (%)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('white')

    fig.suptitle('CAR[−1,+1] Distribution: Firm Size × Sector\nUS-only, FF4 Model', fontsize=13)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    save2 = os.path.join(OUT_DIR, 'fig_size_sector_box.png')
    plt.savefig(save2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {save2}')

    # ── Summary ──
    print(f'\n{"="*65}')
    print('INTERPRETATION GUIDE')
    print(f'{"="*65}')
    r11 = results_df[results_df['window'] == 'CAR[-1,+1]']
    for tier in tiers:
        for sector in ['Core tech', 'Non-tech']:
            row = r11[(r11['size_group'] == tier) & (r11['sector'] == sector)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            if np.isnan(row['CAAR_pct']):
                continue
            direction = 'POSITIVE' if row['CAAR_pct'] > 0 else 'NEGATIVE'
            sig_note  = f"({row['sig']})" if row['sig'] else '(not sig)'
            print(f'  {tier:<28} × {sector:<12}: '
                  f'{row["CAAR_pct"]:+.2f}%  {direction} {sig_note}  N={row["N"]}')

    print(f'\n  Size proxy: {size_proxy_label}')
    print(f'  Saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
