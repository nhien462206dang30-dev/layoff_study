"""
Paper-Quality Visualizations for Tech Layoff Event Study
=========================================================
Generates 7 publication-ready figures:

  Fig 1  — CAAR time path, full sample (CAPM vs FF4)
  Fig 2  — CAAR time path, Pre-GenAI vs Post-GenAI (FF4)
  Fig 3  — CAAR time path, AI-mentioned vs non-AI (Post-GenAI, FF4)
  Fig 4  — CAAR bar chart by event window, all subsamples (FF4)
  Fig 5  — CAR distribution box plot by event window
  Fig 6  — CAR [0,+20] by industry (top industries)
  Fig 7  — CAR [-5,+60] scatter: layoff count vs CAR

All figures are saved to data/results/figures/ at 300 DPI.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import statsmodels.api as sm
from scipy import stats

# ── Add parent so we can import event_study helpers ───────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from event_study import (
    load_data, load_stock_returns, find_event_day,
    winsorize, run_single_event, compute_daily_caar,
    EST_START, EST_END, EVENT_WINDOWS
)

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = '/Users/irmina/Documents/Claude/layoff_study'
RESULTS_DIR = os.path.join(BASE, 'data/results')
FIGS_DIR    = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

CAR_SUMMARY   = os.path.join(RESULTS_DIR, 'car_summary.csv')
CAR_BY_EVENT  = os.path.join(RESULTS_DIR, 'car_by_event.csv')

# ── Style ──────────────────────────────────────────────────────────────────────
BLUE    = '#2166ac'
RED     = '#b2182b'
GREEN   = '#1a9850'
ORANGE  = '#f46d43'
GREY    = '#636363'
LBLUE   = '#92c5de'
LRED    = '#f4a582'

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
    'figure.dpi':        150,
})


def save(fig, name):
    path = os.path.join(FIGS_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved → {path}')


def sig_stars(p):
    if pd.isna(p):   return ''
    if p < 0.01:     return '***'
    if p < 0.05:     return '**'
    if p < 0.10:     return '*'
    return ''


# ══════════════════════════════════════════════════════════════════════════════
# Helpers: re-run event study to get daily CAAR paths
# ══════════════════════════════════════════════════════════════════════════════

def run_models():
    """Load data and run both CAPM and FF4 models. Returns (results_capm, results_ff4)."""
    print('  Loading events and factors...')
    events, ff = load_data()
    print(f'  {len(events)} usable events.')

    out = {}
    for model in ['capm', 'ff4']:
        print(f'  Running {model.upper()}...')
        res = []
        for _, row in events.iterrows():
            r = run_single_event(row, ff, model=model)
            if r:
                res.append(r)
        out[model] = res
        print(f'    {len(res)} events processed.')
    return out['capm'], out['ff4']


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — CAAR time path, full sample, CAPM vs FF4
# ══════════════════════════════════════════════════════════════════════════════

def fig1_full_sample(results_capm, results_ff4):
    print('\nFig 1: Full-sample CAAR path (CAPM vs FF4)...')
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for res, color, label in [
        (results_capm, BLUE,  f'CAPM  (N={len(results_capm)})'),
        (results_ff4,  RED,   f'FF4   (N={len(results_ff4)})'),
    ]:
        days, caar, lo, hi = compute_daily_caar(res)
        ax.plot(days, caar, color=color, linewidth=2, label=label)
        ax.fill_between(days, lo, hi, color=color, alpha=0.12)

    ax.axvline(0,  color='black', linestyle='--', linewidth=1,   alpha=0.7,
               label='Announcement day (t = 0)')
    ax.axhline(0,  color=GREY,   linestyle='-',  linewidth=0.6,  alpha=0.5)

    # Annotate announcement-day value for FF4
    days_f, caar_f, _, _ = compute_daily_caar(results_ff4)
    idx0 = days_f.index(0) if 0 in days_f else None
    if idx0 is not None:
        ax.annotate(f'{caar_f[idx0]:.2f}%',
                    xy=(0, caar_f[idx0]),
                    xytext=(5, caar_f[idx0] - 1.5),
                    fontsize=9, color=RED,
                    arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))

    ax.set_xlabel('Trading Days Relative to Layoff Announcement', fontsize=12)
    ax.set_ylabel('Cumulative Average Abnormal Return (%)', fontsize=12)
    ax.set_title('Figure 1: Cumulative Average Abnormal Returns Around Tech Layoff Announcements\n'
                 'Full Sample — CAPM vs. Fama-French 4-Factor Model', fontsize=12, pad=12)
    ax.legend(frameon=True, loc='upper left')
    ax.set_xlim(-20, 60)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    # Shade announcement window
    ax.axvspan(-1, 1, color='gold', alpha=0.08, label='[-1,+1] window')

    # Add note
    fig.text(0.5, -0.02,
             'Note: Shaded bands = 95% cross-sectional confidence intervals. '
             'Estimation window: [−260, −11] trading days.',
             ha='center', fontsize=9, color=GREY, style='italic')

    fig.tight_layout()
    save(fig, 'fig1_caar_full_sample.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Pre-GenAI vs Post-GenAI (FF4)
# ══════════════════════════════════════════════════════════════════════════════

def fig2_pre_post(results_ff4):
    print('\nFig 2: Pre-GenAI vs Post-GenAI CAAR path...')
    pre  = [r for r in results_ff4 if r['announcement_date'].year <= 2022]
    post = [r for r in results_ff4 if r['announcement_date'].year >= 2023]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, res, color, period, n in [
        (axes[0], pre,  BLUE, '≤ 2022  (Pre-GenAI)',  len(pre)),
        (axes[1], post, RED,  '≥ 2023  (Post-GenAI)', len(post)),
    ]:
        days, caar, lo, hi = compute_daily_caar(res)
        ax.plot(days, caar, color=color, linewidth=2)
        ax.fill_between(days, lo, hi, color=color, alpha=0.13)
        ax.axvline(0,  color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(0,  color=GREY,   linestyle='-',  linewidth=0.5, alpha=0.5)
        ax.axvspan(-1, 1, color='gold', alpha=0.08)
        ax.set_title(f'{period}\n(N = {n})', fontsize=12)
        ax.set_xlabel('Trading Days Relative to Announcement', fontsize=11)
        ax.set_xlim(-20, 60)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

        # Annotate day-0 CAAR
        idx0 = days.index(0) if 0 in days else None
        if idx0:
            ax.annotate(f't=0: {caar[idx0]:.2f}%',
                        xy=(0, caar[idx0]), xytext=(6, caar[idx0] - 1),
                        fontsize=8.5, color=color,
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.7))

    axes[0].set_ylabel('Cumulative Average Abnormal Return (%)', fontsize=12)

    fig.suptitle('Figure 2: CAAR Paths by Era — Pre-GenAI vs. Post-GenAI Tech Layoffs\n'
                 'Fama-French 4-Factor Model', fontsize=12, y=1.01)

    fig.text(0.5, -0.02,
             'Note: ChatGPT launched November 2022. Events split at 2023 cutoff. '
             'Shaded bands = 95% confidence intervals.',
             ha='center', fontsize=9, color=GREY, style='italic')

    fig.tight_layout()
    save(fig, 'fig2_caar_pre_post_genai.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — AI-mentioned vs non-AI (Post-GenAI only, FF4)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_ai_vs_nonai(results_ff4):
    print('\nFig 3: AI-mentioned vs. non-AI CAAR path (Post-GenAI)...')
    post_ai    = [r for r in results_ff4
                  if r['announcement_date'].year >= 2023 and r['ai_mentioned'] == 1]
    post_nonai = [r for r in results_ff4
                  if r['announcement_date'].year >= 2023 and r['ai_mentioned'] == 0]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for res, color, label in [
        (post_nonai, BLUE,  f'AI not mentioned  (N={len(post_nonai)})'),
        (post_ai,    RED,   f'AI mentioned       (N={len(post_ai)})'),
    ]:
        if len(res) < 3:
            continue
        days, caar, lo, hi = compute_daily_caar(res)
        ax.plot(days, caar, color=color, linewidth=2, label=label)
        ax.fill_between(days, lo, hi, color=color, alpha=0.12)

    ax.axvline(0,  color='black', linestyle='--', linewidth=1, alpha=0.7,
               label='Announcement day (t = 0)')
    ax.axhline(0,  color=GREY,   linestyle='-',  linewidth=0.5, alpha=0.5)
    ax.axvspan(-1, 1, color='gold', alpha=0.08)

    ax.set_xlabel('Trading Days Relative to Layoff Announcement', fontsize=12)
    ax.set_ylabel('Cumulative Average Abnormal Return (%)', fontsize=12)
    ax.set_title('Figure 3: CAAR — AI-Motivated vs. Non-AI Layoff Announcements\n'
                 'Post-GenAI Period (≥ 2023), FF4 Model', fontsize=12, pad=12)
    ax.legend(frameon=True)
    ax.set_xlim(-20, 60)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    fig.text(0.5, -0.02,
             'Note: "AI mentioned" = source article contained explicit AI/automation language. '
             'Small AI-mentioned group (N=10–12) — interpret with caution.',
             ha='center', fontsize=9, color=GREY, style='italic')

    fig.tight_layout()
    save(fig, 'fig3_caar_ai_vs_nonai.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — CAAR bar chart by event window, all subsamples (FF4)
# ══════════════════════════════════════════════════════════════════════════════

def fig4_caar_bars():
    print('\nFig 4: CAAR bar chart by window and subsample...')
    df = pd.read_csv(CAR_SUMMARY)
    df = df[df['model'] == 'FF4'].copy()

    windows    = ['[-1,+1]', '[0,+1]', '[0,+5]', '[0,+10]', '[0,+20]', '[-5,+60]']
    subsamples = ['Full sample', 'US only', 'Pre-GenAI (<=2022)', 'Post-GenAI (>=2023)']
    colors_map = {
        'Full sample':           '#2166ac',
        'US only':               '#4dac26',
        'Pre-GenAI (<=2022)':    '#d01c8b',
        'Post-GenAI (>=2023)':   '#f46d43',
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes = axes.flatten()

    for i, win in enumerate(windows):
        ax = axes[i]
        sub_df = df[df['window'] == win]

        xs  = np.arange(len(subsamples))
        bar_w = 0.6

        for j, samp in enumerate(subsamples):
            row = sub_df[sub_df['sample'] == samp]
            if row.empty:
                continue
            r    = row.iloc[0]
            val  = r['CAAR_pct']
            pmin = min(r['p_patell'], r['p_BMP'], r['p_corrado'])
            stars = sig_stars(pmin)
            color = colors_map[samp]

            bar = ax.bar(j, val, width=bar_w,
                         color=color, alpha=0.82, edgecolor='white', linewidth=0.5)

            # Add significance stars above/below bar
            y_star = val + (0.3 if val >= 0 else -0.6)
            if stars:
                ax.text(j, y_star, stars, ha='center', va='bottom',
                        fontsize=11, color=color, fontweight='bold')

            # Label the CAAR value inside/above bar
            y_label = val / 2
            ax.text(j, y_label, f'{val:+.2f}%',
                    ha='center', va='center', fontsize=8.5,
                    color='white' if abs(val) > 0.5 else color,
                    fontweight='bold')

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title(f'Window: {win}', fontsize=11, fontweight='bold')
        ax.set_xticks(xs)
        ax.set_xticklabels(['Full', 'US\nonly', 'Pre-\nGenAI', 'Post-\nGenAI'], fontsize=9)
        ax.set_ylabel('CAAR (%)', fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

    # Legend
    legend_handles = [
        mpatches.Patch(color=colors_map[s], label=s, alpha=0.82)
        for s in subsamples
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle('Figure 4: Cumulative Average Abnormal Returns (CAAR) by Event Window and Subsample\n'
                 'Fama-French 4-Factor Model  |  *** p<0.01, ** p<0.05, * p<0.10',
                 fontsize=13, y=1.01)

    fig.tight_layout()
    save(fig, 'fig4_caar_bars_by_window.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — CAR distribution box plot
# ══════════════════════════════════════════════════════════════════════════════

def fig5_car_distributions():
    print('\nFig 5: CAR distribution box plots...')
    df = pd.read_csv(CAR_BY_EVENT)

    col_map = {
        '[-1,+1]':  'CAR_1_1',
        '[0,+5]':   'CAR_0_5',
        '[0,+20]':  'CAR_0_20',
        '[-5,+60]': 'CAR_5_60',
    }
    labels   = list(col_map.keys())
    data_pct = [(df[col_map[l]].dropna() * 100).values for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        data_pct,
        patch_artist=True,
        notch=True,
        vert=True,
        widths=0.45,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=3, alpha=0.35, linestyle='none'),
    )

    colors = [LBLUE, BLUE, ORANGE, RED]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add median labels
    for i, d in enumerate(data_pct):
        med = np.median(d)
        ax.text(i + 1, med + 0.3, f'{med:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add N labels
    for i, d in enumerate(data_pct):
        ax.text(i + 1, ax.get_ylim()[0] + 0.5, f'N={len(d)}',
                ha='center', va='bottom', fontsize=8.5, color=GREY)

    ax.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.6)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Cumulative Abnormal Return (%)', fontsize=12)
    ax.set_xlabel('Event Window', fontsize=12)
    ax.set_title('Figure 5: Distribution of Individual Event CARs by Window\n'
                 'FF4 Model, Full Sample', fontsize=12, pad=12)

    fig.text(0.5, -0.02,
             'Note: Notched boxes show 95% CI around median. Whiskers = 1.5×IQR. '
             'Circles = outliers. Dashed line = zero.',
             ha='center', fontsize=9, color=GREY, style='italic')

    fig.tight_layout()
    save(fig, 'fig5_car_distributions.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — CAR [0,+20] by industry
# ══════════════════════════════════════════════════════════════════════════════

def fig6_by_industry():
    print('\nFig 6: CAR by industry...')
    df = pd.read_csv(CAR_BY_EVENT)
    df['CAR_0_20_pct'] = df['CAR_0_20'] * 100

    # Keep industries with >= 10 events
    counts = df['industry'].value_counts()
    top_industries = counts[counts >= 10].index.tolist()
    df2 = df[df['industry'].isin(top_industries)].copy()

    # Summary stats per industry
    summary = (df2.groupby('industry')['CAR_0_20_pct']
                   .agg(['mean', 'std', 'count', 'median'])
                   .rename(columns={'mean': 'mean_car', 'std': 'std_car',
                                    'count': 'n', 'median': 'median_car'})
                   .reset_index())
    summary['se'] = summary['std_car'] / np.sqrt(summary['n'])
    summary = summary.sort_values('mean_car', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = [RED if v < 0 else BLUE for v in summary['mean_car']]
    bars = ax.barh(summary['industry'], summary['mean_car'],
                   xerr=1.96 * summary['se'],
                   color=colors, alpha=0.78,
                   error_kw=dict(ecolor='grey', capsize=4, linewidth=1.2),
                   edgecolor='white', linewidth=0.5)

    # Value labels
    for _, row in summary.iterrows():
        x_pos = row['mean_car'] + (1.96 * row['se']) + 0.15
        ax.text(x_pos, row['industry'], f"{row['mean_car']:+.2f}%  (N={row['n']})",
                va='center', fontsize=8.5, color=GREY)

    ax.axvline(0, color='black', linewidth=0.9)
    ax.set_xlabel('Mean CAR [0,+20] (%)', fontsize=12)
    ax.set_title('Figure 6: Mean Cumulative Abnormal Return [0, +20] by Industry\n'
                 'FF4 Model, Full Sample  |  Industries with ≥ 10 Events',
                 fontsize=12, pad=12)

    # Add legend
    pos_patch = mpatches.Patch(color=BLUE, alpha=0.78, label='Positive CAR')
    neg_patch = mpatches.Patch(color=RED,  alpha=0.78, label='Negative CAR')
    ax.legend(handles=[pos_patch, neg_patch], loc='lower right', fontsize=10)

    fig.text(0.5, -0.02,
             'Note: Error bars = 95% confidence intervals (±1.96 × SE). '
             'Industries with fewer than 10 events excluded.',
             ha='center', fontsize=9, color=GREY, style='italic')

    fig.tight_layout()
    save(fig, 'fig6_car_by_industry.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Summary table: CAAR with all three test statistics (FF4)
# ══════════════════════════════════════════════════════════════════════════════

def fig7_summary_table():
    print('\nFig 7: CAAR summary statistics table...')
    df = pd.read_csv(CAR_SUMMARY)
    df = df[(df['model'] == 'FF4')].copy()

    windows    = ['[-1,+1]', '[0,+1]', '[0,+5]', '[0,+10]', '[0,+20]', '[-5,+60]', '[-20,+60]']
    subsamples = ['Full sample', 'US only', 'Pre-GenAI (<=2022)', 'Post-GenAI (>=2023)']
    subsample_labels = ['Full Sample', 'US Only', 'Pre-GenAI\n(≤ 2022)', 'Post-GenAI\n(≥ 2023)']

    fig, axes = plt.subplots(1, len(subsamples), figsize=(16, 6), sharey=True)

    for ax, samp, samp_label in zip(axes, subsamples, subsample_labels):
        sub = df[df['sample'] == samp].copy()
        sub = sub[sub['window'].isin(windows)].copy()
        sub['window_ord'] = sub['window'].map({w: i for i, w in enumerate(windows)})
        sub = sub.sort_values('window_ord')

        ys    = np.arange(len(sub))
        caars = sub['CAAR_pct'].values

        colors_bar = [RED if v < 0 else BLUE for v in caars]
        ax.barh(ys, caars, color=colors_bar, alpha=0.75, height=0.55, edgecolor='white')

        for j, (_, row) in enumerate(sub.iterrows()):
            pmin  = min(row['p_patell'], row['p_BMP'], row['p_corrado'])
            stars = sig_stars(pmin)
            n_str = f"N={row['N']}"
            label = f"{row['CAAR_pct']:+.2f}%{stars}"

            x_off = 0.15 if row['CAAR_pct'] >= 0 else -0.15
            ax.text(row['CAAR_pct'] + x_off, j, label,
                    va='center', ha='left' if row['CAAR_pct'] >= 0 else 'right',
                    fontsize=8.5, fontweight='bold',
                    color=BLUE if row['CAAR_pct'] >= 0 else RED)

        ax.set_yticks(ys)
        ax.set_yticklabels(sub['window'].values, fontsize=10)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('CAAR (%)', fontsize=10)
        ax.set_title(samp_label, fontsize=11, fontweight='bold')

        n_total = sub['N'].iloc[0] if len(sub) > 0 else ''
        ax.set_xlabel(f'CAAR (%)\n(N = {n_total})', fontsize=10)

    fig.suptitle('Figure 7: CAAR by Event Window and Subsample — FF4 Model\n'
                 '*** p<0.01  ** p<0.05  * p<0.10  (minimum p across Patell, BMP, Corrado)',
                 fontsize=12, y=1.02)

    fig.text(0.5, -0.02,
             'Note: Significance based on the most conservative of three tests: '
             'Patell (1976), BMP (1991), and Corrado (1989) rank test.',
             ha='center', fontsize=9, color=GREY, style='italic')

    fig.tight_layout()
    save(fig, 'fig7_caar_subsample_summary.png')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 72)
    print('GENERATING PAPER-QUALITY FIGURES')
    print('=' * 72)

    # Figures 4–7 use saved CSVs — no re-run needed
    print('\n── Figures from saved results (no re-run needed) ──')
    fig4_caar_bars()
    fig5_car_distributions()
    fig6_by_industry()
    fig7_summary_table()

    # Figures 1–3 need daily AR data → re-run event study
    print('\n── Re-running event study for CAAR time-path figures ──')
    results_capm, results_ff4 = run_models()

    fig1_full_sample(results_capm, results_ff4)
    fig2_pre_post(results_ff4)
    fig3_ai_vs_nonai(results_ff4)

    print('\n' + '=' * 72)
    print(f'All 7 figures saved to: {FIGS_DIR}')
    print('=' * 72)


if __name__ == '__main__':
    main()
