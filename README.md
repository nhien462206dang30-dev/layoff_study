# Tech Layoff Announcement Study: Stock Price Reactions & AI Context

Event study and cross-sectional regression examining how markets react to tech layoff announcements, with a focus on whether AI/automation context changes that reaction — particularly after the ChatGPT launch (Nov 2022).

---

## Research Question

**Does the market react differently to layoff announcements that cite AI/automation as a reason, compared to purely financial/restructuring layoffs?**

Secondary question: Did the ChatGPT launch change how investors interpret AI-driven layoffs?

---

## Key Findings

### Event Study (All 444–467 events)

| Window | CAAR | Patell t | Sig |
|--------|------|----------|-----|
| [−1, +1] | −0.49% | −3.03 | *** |
| [0, +5]  | −0.35% | −2.06 | **  |
| [0, +20] | +2.25% | +2.42 | **  |
| [−5, +60]| +6.40% | +4.31 | *** |

Markets react negatively in the short window (−0.49%) but recover and outperform over the following months (+2.25% by day 20, +6.40% by day 60).

### Cross-Sectional Regression

Observable event characteristics (AI mention, size, percentage cut, era) do **not** significantly predict short-window abnormal returns. R² ≈ 1–2% across all specifications. Consistent with market efficiency.

### DID: Did ChatGPT Change AI-Layoff Reactions?

`CAR = α + β₁·ai + β₂·post_chatgpt + β₃·(ai×post_chatgpt) + controls`

β₃ (the DID estimator) is consistently positive (AI layoffs slightly outperform non-AI post-ChatGPT), but not statistically significant at conventional levels with reliable sample sizes.

---

## AI-Mention Definition

The key methodological challenge is identifying whether a layoff is "AI-related" from news text. Three approaches were compared:

| Definition | N (ai=1) | Rate | Assessment |
|------------|----------|------|------------|
| Desktop broad (keyword + efficiency + substring AI) | 234 | 50.1% | ⚠️ ~30–40% false positives |
| Claude strict (explicit causal regex) | 12 | 2.6% | ❌ Misses most real cases |
| **Tiered ai_broad (T1+T2+T3, whole-word) ★** | **74** | **15.8%** | **✓ Recommended** |
| Tiered ai_primary (T1+T2) | 22 | 4.7% | ✓ Robustness check |

The recommended variable (`ai_broad`) measures: *"did the news coverage of this layoff explicitly mention a specific AI technology"* — not whether AI was the true cause. Full comparison: [`docs/ai_definition_comparison.md`](docs/ai_definition_comparison.md).

**Paywall limitation:** 42% of source articles (197/467) are behind paywalls (Bloomberg, WSJ, CNBC, Reuters) and return empty text → systematic undercounting. Estimated true rate ~27% if paywalled articles follow the same distribution.

---

## Data Pipeline

```
scrapers/          → Raw layoffs.fyi event data (Airtable scrape)
analysis/
  get_data.py      → Fetch stock returns (yfinance) + FF4 factors
  enrich_events.py → Match tickers, classify initial ai_mentioned
  relabel_tiered.py→ Three-tier AI-mention classification (recommended)
  relabel_ai.py    → Claude API-based AI labeling (experimental)
  event_study.py   → CAR computation + Patell/BMP/Corrado tests
  improved_analysis.py → Full pipeline: DID + grouped CAAR event study
  cross_section.py → Cross-sectional OLS regression (HC3 SEs)
  visualize.py     → CAAR plots

data/
  raw/             → Scraped layoffs.fyi (not tracked in git)
  processed/
    master_events_final.csv  → 467 events with tickers + labels
    ticker_mapping.csv       → Company → ticker mapping
  results/
    car_by_event.csv         → Event-level CARs (original)
    improved/
      ai_labels_tiered.csv   → Three-tier AI labels for all 467 events
      final_labels_and_cars.csv → Labels merged with CARs (444 events)
      car_by_event_v2.csv    → Recomputed CARs with updated labels
      did_results.csv        → DID regression output

docs/
  ai_definition_comparison.md → Full methodology comparison across definitions
```

---

## Methodology

- **Model:** Fama-French 4-factor (MKT_RF, SMB, HML, MOM)
- **Estimation window:** [−260, −11] (minimum 60 trading days)
- **Event windows:** [−1,+1], [0,+5], [0,+20], [−5,+60]
- **Test statistics:** Patell (1976), BMP (1991), Corrado (1989) rank test
- **Regression SEs:** HC3 heteroskedasticity-robust OLS
- **Sample:** 444–467 tech company layoff announcements, 2020–2024
- **DID breakpoint:** 2022-11-30 (ChatGPT launch)

---

## Setup

```bash
pip install -r requirements.txt

# Run full pipeline
python analysis/get_data.py          # fetch stock data
python analysis/relabel_tiered.py    # classify AI mentions
python analysis/improved_analysis.py # run regressions + event study
```

---

## Comparison with Desktop Version

A parallel analysis ("裁员影响模型") used manual data cleaning and a broader keyword definition. Key differences:

- Desktop CAR[−1,+1]: **+1.00%** (t=6.33***) vs Claude: **−0.49%** (t=−3.03***)
- Opposite sign likely due to sample composition: desktop manually filtered to post-IPO companies with solid trading history; Claude sample includes more early-stage and international listings.
- Desktop ai_mentioned rate: 50.1% (broad substring match including `efficiency`) vs Claude 15.8% (whole-word, no generic terms).

See [`docs/ai_definition_comparison.md`](docs/ai_definition_comparison.md) for full details.
