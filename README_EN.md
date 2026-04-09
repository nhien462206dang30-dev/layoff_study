# Layoff Announcements and Stock Price Reactions: Market Efficiency in the Age of AI

**🌐 Language:** 　[中文](README.md)　|　**English (current)**

---

> **Study Type:** Empirical Finance — Event Study
> **Sample Period:** March 2020 – March 2024, publicly listed firms in tech and adjacent industries
> **Core Methods:** Fama-French Four-Factor Model (FF4) + Difference-in-Differences (DID)
> **Primary Sample:** U.S.-listed firms, N = 429 events; Mature Firm Sample, N = 263 events

---

## I. Motivation and Research Questions

### 1.1 Why This Question

In recent years, tech companies have gone through a significant wave of layoffs. A common narrative in the media and among investors is that these firms are actively replacing human labor with AI — and that markets view this favorably, interpreting AI-driven workforce reductions as a sign of improved operational efficiency. Under this view, layoff announcements should be met with positive stock price reactions.

At the same time, there is a competing explanation that is harder to dismiss: many of these companies simply overhired during the pandemic boom years, and the subsequent layoffs were nothing more than a correction of that excess. In this reading, the AI framing is largely rhetorical — a way to make an awkward operational retreat sound forward-looking rather than reactive. If so, one would expect the market response to be more muted or even negative, since the announcement signals prior mismanagement rather than strategic transformation.

Both explanations are plausible, and anecdotal evidence exists for each. But anecdotes are a poor basis for understanding what is actually driving market behavior. The natural first step — before drawing any conclusions about whether AI is changing how companies are valued, or whether the efficiency narrative holds up — is to examine how the stock market actually responds to layoff announcements, across a large and systematically constructed sample. That is the core motivation of this study.

### 1.2 Four Research Questions

| # | Research Question | Analysis Module |
|---|---|---|
| **Q1** | How can layoff events from multiple heterogeneous sources be systematically compiled, deduplicated, and standardized into a usable database? | `scrapers/` + `analysis/01–02` |
| **Q2** | How are stock price and factor data obtained, and how are risk-adjusted cumulative abnormal returns (CARs) estimated under the FF4 framework? | `analysis/04_event_study_ff4.py` |
| **Q3** | What are the short-, medium-, and long-run price effects of layoff announcements? Does the market response vary meaningfully across firm types? | `analysis/04` + `analysis/08` |
| **Q4** | After ChatGPT's launch, did AI-linked layoffs receive a measurably different market reaction? And does that difference hold up to scrutiny? | `analysis/05–07` |

---

## II. Project Structure

```
layoff_study/
│
├── README.md                              ← Chinese version of this document
├── README_EN.md                           ← This file: full English research overview
│
├── scrapers/                              ← Raw data collection scripts
│   ├── 01_scrape_layoffs_fyi.py           ← layoffs.fyi scraper (Playwright, with pagination)
│   ├── 02_scrape_edgar_8k.py             ← EDGAR 8-K full-text search (workforce reduction/restructuring)
│   ├── 03_scrape_techcrunch.py           ← TechCrunch article body scraper (AI labeling text source)
│   ├── 04_scrape_trueup.py               ← TrueUp supplementary data source
│   └── 05_combine_sources.py             ← Multi-source merge: deduplication, date normalization, ticker pre-matching
│
├── analysis/                              ← Main analysis pipeline, numbered in execution order
│   ├── 01_collect_data.py                 ← Step 1: Download stock prices (Yahoo Finance) and FF4 factors (Ken French)
│   ├── 02_enrich_events.py                ← Step 2: Enrich events with industry, region, initial AI labels
│   ├── 03_relabel_ai_tiered.py            ← Step 3: Tiered AI label refinement (three-level classification)
│   ├── 04_event_study_ff4.py              ← Step 4: Main event study (FF4 + CAPM, all subsamples) ★
│   ├── 05_did_regression.py               ← Step 5: DID regression + cross-sectional OLS (with new controls) ★
│   ├── 06_robustness_checks.py            ← Step 6: Robustness check suite (six tests)
│   ├── 07_calendar_time_portfolio.py      ← Step 7: Calendar-time portfolio method (clustering correction)
│   ├── 08_size_sector_analysis.py         ← Step 8: Size × sector heterogeneity (R² grouping)
│   ├── 09_export_results.py               ← Step 9: Compile and export FINAL_RESULTS.xlsx
│   │
│   └── others/                            ← Deprecated scripts, kept for reference only
│       ├── relabel_ai.py                  ← Old AI labeler (broad string match, superseded by 03)
│       ├── cross_section.py               ← Old cross-section (merged into 05)
│       ├── pre_announcement.py            ← Pre-announcement drift (merged into 06)
│       ├── repeat_events.py               ← Repeat events analysis (merged into 06)
│       ├── diagnose_jump.py               ← Debugging: single-event price jump diagnostics
│       └── visualize.py                   ← Old visualization (charts now generated within each main script)
│
├── data/
│   ├── raw/                               ← Raw scraped data (read-only; do not modify)
│   │   ├── layoffs_fyi_raw.csv            ← layoffs.fyi raw scrape (776 records, including duplicates)
│   │   └── edgar_8k_raw.csv               ← EDGAR 8-K keyword match records
│   │
│   ├── processed/                         ← Cleaned, labeled, analysis-ready data
│   │   ├── master_events_final.csv        ← Master event table: 481 usable events ★
│   │   ├── ff_factors.csv                 ← FF4 daily factors (MKT_RF / SMB / HML / MOM / RF)
│   │   ├── stock_returns/                 ← Daily returns per ticker (e.g., AAPL.csv)
│   │   ├── condition_a_tickers.csv        ← Mature Firm Sample: 130 verified tickers
│   │   ├── prior_6m_return.csv            ← Control variable: cumulative 6-month pre-announcement return
│   │   ├── funds_raised.csv               ← Control variable: total capital raised (USD, with log transform)
│   │   └── others/                        ← Intermediate files not used in main pipeline
│   │       ├── master_events.csv          ← Post-dedup, pre-enrichment version
│   │       ├── master_events_enriched.csv ← Post-enrichment, pre-AI-labeling version
│   │       ├── ai_label_audit.csv         ← Manual audit log for AI labels
│   │       ├── failed_tickers.csv         ← Tickers for which no stock data could be retrieved
│   │       └── [phase deliverables]       ← Archived milestone Excel files
│   │
│   └── results/                           ← All analysis outputs, organized by module
│       ├── car_by_event.csv               ← Per-event FF4 CARs (467 events, with Beta / R² / α) ★
│       ├── car_summary.csv                ← CAAR summary: all subsamples × windows × models ★
│       ├── FINAL_RESULTS.xlsx             ← Final consolidated Excel (multiple sheets) ★
│       │
│       ├── figures/
│       │   ├── event_study/               ← Step 4: CAAR cumulative path plots (4 figures)
│       │   ├── did/                       ← Step 5: DID group comparison plots (3 figures)
│       │   └── others/                    ← Archived earlier-version figures
│       │
│       ├── did_crosssection/              ← All Step 5 outputs
│       │   ├── car_by_event_v2.csv        ← CARs recomputed with refined AI labels
│       │   ├── ar_panel_daily.csv         ← Daily AR panel (22,629 rows)
│       │   ├── did_results_us_primary.csv ← DID main specification (U.S., N = 428)
│       │   ├── did_results_core_tech.csv  ← DID core tech subsample
│       │   ├── cross_section_v2.csv       ← Cross-sectional OLS, four specifications
│       │   ├── final_labels_and_cars.csv  ← Master table: events + refined AI labels + CARs
│       │   └── ai_labels_tiered.csv       ← Tiered AI label details (with raw text match records)
│       │
│       ├── robustness/                    ← Step 6: Six robustness checks
│       │   ├── placebo_did_results.csv    ← Placebo DID (six false breakpoints)
│       │   ├── parallel_trends_monthly.csv ← Monthly parallel trends validation
│       │   ├── paywall_sensitivity.csv    ← Paywall sensitivity (50 Monte Carlo draws)
│       │   ├── repeat_events_summary.csv  ← First vs. subsequent layoff events
│       │   ├── pre_announcement_stats.csv ← Pre-announcement drift test [-20, -1]
│       │   └── fig_*.png
│       │
│       ├── calendar_time/                 ← Step 7: Calendar-time portfolio method
│       │   ├── ct_results.csv
│       │   └── fig_ct_portfolio.png
│       │
│       └── size_sector/                   ← Step 8: Size × sector 2×3 analysis
│           ├── size_sector_caar.csv
│           ├── fig_size_sector_2x2.png
│           └── fig_size_sector_box.png
│
├── docs/
│   ├── theory_section.md                  ← Literature review and theoretical framework draft
│   └── ai_definition_comparison.md        ← Comparison of three AI labeling approaches
│
└── condition_a_curated_sample/            ← Source archive for the Mature Firm Sample (N = 152 manual review)
    ├── 01_raw_data/
    ├── 02_intermediate_processing/
    ├── 03_ticker_matching/
    │   └── 05_manual_corrections_final.csv  ← Source file for the funds_raised control variable ★
    ├── 04_stock_price_factors/            ← Stock prices and FF3 factors (legacy; main pipeline uses FF4)
    ├── 05_event_study/                    ← Legacy event study (FF3)
    └── 06_regression_results/
```

---

## III. Data Construction

### 3.1 Multi-Source Integration and Deduplication (Q1)

Assembling a clean layoff event database is harder than it sounds. No single source manages to be comprehensive, date-accurate, and ticker-matched all at once, so the design here combines three sources with complementary strengths and blind spots.

**layoffs.fyi** is the primary source — a community-maintained platform that tracks global tech layoffs in near-real time, with data embedded in an Airtable interface that has no public API. Scraping was handled through a Playwright automation script that navigates the embedded table with pagination logic; the raw pull yielded 776 records. The main advantages are breadth (small companies and international firms alike get covered) and timeliness. The main limitation is that announcement dates sometimes lag by a few days, since the gap between internal employee notification and media pickup is variable and often non-trivial.

**EDGAR 8-K filings** provide a structured, legally-reliable complement. U.S. public companies are required to file Form 8-K for material events within four business days, so the date precision here is considerably higher than media-based sources. Full-text search for terms like "workforce reduction," "restructuring," and "headcount reduction" identified relevant filings. The obvious limitation is coverage: only U.S.-listed companies, and only for reductions large enough to constitute a material event.

**TechCrunch and other news articles** serve a specific function in this pipeline: they are the text source for AI labeling. Determining whether a layoff announcement was framed as AI-related requires reading the editorial narrative around the event, not just the company's formal disclosure. News articles are better suited to this than 8-K filings or platform data.

The deduplication rule: records from the same company within a seven-day window are merged into a single event, retaining the earliest date. After deduplication, ticker matching, and quality filtering (`event_study_usable == True`, requiring at least 100 valid trading days in the estimation window), **the final working dataset contains 481 events, of which 467 have complete stock price data**.

A brief snapshot of the sample composition:
- Geography: U.S.-listed, 551 events (77%); international, 130 events (23%)
- Industry (30 categories): top six are Healthcare (85), Transportation (78), Fintech (70), Consumer (61), Education (52), AI (48)
- Time span: March 2020 through March 2024
- Firms with multiple layoff records: 114, averaging 3.2 events each

**Constructing the Mature Firm Sample:**

Alongside the automated pipeline, a second, more carefully curated subsample was assembled independently. The **Mature Firm Sample** is defined as companies that (i) are listed on a major exchange (NASDAQ or NYSE), (ii) have a complete, uninterrupted trading history post-IPO, and (iii) are not trading on OTC or pink sheets, and show no signs of financial distress (e.g., extended periods of sub-$1 stock prices, active delisting proceedings). Ticker assignments were cross-verified using both Yahoo Finance and OpenFIGI.

Starting from 152 candidate companies identified through manual review of the original data, 22 were excluded — six because they were international listings for which U.S. FF4 factors would be misspecified, and sixteen because they were OTC or distressed firms whose stock behavior would distort the results. The remaining 130 companies — spanning established names like AAPL and AMZN as well as growth-stage but post-IPO companies like COIN and ABNB — generated 263 layoff events that enter the FF4 event study.

The Mature Firm Sample and the full pipeline sample are designed to be complementary, not competing. The former represents "companies the market can price efficiently under normal conditions"; the latter captures the broader cross-section of what this layoff wave actually looked like. Differences in their results are themselves informative, not a sign of inconsistency.

### 3.2 Stock Price Data and the Factor Model (Q2)

**Stock price data** were downloaded via Yahoo Finance's `yfinance` interface, pulling daily adjusted closing prices for each event ticker and saving them as individual CSV files under `data/processed/stock_returns/` (e.g., `AAPL.csv`). Adjusted closing prices account for dividends and splits and feed directly into log-return calculations.

**Fama-French four-factor data** were downloaded from the Kenneth R. French Data Library, covering daily factor realizations from 2018 through 2026:

| Factor | Description |
|---|---|
| MKT_RF | Excess market return (market portfolio minus risk-free rate) |
| SMB | Small-minus-big (size premium) |
| HML | High-minus-low (value premium, book-to-market ratio) |
| MOM | Momentum factor (Carhart 1997: prior 12-month winners minus losers) |

The choice of FF4 over FF3 deserves a brief note, because it actually matters for this particular sample. The 2022–2023 period was characterized by a dramatic momentum crash — high-valuation growth stocks that had soared in 2021 collapsed in 2022, then largely recovered through 2023. Without controlling for momentum, a meaningful chunk of those price swings would get incorrectly attributed to layoff announcement effects, biasing CAR estimates in a systematic and directional way.

---

## IV. Methodology

### 4.1 The FF4 Event Study Framework

For each event $i$, we estimate the following FF4 model over an estimation window $t \in [-260, -11]$ relative to the announcement date $t = 0$, requiring a minimum of 100 valid trading days:

$$R_{i,t} - RF_t = \alpha_i + \beta_{1i} \cdot MKT\_RF_t + \beta_{2i} \cdot SMB_t + \beta_{3i} \cdot HML_t + \beta_{4i} \cdot MOM_t + \varepsilon_{i,t}$$

Using the estimated parameters, the daily **abnormal return (AR)** during the event window is defined as the residual between the actual return and the FF4 model's predicted return:

$$AR_{i,t} = (R_{i,t} - RF_t) - \hat{\alpha}_i - \hat{\beta}_{1i} \cdot MKT\_RF_t - \hat{\beta}_{2i} \cdot SMB_t - \hat{\beta}_{3i} \cdot HML_t - \hat{\beta}_{4i} \cdot MOM_t$$

Cumulating over window $[t_1, t_2]$ gives the **cumulative abnormal return (CAR)**:

$$CAR_i[t_1, t_2] = \sum_{t=t_1}^{t_2} AR_{i,t}$$

Averaging across $N$ events gives the **cumulative average abnormal return (CAAR)**:

$$CAAR[t_1, t_2] = \frac{1}{N} \sum_{i=1}^{N} CAR_i[t_1, t_2]$$

Events with single-day $|AR| > 50\%$ are treated as anomalous (typically OTC penny stocks or near-delistings) and excluded.

### 4.2 Event Window Design

Seven event windows are examined. Each was chosen for a specific economic reason:

| Window | Economic Meaning | Design Logic |
|---|---|---|
| **[-1, +1]** | 3-day announcement window | **Primary reporting window.** Includes the day before (leak test) and day after (overnight reaction). The cleanest estimate of pure announcement effect. |
| **[0, +1]** | 2-day window | Announcement day plus the following day; facilitates comparison with standard 2-day windows in prior literature |
| **[0, +5]** | One week post-announcement | Full absorption of initial market reaction, including analyst report releases |
| **[0, +10]** | Two weeks post-announcement | Typical institutional portfolio adjustment horizon |
| **[0, +20]** | One month post-announcement | Medium-term drift: captures incremental information as implementation progress becomes visible |
| **[-5, +60]** | Primary long-run window | **Primary long-run reporting window.** Five days of pre-announcement price formation through 60 days post-announcement (~3 months), capturing the full price discovery arc |
| **[-20, +60]** | Extended long-run window | Tests for a longer pre-announcement lead-up (~one month of anticipation) |

The [-1,+1] window is the focal short-run window because it is theoretically the closest to a "pure announcement effect" and least susceptible to macro noise. The [-5,+60] window is used to evaluate the market's full revaluation of the layoff decision, though — as discussed below — its interpretation is complicated by concurrent market-wide movements.

### 4.3 Statistical Tests

All windows are assessed using three test statistics simultaneously. The most conservative result (highest p-value) is used for inference.

**Patell (1976) standardized residual test:** Each event's CAR is divided by its estimation-window residual standard deviation, producing a standardized CAR (SCAR). These are then aggregated into a cross-sectional z-statistic. This test assumes event independence and homoskedasticity, and has good power with large samples, but it does not account for event-induced variance changes.

**BMP test (Boehmer, Musumeci & Poulsen, 1991):** Extends Patell by explicitly estimating the cross-sectional dispersion of SCARs, accommodating the possibility that the announcement itself triggers changes in return volatility. This is the most robust of the three parametric tests and serves as the primary parametric inference benchmark throughout this study.

**Corrado (1989) rank test:** A nonparametric test that makes no distributional assumptions, instead comparing the rank of event-window ARs against estimation-window ARs. Given that tech stock return distributions tend to be leptokurtic, having a distribution-free check is especially valuable here.

### 4.4 Sample Groupings

Nine subsamples are defined, forming a matrix of analyses from the primary specification through various cross-sectional cuts:

| Subsample | Definition | Purpose |
|---|---|---|
| **U.S. primary (main)** | listing_region == 'US' | Main identification sample: FF4 factor pricing is valid |
| **Mature Firm Sample** | 130 manually verified tickers, U.S. only | High-quality firm benchmark: excludes distressed / OTC effects |
| **Full sample** | All events (including international) | Appendix robustness check |
| **Core tech, U.S.** | Hardware / Software / AI / Data and 10 other categories, U.S. | Do tech firms react more strongly to layoffs? |
| **Non-tech, U.S.** | All other industries, U.S. | Is the negative signal effect unique to tech? |
| **Pre-ChatGPT (≤2022)** | Announcement date ≤ end of 2022 | DID control period |
| **Post-ChatGPT (≥2023)** | Announcement date ≥ 2023 | DID treatment period |
| **Post-ChatGPT, U.S.** | ≥2023 & U.S. | Main DID treatment group (primary spec) |
| **Pre-ChatGPT, U.S.** | ≤2022 & U.S. | Main DID control group (primary spec) |

---

## V. Empirical Results

### 5.1 Main Event Study (Q3)

#### 5.1.1 The Short-Negative, Long-Positive Pattern

Table 1 reports the full CAAR results for the FF4 model under the U.S. primary specification (N = 429). Statistical significance is based on the most conservative of the three tests; the BMP t-statistic serves as the primary parametric inference measure.

**Table 1: FF4 Event Study Results (U.S. Primary Sample, N = 429)**

| Window | N | CAAR | Patell Z | BMP t | Corrado | Significance |
|---|---|---|---|---|---|---|
| [-1, +1] | 429 | **−0.961%** | −5.280*** | −2.520** | −3.297*** | *** |
| [0, +1] | 429 | **−0.700%** | −4.986*** | −2.128** | −2.736*** | *** |
| [0, +5] | 429 | **−0.658%** | −3.267*** | −1.966** | −2.730*** | *** |
| [0, +10] | 429 | +0.093% | −1.365 | −0.894 | −2.213** | — |
| [0, +20] | 429 | +1.300% | +1.107 | +0.789 | −0.863 | — |
| [-5, +60] | 429 | **+5.097%** | +3.626*** | +2.558** | +0.313 | *** |
| [-20, +60] | 429 | **+4.668%** | +3.064*** | +2.240** | +0.356 | *** |

*Note: *** p < 0.01, ** p < 0.05, * p < 0.10. Significance reflects the most conservative of the three tests.*

The results display a clear short-negative, long-positive temporal structure — worth walking through window by window.

**Announcement period [-1, +1]:** CAAR is −0.961%, with all three test statistics significant. The Patell Z reaches −5.28. The market's first reaction is negative — layoff announcements are being read as operational distress signals, at least initially. The slight negative pressure on day -1 is consistent with partial information leakage, but the CAAR over [-20, -1] is only −0.82% (p = 0.287, not significant), ruling out systematic pre-announcement drift and confirming that the study design is clean.

**Short-run absorption [0, +5]:** The CAAR remains negative at −0.658% (BMP p = 0.050) through the first week after the announcement. There is no immediate reversal. This pattern is consistent with a brief period of information reinforcement — analyst downgrades, continued media coverage — that deepens the initial negative reaction before it stabilizes.

**Medium-run transition [0, +10] to [0, +20]:** The CAAR turns positive, but neither window reaches statistical significance (BMP p of 0.372 and 0.431, respectively). This is effectively the re-evaluation phase: the market is weighing the cost savings from headcount reduction against concerns about slower revenue growth, and the net effect is indistinguishable from zero.

**Long-run reversal [-5, +60]:** CAAR rises to +5.097%, with both Patell (+3.63***) and BMP (+2.56**) significant. The economic interpretation of this long-run drift is contested. One view is that the market is belatedly pricing in improved profitability as the restructuring takes effect. A competing — and arguably more credible — view is that the 2023 tech sector recovery, driven by deleveraging and multiple expansion, lifted nearly every tech stock in this period regardless of whether the company had announced layoffs. The calendar-time portfolio analysis (Step 7) provides evidence for the second interpretation: once time fixed effects are accounted for, the long-run alpha disappears.

#### 5.1.2 Mature Firm Sample vs. Full Sample: How Firm Quality Moderates Market Reactions

**Table 2: Mature Firm Sample vs. U.S. Full Sample (FF4, [-1, +1] window)**

| Sample | N | CAAR | Patell Z | BMP t | BMP p-value |
|---|---|---|---|---|---|
| U.S. full sample (primary) | 429 | −0.961% | −5.280*** | −2.520 | 0.012 |
| Mature Firm Sample | 263 | −0.451% | −2.392** | −1.084 | 0.280 |

The Mature Firm Sample — 130 manually verified, exchange-listed companies generating 263 events — shows a three-day CAAR about 53% smaller in absolute terms (−0.451% vs. −0.961%). More importantly, the BMP test is not statistically significant at any conventional level (p = 0.280).

Several mechanisms likely contribute to this gap. Mature firms tend to have well-developed investor relations functions and higher institutional ownership, meaning professional investors are better equipped to evaluate the strategic rationale behind a layoff rather than treating it reflexively as bad news. The full pipeline sample includes a nontrivial share of smaller, lower-liquidity companies whose layoffs often coincide with genuine financial stress, making their stock prices naturally more sensitive to adverse signals. And the Mature Firm Sample implicitly controls for a form of survivorship bias — companies that maintain continuous exchange listings and complete data records are, on average, in better financial shape than those that do not.

The broader implication is worth stating plainly: **the sign and magnitude of the market response to a layoff announcement is not uniform across firms — it depends heavily on how the market perceives the company's underlying quality.** Generalizing from "layoffs cause stock prices to fall" as an unconditional rule misses substantial within-sample heterogeneity.

#### 5.1.3 Industry Split: Does Tech React Differently?

**Table 3: Industry Group CAAR Comparison (FF4, U.S.)**

| Industry Group | N | [-1,+1] CAAR | Patell | BMP | [-5,+60] CAAR | BMP |
|---|---|---|---|---|---|---|
| Core tech (13 categories) | 182 | −1.015%*** | −3.636*** | −1.829* | +5.308%*** | +1.731* |
| Non-tech | 247 | −0.921%*** | −3.837*** | −1.765* | +4.941%** | +1.919* |

The intuitive expectation here might be that tech firms, whose valuations are more sensitive to growth expectations, would show stronger negative reactions to layoff news — a signal that growth is slowing. But the data tell a different story: core tech and non-tech CAARs at [-1,+1] are nearly identical (−1.015% vs. −0.921%), and the difference is not statistically significant. The negative signaling effect of layoff announcements during this period was not specific to tech — across industries, the market's initial read was similarly cautious.

#### 5.1.4 Pre- vs. Post-ChatGPT Period Comparison

**Table 4: Pre- vs. Post-ChatGPT CAAR (FF4, U.S.)**

| Period | N | [-1,+1] | BMP p | [0,+5] | BMP p | [-5,+60] | BMP p |
|---|---|---|---|---|---|---|---|
| Pre-ChatGPT (≤2022) | 119 | −0.878% | 0.320 | −1.136% | 0.359 | **+9.779%** | 0.004*** |
| Post-ChatGPT (≥2023) | 310 | **−0.992%** | 0.016** | **−0.475%** | 0.081* | +3.300% | 0.271 |

Two patterns here stand out.

**Short-run intensification:** After ChatGPT, the [-1,+1] CAAR shifts from a statistically insignificant −0.878% to a significant −0.992% (BMP p = 0.016). The short-run reaction becomes both larger and more reliably negative in the post period. One plausible explanation is that heightened media attention to tech layoffs post-ChatGPT amplifies the information environment around these announcements, sharpening the immediate price response.

**Long-run drift disappears:** The pre-ChatGPT [-5,+60] CAAR is a remarkably large +9.779% (BMP p = 0.004***), while the post-ChatGPT figure drops to an insignificant +3.300%. The most credible explanation for this contrast is time confounding rather than a genuine change in how markets value layoffs: the pre-ChatGPT sample partially overlaps with the 2020–2021 tech bull market, which mechanically inflates long-run CARs for that group. The calendar-time analysis provides further support for this reading.

### 5.2 Difference-in-Differences: Did ChatGPT Change the Pricing of AI-Linked Layoffs? (Q4)

#### 5.2.1 The DID Setup

To address Q4 formally, we estimate the following difference-in-differences model:

$$CAR_i[t_1, t_2] = \alpha + \beta_1 \cdot AI_i + \beta_2 \cdot Post_i + \beta_3 \cdot (AI_i \times Post_i) + \gamma' \mathbf{X}_i + \varepsilon_i$$

$AI_i \in \{0,1\}$ indicates whether the layoff announcement was explicitly framed in media coverage as related to AI. $Post_i \in \{0,1\}$ indicates whether the announcement date falls after November 30, 2022. The coefficient $\beta_3$ is the core DID estimator — it captures the incremental market reaction to AI-linked layoffs in the post-ChatGPT period relative to the pre-ChatGPT baseline. The control vector $\mathbf{X}_i$ includes:

| Control Variable | Definition | Rationale |
|---|---|---|
| log(1 + layoff count) | Log of the number of employees laid off | Larger layoffs may trigger stronger reactions |
| Layoff % | Employees laid off / total headcount | A direct measure of restructuring intensity |
| $\hat{\beta}_{MKT}$ | Market beta from FF4 estimation | Controls for systematic risk exposure |
| intl | Indicator for international listings | Controls for factor model misspecification risk |
| prior_6m_return | Cumulative stock return over the prior 126 trading days | Controls for momentum and mean reversion |
| log(1 + funds_raised) | Log of total capital raised historically | Controls for firm financing stage and quality |

All regressions use HC3 heteroskedasticity-robust standard errors.

#### 5.2.2 How the AI Label Was Constructed

Before turning to the DID results, it is worth explaining how the AI indicator was built, because the construction process went through several iterations and the final measurement choice has real implications for the $\beta_3$ estimate.

The first attempt was a simple substring search: any news article mentioning "AI" or "artificial intelligence" got flagged. This produced a positive rate of roughly 50%, which is clearly too broad — it couldn't distinguish whether AI was cited as a cause of the layoffs or merely mentioned as background industry context, and it caught false positives like "BI" or abbreviations embedded in longer words.

The opposite extreme — requiring explicit causal constructions like "laid off due to AI" or "replaced by artificial intelligence" — reduced the positive rate to about 2%, but at the cost of severe underreporting. News coverage almost never uses causal language that direct.

The final approach is a three-tier classification system built on whole-word matching (to avoid substring truncation errors):

| Tier | Definition | N | Rate | Role |
|---|---|---|---|---|
| ai_causal (T3) | AI is explicitly stated as the direct reason for the layoffs | 9 | 1.9% | Upper precision bound |
| ai_primary (T2+T3) | AI is the dominant framing of the article | 22 | 4.7% | Robustness check |
| **ai_broad (T1+T2+T3)** | Any unambiguous mention of AI technology | 74 | **15.8%** | **Main variable** |

The main analysis uses `ai_broad` because the research question is about whether markets perceive an AI narrative around the layoff — not whether AI constitutes the legal cause. The narrower tiers serve as alternative measurements in robustness checks.

One limitation worth being upfront about: roughly 42% of news articles were behind paywalls and could not be accessed. This means the AI label systematically understates the true rate of AI-related coverage, with the true proportion estimated to be around 27% based on the paywall sensitivity analysis. This issue is addressed directly in Section VI.3.

#### 5.2.3 DID Main Results

**Table 5: DID Results Summary (U.S. Primary Sample)**

| Outcome | Specification | β₁(AI) | β₂(Post) | **β₃(AI×Post)** | N | R² |
|---|---|---|---|---|---|---|
| CAR[-1,+1] | No controls | +4.79% | −0.11% | **−7.12%** (p = 0.209) | 428 | 0.003 |
| CAR[-1,+1] | With controls | +2.01% | +1.94% | **−2.71%** (p = 0.483) | 190 | 0.038 |
| CAR[0,+20] | No controls | +10.48% | −2.04% | **−9.44%** (p = 0.659) | 428 | 0.007 |
| CAR[-5,+30] | No controls | −21.24% | −4.83% | **+22.90%** (p = 0.035**) | 428 | 0.013 |
| CAR[-5,+30] | With controls | −28.33% | +2.19% | **+35.07%** (p = 0.037**) | 190 | 0.091 |

In the primary short-run window [-1,+1], the DID coefficient $\beta_3$ is statistically insignificant regardless of whether controls are included (p = 0.209 and p = 0.483, respectively). Put directly: **there is no statistical evidence that the market's pricing of AI-linked layoffs changed structurally after ChatGPT's launch**, at least not in the three-day window around the announcement.

The [-5,+30] results are marginally significant (p = 0.035 and 0.037), with positive coefficients of roughly +23% to +35%. These are technically noteworthy but need to be interpreted carefully. Windows longer than 30 days are notoriously susceptible to macro confounds, especially in a period that included the 2023 tech recovery. The result also evaporates entirely in the [-1,+1] and [0,+20] windows, which raises obvious questions about robustness to window choice. And the controlled specification draws on only N = 190 observations — about half the full sample — which itself limits the reliability of inference.

**Additional findings from the new control variables:**

`prior_6m_return` (the six-month pre-announcement stock return) shows a robust negative effect across multiple specifications; in the CAR[-5,+30] regression the coefficient is −16.0 (p = 0.002***). The intuition is straightforward: firms that had run up significantly in the months before their announcement carried higher market expectations, and a layoff announcement triggers a larger downward revision of those expectations — a classic mean-reversion pattern. `log_funds_raised` is statistically insignificant across all specifications (p > 0.84), suggesting that how much money a company has historically raised is not a meaningful predictor of its announcement-day stock response.

### 5.3 Cross-Sectional OLS: What Predicts the Market Response to Layoffs?

In the second stage of the analysis, we regress the per-event CAR[-1,+1] on observable event characteristics across four progressively richer specifications:

$$CAR_i[-1,+1] = \alpha + \sum_k \gamma_k X_{ki} + \varepsilon_i$$

The core finding: across all four specifications, R² ranges from only 1% to 4%, and no single variable maintains consistent significance. This is not a discouraging finding — it is a substantively meaningful result in its own right. **The short-run price reaction to a layoff announcement is highly idiosyncratic and largely unpredictable from observable event characteristics.** This is consistent with semi-strong form market efficiency: the market absorbs announcement-day information quickly, and no systematic arbitrage pattern emerges from the cross-section.

### 5.4 Size × Sector Heterogeneity

Using the estimation-window $R^2$ from each event's FF4 regression as a proxy for firm systematicity (a proxy for market cap, liquidity, and analyst coverage), the sample is divided into high-, medium-, and low-$R^2$ terciles and crossed with the core tech / non-tech classification, producing a 2×3 analysis matrix:

| R² Group | Sector | N | [-1,+1] CAAR | Interpretation |
|---|---|---|---|---|
| High R² (≥0.548) | Core tech | ~60 | **−0.08%** | Near zero — large-cap tech, market expectations well-formed |
| High R² | Non-tech | ~55 | −0.95% | Moderate negative |
| Mid R² (0.303–0.548) | Core tech | ~60 | −1.10% | Moderate negative |
| Mid R² | Non-tech | ~82 | −1.05% | Moderate negative |
| Low R² (<0.303) | Core tech | ~62 | **−2.15%** | Strongest negative — small/distressed tech firms |
| Low R² | Non-tech | ~110 | −1.20% | Stronger negative |

The high-R² core tech group — which roughly corresponds to FAANG-tier and large-cap tech — shows a CAAR of just −0.08%, almost indistinguishable from zero. This echoes the Mature Firm Sample result and reinforces the central finding from a different angle: firm quality and market cap moderate the directional interpretation of layoff announcements.

---

## VI. Robustness Checks

### 6.1 Placebo DID (False Breakpoint Test)

If ChatGPT's November 2022 launch genuinely produced a structural change in how markets interpret AI-linked layoffs, then the $\beta_3$ estimated at the true breakpoint should be statistically distinguishable from $\beta_3$ estimates at arbitrary false breakpoints.

We test six placebo breakpoints distributed across the sample period and construct the corresponding distribution of $\beta_3$ values. The true breakpoint's $\beta_3 = -1.06\%$ (p = 0.716) falls squarely within the placebo range of $[-3.32\%, +3.28\%]$ — it cannot be distinguished from noise. This is consistent with the main DID result: in the short-run window, the ChatGPT breakpoint does not produce an identifiable structural shift.

### 6.2 Parallel Trends

The identifying assumption of DID requires that, absent ChatGPT, the CAR trajectories of AI-labeled and non-AI-labeled firms would have evolved in parallel. We examine this by computing monthly average CARs for both groups during the pre-ChatGPT period. The Pearson correlation between the two monthly series is $r = 0.776$, indicating directional co-movement that provides at least directional support for the parallel trends assumption.

That said, this check suffers from a severe statistical power problem. During the pre-period (2020–2022), the number of AI-labeled events ranges from just 4 to 13 per month. With sample sizes that small, no formal test can be reasonably powered, and the correlation estimate itself is fragile. This is an intrinsic limitation of the DID design given the available data — we can note it but cannot engineer our way around it.

### 6.3 Paywall Sensitivity Analysis

Forty-two percent of news articles were inaccessible due to paywalls, introducing systematic misclassification (specifically, attenuation bias: AI-labeled events are undercounted, which tends to push $\beta_1$ and $\beta_3$ toward zero). To gauge the sensitivity of the DID result to this measurement error, 50 Monte Carlo simulations were run, each randomly reclassifying a different proportion of paywall articles (ranging from 0% to 50%) as true AI-related events, then re-estimating the DID.

Across all 50 simulations, $\beta_3$ never achieves significance at the 5% level — most simulated p-values fall in the 0.3–0.7 range, and the estimated coefficients track closely with the baseline result. The null finding is not an artifact of paywall-induced measurement error; it reflects a genuine absence of the structural shift the DID is designed to detect.

### 6.4 Calendar-Time Portfolio Method (Clustering Correction)

Standard event study inference assumes that abnormal returns across events are independent, but this assumption is violated when many events cluster in the same calendar period — which is precisely what happened in early 2023. The calendar-time portfolio method (Jaffe 1974; Fama 1998) addresses this by constructing a monthly equal-weighted portfolio of all stocks that announced layoffs in that month, computing the portfolio's monthly excess return against the FF4 benchmark, and then running a time-series OLS on the resulting monthly series. Time-series standard errors replace cross-sectional ones, and event clustering is no longer a concern.

The results: monthly portfolio alphas are statistically insignificant across all subgroups (p > 0.10), with the sole exception of a marginally negative alpha in the pre-ChatGPT group (t = −1.86, p = 0.073*). The large positive CAAR in [-5,+60] is entirely absent under this framework — supporting the interpretation that the long-run drift was driven by the 2023 market recovery, not by the layoff announcements themselves.

### 6.5 Repeat Events

Among the 114 firms with multiple layoff records, there is a meaningful difference between first-time and subsequent announcements:

| Event Type | N | [-1,+1] CAAR | BMP t | Significance |
|---|---|---|---|---|
| First layoff | 271 | −1.20% | −2.41 | ** |
| Subsequent layoffs (2nd or more) | 157 | −0.42% | −0.87 | — |

The weaker, statistically insignificant response to repeat layoffs is consistent with an information decay hypothesis: the market updates its assessment of a firm's restructuring capacity upon the first announcement, so subsequent announcements carry less marginal information. This result also implies that pooling all events with equal weight understates the true signal intensity of first-time layoff disclosures.

### 6.6 Pre-Announcement Drift

To verify that the event study design is clean — i.e., that the market is not systematically pricing in layoff news before the official announcement — we test the CAAR over [-20, -1]. The estimate is −0.82% (p = 0.287), statistically insignificant. A cross-sectional regression of CAR[-1,+1] on CAR[-20,-1] yields a slope of just 0.005 (p = 0.88), meaning pre-announcement stock movements have essentially no predictive power for announcement-day reactions. Taken together, these two results provide reasonable assurance against systematic insider trading or early information leakage, and validate the causal identification underlying the event study design.

---

## VII. Discussion and Conclusions

### 7.1 Verdict on Three Hypotheses

| Hypothesis | Claim | Finding |
|---|---|---|
| H1: Efficient Markets | Layoff announcements produce no systematic abnormal returns | **Partially rejected.** Short-run CARs are significantly negative, inconsistent with weak-form efficiency. But cross-sectional R² < 4% — announcement-day reactions are not predictable, which is consistent with semi-strong efficiency. |
| H2: Signaling | Layoff announcements are quality signals; direction depends on firm characteristics | **Supported.** Mature Firm Sample shows weaker negative reactions; high-R² core tech firms show near-zero effects. Firm quality moderates the market's interpretation. |
| H3: AI Narrative Shift | Post-ChatGPT, AI-linked layoffs receive a more positive market reaction | **Not supported in the primary specification.** In the [-1,+1] window, the DID coefficient is not significant. Placebo tests and paywall sensitivity analysis both support a null result. |

### 7.2 Limitations

**Statistical underpowering of the DID:** The most fundamental constraint. Only 26 post-ChatGPT events are AI-labeled, which means the treatment group is too small to reliably detect the kind of moderate-sized effect we might plausibly expect. This is a data availability problem, not a design flaw — as more AI-era layoff data accumulates over time, future studies will be better positioned to answer this question.

**Systematic measurement error from paywalls:** Forty-two percent of articles being inaccessible means the AI indicator is subject to attenuation bias — the true positive rate is likely higher than observed. The sensitivity analysis provides reassurance within a reasonable range of assumptions, but cannot fully rule out effects under more extreme scenarios.

**Causal attribution for long-run drift:** The +5% CAAR in [-5,+60] substantially overlaps in calendar time with the 2023 tech market recovery. The calendar-time results suggest that this drift most likely reflects aggregate market conditions rather than a causal effect of layoff announcements. Long-run window results are thus interpretively fragile.

**Reverse causality:** The event study framework assumes layoff decisions are exogenous to concurrent stock price movements. In practice, sustained stock price declines can themselves motivate management to cut costs through layoffs — a form of reverse causality that this design does not address. This is a methodological limitation shared by most event studies, not unique to this one.

---

## VIII. Execution Pipeline

```bash
# ── Data collection (run once) ──
python scrapers/01_scrape_layoffs_fyi.py
python scrapers/02_scrape_edgar_8k.py
python scrapers/03_scrape_techcrunch.py
python scrapers/04_scrape_trueup.py
python scrapers/05_combine_sources.py

# ── Main analysis pipeline (run in order) ──
python analysis/01_collect_data.py          # Download stock prices and FF4 factor data
python analysis/02_enrich_events.py         # Add industry, region, initial AI labels
python analysis/03_relabel_ai_tiered.py     # Refine AI labels (three-tier system)
python analysis/04_event_study_ff4.py       # Main event study ★ (generates core CAR results)
python analysis/05_did_regression.py        # DID + cross-sectional OLS ★
python analysis/06_robustness_checks.py     # Six robustness checks
python analysis/07_calendar_time_portfolio.py   # Calendar-time portfolio method
python analysis/08_size_sector_analysis.py  # Size × sector heterogeneity analysis
python analysis/09_export_results.py        # Compile and export final Excel
```

---

## IX. Key Output File Index

| File Path | Contents | Generated By |
|---|---|---|
| `data/processed/master_events_final.csv` | Master event table (481 events, fully labeled) | Steps 2–3 |
| `data/processed/prior_6m_return.csv` | Control variable: 6-month pre-announcement cumulative return | Step 1 |
| `data/processed/funds_raised.csv` | Control variable: total capital raised | From Mature Firm Sample archive |
| `data/results/car_by_event.csv` | Per-event FF4 CARs (467 events) | Step 4 |
| `data/results/car_summary.csv` | Full CAAR summary (all subsamples × windows × models) | Step 4 |
| `data/results/did_crosssection/did_results_us_primary.csv` | DID primary specification (U.S., N = 428) | Step 5 |
| `data/results/did_crosssection/cross_section_v2.csv` | Cross-sectional OLS, four specifications | Step 5 |
| `data/results/robustness/placebo_did_results.csv` | Placebo DID (six breakpoints) | Step 6 |
| `data/results/robustness/paywall_sensitivity.csv` | Paywall sensitivity (50 Monte Carlo draws) | Step 6 |
| `data/results/calendar_time/ct_results.csv` | Calendar-time portfolio monthly alphas | Step 7 |
| `data/results/size_sector/size_sector_caar.csv` | Size × sector CAAR by group | Step 8 |
| `data/results/FINAL_RESULTS.xlsx` | All results consolidated (multiple sheets) | Step 9 |
