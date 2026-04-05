# AI-Mention Definition Comparison

## Research Question

Does the market react differently to layoff announcements that cite AI/automation as a reason or context, compared to purely financial/restructuring layoffs?

The key measurement challenge: **how do we identify whether a layoff is "AI-related"?** Different definitions produce very different samples and results.

---

## The Three Definitions Compared

### Definition 1 — Desktop (Broad Keyword Match)

**Source:** `裁员影响模型/为上市公司匹配ticker并清洗脏数据3/5手动匹配错误公司并删除后.csv`

**Method:** Fetch news article text. Mark `ai_mentioned = 1` if any of these strings appear (case-insensitive substring):

```python
ai_keywords = [
    'artificial intelligence', 'AI',        # 'AI' as substring → matches "said", "paid"
    'machine learning', 'automation', 'automated',
    'algorithm', 'neural network', 'deep learning',
    'robotics', 'computer vision', 'natural language processing',
    '智能', '人工智能', '机器学习', '自动化', '算法', '机器人',
    '计算机视觉', '自然语言处理', '技术效率',
    'efficiency'                             # matches almost every layoff article
]
```

**Problems:**
- `'ai' in text.lower()` matches substrings: "s**ai**d", "p**ai**d", "av**ai**lable", "aga**i**nst"
- `'efficiency'` appears in virtually every restructuring article
- `'algorithm'` is generic for any tech company

**Result:** N = 234 / 467 = **50.1%** labeled AI  
**Assessment:** ⚠️ Significantly over-labeled (~30-40% likely false positives)

---

### Definition 2 — Claude Original (Strict Causal Regex)

**Source:** `analysis/enrich_events.py` — `AI_STRONG` and `AI_WEAK` patterns

**Method:** Fetch article text, require explicit AI causation language:

```python
AI_STRONG = r"""
  (replacing|automating|displacing) ... (by|with) ... (AI|automation|machine learning)
  | (due to|because of|driven by) ... (AI|automation)
  | AI-driven efficiency | investing into AI [+ restructuring]
"""
```

**Problems:**
- Very strict — only explicit causal phrases trigger a match
- Most articles describe AI investment as opportunity, not as replacing workers
- Many genuine AI-driven layoffs use softer language ("transformation", "efficiency")

**Result:** N = 12 / 467 = **2.6%** labeled AI  
**Assessment:** ❌ Severely under-labeled — misses most real AI-related layoffs

---

### Definition 3 — Tiered (This Study's Recommended Approach)

**Source:** `analysis/relabel_tiered.py`

**Method:** Fetch article text, apply three-tier whole-word matching:

#### Tier 3 — CAUSAL (highest confidence)
Explicit statement that AI/automation caused the layoffs:
```
"laying off workers due to AI"
"roles replaced by automation"
"restructuring toward AI [+ headcount reduction language]"
"due to / because of / driven by + [AI|automation|ChatGPT|LLM]"
```

#### Tier 2 — STRATEGIC (strong implied link)
Specific modern AI tools/platforms mentioned alongside restructuring:
```
"generative AI", "ChatGPT", "GPT-4", "LLM", "large language model"
"OpenAI", "Copilot", "Gemini", "Anthropic"
"AI transformation", "AI strategy", "AI investment"
"automating roles/tasks/workflows"
```

#### Tier 1 — SPECIFIC (technology mentioned, no explicit link)
Specific AI technology terms appear in article (whole-word only):
```
\bAI\b  (NOT substring — won't match "said", "paid")
"machine learning", "deep learning", "neural network"
"automation", "automated", "automating"
"natural language processing", "computer vision", "RPA"
```

**Key fixes vs Desktop:**
- `\bAI\b` word boundary (not substring)
- Removed: `efficiency`, `algorithm`
- Added: `ChatGPT`, `LLM`, `generative AI`, `OpenAI`, `Copilot`

**Limitation:** 42% of articles (197/467) are behind paywalls (Bloomberg, WSJ, CNBC, Reuters) and return empty text → systematic undercounting for major-outlet coverage.

---

## Results Comparison

### Sample Sizes

| Definition | N (ai=1) | % | Pre-ChatGPT AI | Post-ChatGPT AI |
|------------|----------|---|----------------|-----------------|
| Desktop (broad) | 234 | 50.1% | 54 | 172 |
| Claude original (strict) | 12 | 2.6% | 1 | 11 |
| **Tiered: ai_causal (T3)** | **9** | **1.9%** | **1** | **8** |
| **Tiered: ai_primary (T3+T2)** | **22** | **4.7%** | **5** | **17** |
| **Tiered: ai_broad (T3+T2+T1) ★** | **74** | **15.8%** | **13** | **59** |

Estimated true count (accounting for 42% paywall gap): **~128 (27%)** if paywalled articles have similar AI-mention rate.

### Cross-Sectional Regression: CAR[-1,+1]

All definitions find the same result: **nothing is significant**. Observable event characteristics do not predict short-window announcement returns. R² ≈ 1–2% across all specifications. This aligns with market efficiency.

| Definition | ai_mentioned coef | p-value | R² |
|------------|-------------------|---------|-----|
| Desktop | +0.033 | 0.28 | 1.1% |
| Claude orig (12 events) | −22.2% | 0.004*** | 2.3% |
| ai_broad (74 events) | +0.07% | 0.98 | 0.0% |
| ai_primary (22 events) | +5.3% | 0.38 | 0.2% |

> ⚠️ The "significant" result from Claude original (−22.2%\*\*\*) was driven by only 12 events. After correcting the definition, the signal disappears.

### DID Regression: Did ChatGPT Change the Market Reaction to AI Layoffs?

**Breakpoint:** 2022-11-30 (ChatGPT launch)  
**Spec:** CAR = α + β₁·ai + β₂·post + **β₃·(ai×post)** + controls + ε  
**β₃ = DID estimate**

#### Short window CAR[-1,+1]

| Definition | β₃ | SE | p | N |
|------------|----|----|---|---|
| Desktop | −5.4% | 3.3 | 0.097* | 235 |
| ai_broad | −4.1% | 3.4 | 0.234 | 235 |
| ai_primary | −5.3% | 6.1 | 0.384 | 444 |

→ Consistent direction (negative): post-ChatGPT, AI-labelled layoffs slightly underperform non-AI. But statistical evidence is weak across all definitions.

#### Long window CAR[-5,+30]

| Definition | β₁ (ai main) | β₃ (DID) | p(β₃) | N |
|------------|-------------|----------|--------|---|
| Claude orig (12 events) | −22.2% | +51–70%** | 0.10–0.05 | 467 |
| ai_broad (74 events) | −14.6%** | +10.1% | 0.15 | 444 |
| ai_primary (22 events) | −21.8%*** | +18.4%*** | 0.006 | 444 |

→ **Consistent pattern across definitions**: AI-related layoffs experience worse long-run performance overall (β₁ < 0), but this is partially offset post-ChatGPT (β₃ > 0). The DID term reaches significance only in small samples (ai_primary, ai_original) — likely noise given tiny pre-ChatGPT AI groups (N=1 and N=5 respectively).

### Event Study: Average CAAR (All Events)

This analysis does **not** depend on the ai_mentioned definition:

| Window | CAAR | Patell t | Significance |
|--------|------|----------|--------------|
| [-1,+1] | −0.49% | −3.03 | *** |
| [0,+5] | −0.35% | −2.06 | ** |
| [0,+20] | +2.25% | +2.42 | ** |
| [-5,+60] | +6.40% | +4.31 | *** |

> **Note:** The Claude event study finds negative short-term CAR (−0.49%), opposite to the desktop result (+1.00%). This is likely due to different sample composition: the desktop manually filtered to Post-IPO public companies with solid trading history, while the Claude sample includes more early-stage and international listings.

---

## Recommended Definition for Future Analysis

**Use `ai_broad` (N=74) as primary, `ai_primary` (N=22) as robustness check.**

Rationale:
1. Whole-word matching eliminates the biggest source of false positives
2. Removing `efficiency` reduces noise substantially  
3. N=74 provides enough power for regression (vs N=9/22 which are too small)
4. Clearly interpretable: "article explicitly mentions a specific AI technology"

**Honest label:** The variable measures *"did the news coverage of this layoff mention AI/automation technology"* — not *"was AI the true cause"*. Causal attribution from news text is impossible without structured disclosure data.

**To improve further:** Use structured sources (SEC 8-K filings with standardized language, earnings call transcripts) or human annotation of a stratified sample.

---

## Files

| File | Description |
|------|-------------|
| `analysis/relabel_tiered.py` | Three-tier labeling script |
| `analysis/improved_analysis.py` | DID + grouped event study with new labels |
| `data/results/improved/ai_labels_tiered.csv` | Full tiered labels for all 467 events |
| `data/results/improved/final_labels_and_cars.csv` | Final labels merged with CARs |
| `data/results/improved/did_results.csv` | DID regression output (ai_broad) |
| `data/results/improved/car_by_event_v2.csv` | Updated event-level CARs |
| `data/results/improved/figA_caar_ai_vs_nonai.png` | CAAR: AI vs non-AI groups |
| `data/results/improved/figB_caar_4way.png` | CAAR: 4-way (ai × era) |
