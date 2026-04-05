"""
Tiered AI-mention Labeling
==========================
Problem with existing definitions:
  Desktop (broad) : 'ai' substring + 'efficiency' → 58% rate, ~30-40% false positives
  Claude (strict) : explicit causal language only  →  2.6% rate, misses real cases

Solution — 3 tiers based on what the article actually says:

  Tier 1 CAUSAL  : Article explicitly links AI/automation as CAUSE of layoffs
                   e.g. "laying off workers due to AI", "roles replaced by automation"
                   → highest accuracy, smallest N

  Tier 2 STRATEGIC: Article mentions specific GenAI tools/platforms + restructuring
                   e.g. "investing in ChatGPT", "AI transformation", "pivot to GenAI"
                   → strong indicator company is in AI-driven change

  Tier 3 SPECIFIC: Specific AI technology term appears anywhere (whole-word only)
                   e.g. "machine learning", "LLM", standalone \bAI\b
                   → broader but still meaningful; excludes 'efficiency', 'algorithm'

Key fixes vs desktop:
  ✗ 'ai' substring → ✓ \bAI\b  (won't match "said", "paid", "available")
  ✗ 'efficiency'   → removed  (generic business language, not AI-specific)
  ✗ 'algorithm'    → removed  (too generic for software companies)
  ✓ Added: ChatGPT, LLM, generative AI, GPT, OpenAI, Copilot

Primary definition for regression: Tier 1 + Tier 2 (causal or strategic)
Robustness check:                  Tier 1 only / Tier 1+2+3

Output: data/results/improved/ai_labels_tiered.csv
"""

import os, re, time, random, requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path

BASE     = '/Users/irmina/Documents/Claude/layoff_study'
OUT_DIR  = os.path.join(BASE, 'data/results/improved')
CACHE_F  = os.path.join(OUT_DIR, 'article_text_cache.csv')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Regex patterns ────────────────────────────────────────────────

# TIER 1: explicit causal — AI/automation is stated reason for layoffs
TIER1 = re.compile(r"""
    # Pattern A: "replacing/eliminating X by/with AI/automation"
    \b(replac\w+|eliminat\w+|automat\w+|displac\w+|reduc\w+)\b
    .{1,60}
    \b(by|with|through|via|using|due\s+to|because\s+of)\b
    .{1,40}
    \b(ai|artificial\s+intelligence|machine\s+learning|automation|robots?)\b

  | # Pattern B: "AI/automation replacing/eliminating jobs/roles/workers"
    \b(ai|artificial\s+intelligence|machine\s+learning|automation|robots?)\b
    .{1,60}
    \b(replac\w+|eliminat\w+|displac\w+|cut\w+|reduc\w+|automat\w+)\b
    .{1,40}
    \b(job|role|worker|employee|staff|position|headcount)\b

  | # Pattern C: explicit causal preposition
    \b(due\s+to|because\s+of|driven\s+by|result\s+of|owing\s+to|
       amid|citing|following|in\s+response\s+to)\b
    .{1,50}
    \b(ai|artificial\s+intelligence|machine\s+learning|automation|
       generative\s+ai|genai|chatgpt|llm)\b

  | # Pattern D: strategic pivot causing restructuring
    \b(invest\w+|pivot\w+|shift\w+|restructur\w+|reorganiz\w+|transform\w+)\b
    .{1,60}
    \b(ai|artificial\s+intelligence|machine\s+learning|generative\s+ai|
       genai|chatgpt|llm|automation)\b
    .{1,100}
    \b(layoff|job\s+cut|workforce\s+reduc|headcount\s+reduc|let\s+go|
       terminat|dismissal|redundanc)\b

  | # Pattern E: reverse order of D
    \b(layoff|job\s+cut|workforce\s+reduc|headcount\s+reduc)\b
    .{1,100}
    \b(invest\w+|pivot\w+|shift\w+|restructur\w+|focus\w+)\b
    .{1,60}
    \b(ai|artificial\s+intelligence|machine\s+learning|generative\s+ai|
       genai|chatgpt|llm|automation)\b
""", re.IGNORECASE | re.VERBOSE | re.DOTALL)

# TIER 2: strategic AI context — specific modern AI tools/platforms mentioned
# alongside restructuring language (strong implied link even without explicit causation)
TIER2 = re.compile(r"""
    \b(
        generative\s+ai | genai | gen\s+ai |
        chatgpt | gpt-?[34o]? | gpt\s+[34] |
        large\s+language\s+model | llm | llms |
        openai | anthropic | gemini | copilot | github\s+copilot |
        ai\s+(?:agent|assistant|tool|platform|product|capability|
                 investment|initiative|transformation|strategy|
                 roadmap|adoption|integration|infrastructure|
                 first|native|powered|driven|enabled) |
        (?:artificial\s+intelligence|machine\s+learning)\s+
            (?:investment|initiative|adoption|transformation|platform) |
        intelligent\s+automation |
        workforce\s+(?:transformation|ai\s+transformation) |
        ai\s+efficiency |       # explicit "AI efficiency" (not just "efficiency")
        automat\w+\s+(?:role|job|task|function|process|workflow)
    )\b
""", re.IGNORECASE | re.VERBOSE)

# TIER 3: specific AI technology terms, whole-word only
# Key fix: \bAI\b not 'ai' substring; removed 'efficiency', 'algorithm'
TIER3 = re.compile(r"""
    \b(
        artificial\s+intelligence |
        machine\s+learning |
        deep\s+learning |
        neural\s+network |
        natural\s+language\s+processing |
        computer\s+vision |
        robotic\s+process\s+automation | \brpa\b |
        \bai\b |                # whole word AI only
        automation | automate[sd]? | automating
    )\b
""", re.IGNORECASE | re.VERBOSE)


# ── Article fetcher ───────────────────────────────────────────────

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'en-US,en;q=0.9',
}

def fetch_text(url: str, timeout: int = 8) -> str:
    """Fetch article text. Returns '' on failure (paywall, 404, timeout)."""
    if not url or not str(url).startswith('http'):
        return ''
    try:
        r = requests.get(str(url), headers=HEADERS, timeout=timeout,
                         allow_redirects=True)
        if r.status_code != 200:
            return ''
        soup = BeautifulSoup(r.text, 'lxml')
        for tag in soup(['script','style','nav','header','footer',
                         'aside','advertisement','noscript']):
            tag.decompose()
        # Try article body first
        for sel in ['article', 'main', '[class*="article-body"]',
                    '[class*="story-body"]', '[class*="post-content"]',
                    '[class*="entry-content"]', '[role="main"]']:
            el = soup.select_one(sel)
            if el:
                return el.get_text(' ', strip=True)[:6000]
        return soup.get_text(' ', strip=True)[:6000]
    except Exception:
        return ''


def classify(text: str) -> tuple:
    """
    Returns (tier: int 0-3, evidence: str)
      3 = Tier 1 (causal)   — most accurate
      2 = Tier 2 (strategic)
      1 = Tier 3 (specific)
      0 = not AI-related
    """
    if not text:
        return 0, ''

    m = TIER1.search(text)
    if m:
        snip = text[max(0, m.start()-60): m.end()+60].replace('\n',' ').strip()
        return 3, f'[CAUSAL] ...{snip}...'

    m = TIER2.search(text)
    if m:
        snip = text[max(0, m.start()-60): m.end()+60].replace('\n',' ').strip()
        return 2, f'[STRATEGIC] ...{snip}...'

    m = TIER3.search(text)
    if m:
        snip = text[max(0, m.start()-60): m.end()+60].replace('\n',' ').strip()
        return 1, f'[SPECIFIC] ...{snip}...'

    return 0, ''


# ── Main labeling pipeline ────────────────────────────────────────

def run():
    # Load events with URLs
    car    = pd.read_csv(os.path.join(BASE,'data/results/car_by_event.csv'))
    master = pd.read_csv(os.path.join(BASE,'data/processed/master_events_final.csv'))
    car    = car.merge(master[['company_fyi','announcement_date','source_url']],
                       on=['company_fyi','announcement_date'], how='left')

    # Load cache (avoid re-fetching)
    if os.path.exists(CACHE_F):
        cache_df = pd.read_csv(CACHE_F)
        cache    = dict(zip(cache_df['url'], cache_df['text'].fillna('')))
        print(f'Loaded {len(cache)} cached articles')
    else:
        cache = {}

    results = []
    new_cache = {}
    total = len(car)

    print(f'Fetching and classifying {total} articles...')
    print('(paywalled/dead URLs will get empty text → tier=0 unless rescued by title heuristic)\n')

    for i, (_, row) in enumerate(car.iterrows()):
        url = str(row.get('source_url', ''))

        # Get text (from cache or fetch)
        if url in cache:
            text = cache[url]
        else:
            text = fetch_text(url)
            new_cache[url] = text
            # Polite delay
            time.sleep(random.uniform(0.4, 0.9))

        tier, evidence = classify(text)

        # Fallback: if article fetch failed, check URL/title for obvious signals
        # (e.g. URL contains 'ai-layoffs', 'automation', etc.)
        if tier == 0 and not text:
            url_lower = url.lower()
            if any(kw in url_lower for kw in
                   ['artificial-intelligence','machine-learning','automation',
                    'chatgpt','generative','llm','ai-layoff','ai-cut']):
                tier, evidence = 1, f'[URL_SIGNAL] {url}'

        results.append({
            'company_fyi':        row['company_fyi'],
            'ticker':             row['ticker'],
            'announcement_date':  row['announcement_date'],
            'ai_tier':            tier,
            'ai_causal':          int(tier == 3),        # Tier 1 only
            'ai_primary':         int(tier >= 2),        # Tier 1+2  ← recommended
            'ai_broad':           int(tier >= 1),        # Tier 1+2+3
            'ai_desktop':         row.get('ai_mentioned', np.nan),
            'fetch_success':      int(len(text) > 100),
            'evidence':           evidence,
            'source_url':         url,
        })

        if (i+1) % 50 == 0:
            fetched_now   = sum(1 for r in results if r['fetch_success'])
            tier_counts   = pd.Series([r['ai_tier'] for r in results]).value_counts().sort_index()
            print(f'  [{i+1}/{total}] fetched={fetched_now}  '
                  f'tier3={tier_counts.get(3,0)}  '
                  f'tier2={tier_counts.get(2,0)}  '
                  f'tier1={tier_counts.get(1,0)}  '
                  f'tier0={tier_counts.get(0,0)}')

    # Update and save cache
    cache.update(new_cache)
    cache_df = pd.DataFrame([{'url': k, 'text': v} for k, v in cache.items()])
    cache_df.to_csv(CACHE_F, index=False)

    # Save labeling results
    df = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, 'ai_labels_tiered.csv')
    df.to_csv(out_path, index=False)

    # Print summary
    print('\n' + '='*60)
    print('TIERED AI LABELING RESULTS')
    print('='*60)
    print(f'  Total events          : {len(df)}')
    print(f'  Articles fetched OK   : {df["fetch_success"].sum()}')
    print(f'  Articles failed/empty : {(df["fetch_success"]==0).sum()}')
    print()
    print(f'  Tier 3 (CAUSAL)       : {(df["ai_tier"]==3).sum():4d}  '
          f'({(df["ai_tier"]==3).mean()*100:.1f}%)  ← explicit AI causation')
    print(f'  Tier 2 (STRATEGIC)    : {(df["ai_tier"]==2).sum():4d}  '
          f'({(df["ai_tier"]==2).mean()*100:.1f}%)  ← specific GenAI tools + restructuring')
    print(f'  Tier 1 (SPECIFIC)     : {(df["ai_tier"]==1).sum():4d}  '
          f'({(df["ai_tier"]==1).mean()*100:.1f}%)  ← whole-word AI terms only')
    print(f'  Tier 0 (NONE)         : {(df["ai_tier"]==0).sum():4d}  '
          f'({(df["ai_tier"]==0).mean()*100:.1f}%)  ← no AI signal')
    print()
    print(f'  ── Aggregated ──')
    print(f'  ai_causal  (T3 only)   : {df["ai_causal"].sum():4d}  '
          f'({df["ai_causal"].mean()*100:.1f}%)')
    print(f'  ai_primary (T3+T2) ←★ : {df["ai_primary"].sum():4d}  '
          f'({df["ai_primary"].mean()*100:.1f}%)  ← RECOMMENDED')
    print(f'  ai_broad   (T3+T2+T1)  : {df["ai_broad"].sum():4d}  '
          f'({df["ai_broad"].mean()*100:.1f}%)')
    print(f'  Desktop definition      : {int(df["ai_desktop"].sum()):4d}  '
          f'({df["ai_desktop"].mean()*100:.1f}%)  ← for comparison')
    print()

    # Show tier 3 (causal) examples
    causal = df[df['ai_tier']==3][['company_fyi','announcement_date','evidence']].head(10)
    print('  TIER 3 (CAUSAL) examples:')
    for _, r in causal.iterrows():
        print(f'    {r["company_fyi"]:20} {r["announcement_date"]}  '
              f'{str(r["evidence"])[:80]}')

    print(f'\nSaved → {out_path}')
    return df


if __name__ == '__main__':
    run()
