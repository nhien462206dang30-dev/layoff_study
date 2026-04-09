"""
Enrich master_events.csv with three fixes:
  1. Resolve LOW_CONFLICT tickers (prefer EDGAR for US, yfinance for non-US)
  2. Label ai_mentioned via regex keyword scan on source news URLs
  3. Tag US vs non-US listed stocks and map to appropriate benchmark index
"""

import re
import time
import random
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

INPUT  = Path("data/processed/master_events.csv")
OUTPUT = Path("data/processed/master_events_enriched.csv")

# ── AI keyword patterns ───────────────────────────────────────────────────────
# Grouped by strength: strong = explicit AI causation; weak = general mention
AI_STRONG = re.compile(
    r"""(
        (replac\w+|automat\w+|displac\w+|eliminat\w+)\s+\w+\s+(by|with|through|via)\s+(ai|artificial intelligence|machine learning|automation|llm|genai|generative ai)
      | (ai|artificial intelligence|machine learning|generative ai|llm|chatgpt|large language model)\s+\w+\s+(replac\w+|automat\w+|displac\w+|reduc\w+|elimin\w+)
      | (due to|because of|driven by|result of|owing to)\s+\w*\s*(ai|automation|artificial intelligence|machine learning)
      | ai[\s-]?(driven|enabled|powered|first|related)\s+(efficiency|productivity|transformation|restructur\w+)
      | (invest\w+|pivot\w+|shift\w+)\s+(in|into|toward)\s+(ai|artificial intelligence|machine learning)
    )""",
    re.IGNORECASE | re.VERBOSE
)

AI_WEAK = re.compile(
    r"""\b(
        artificial intelligence | machine learning | generative ai | gen[\s-]?ai
      | large language model | llm | chatgpt | gpt-4 | openai | copilot
      | ai[\s-]tools | ai[\s-]strategy | ai[\s-]invest
      | automation | automate | automated | automat\w+
      | workforce transformation | digital transformation
      | efficiency gain | cost efficiency | operational efficiency
    )\b""",
    re.IGNORECASE | re.VERBOSE
)

# ── US market detection ───────────────────────────────────────────────────────
# Tickers with these suffixes are non-US primary listings
NON_US_SUFFIX = re.compile(r'\.(L|AX|TO|HK|KS|KQ|T|PA|AS|SW|DE|MU|F|SG|VI|BR|MC|MI|CO|ST|OL|HE|LS|LN|BO|NS|SI)$', re.IGNORECASE)

US_EXCHANGES = {'NYQ', 'NMS', 'NGM', 'NCM', 'ASE', 'NYSE', 'NASDAQ', 'BATS', 'PCX'}

# Benchmark index mapping for non-US markets
BENCHMARK_MAP = {
    '.L':  '^FTSE',      # London → FTSE 100
    '.AX': '^AXJO',      # Australia → ASX 200
    '.TO': '^GSPTSE',    # Toronto → S&P/TSX
    '.HK': '^HSI',       # Hong Kong → Hang Seng
    '.KS': '^KS11',      # Korea → KOSPI
    '.KQ': '^KQ11',      # KOSDAQ
    '.T':  '^N225',      # Tokyo → Nikkei 225
    '.PA': '^FCHI',      # Paris → CAC 40
    '.AS': '^AEX',       # Amsterdam → AEX
    '.SW': '^SSMI',      # Switzerland → SMI
    '.DE': '^GDAXI',     # Germany → DAX
    '.MU': '^GDAXI',
    '.F':  '^GDAXI',
}
DEFAULT_INTL_BENCHMARK = '^MSCI'  # Fallback (use ^ACWI in practice)


def get_listing_region(ticker: str, exchange: str = '', country: str = '') -> str:
    """Return 'US' or 'INTL' based on ticker suffix and exchange."""
    if not ticker or str(ticker) == 'nan':
        return 'UNKNOWN'
    suffix_match = NON_US_SUFFIX.search(str(ticker))
    if suffix_match:
        return 'INTL'
    if exchange and exchange.upper() in US_EXCHANGES:
        return 'US'
    # Heuristic: US tickers are 1-5 uppercase letters with no dots
    if re.match(r'^[A-Z]{1,5}$', str(ticker)):
        return 'US'
    return 'INTL'


def get_benchmark(ticker: str) -> str:
    """Map a ticker to its appropriate benchmark index."""
    suffix_match = NON_US_SUFFIX.search(str(ticker) if ticker else '')
    if suffix_match:
        suffix = '.' + suffix_match.group(1).upper()
        return BENCHMARK_MAP.get(suffix, DEFAULT_INTL_BENCHMARK)
    return '^GSPC'  # Default: S&P 500 for US stocks


def fetch_article_text(url: str, timeout: int = 10) -> str:
    """Fetch news article text. Returns empty string on failure."""
    if not url or not str(url).startswith('http'):
        return ''
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        r = requests.get(str(url), headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return ''
        soup = BeautifulSoup(r.text, 'lxml')
        # Remove nav/header/footer/script noise
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            tag.decompose()
        # Get main content
        for selector in ['article', 'main', '[class*="article-body"]', '[class*="story-body"]', '[class*="post-content"]']:
            el = soup.select_one(selector)
            if el:
                return el.get_text(' ', strip=True)[:8000]
        return soup.get_text(' ', strip=True)[:8000]
    except Exception:
        return ''


def label_ai_mention(text: str, company: str = '') -> tuple[int, str]:
    """
    Returns (ai_mentioned: 0/1, evidence: str).
    Checks strong patterns first, then weak.
    """
    if not text:
        return 0, ''

    # Strong match: explicit AI causation language
    m = AI_STRONG.search(text)
    if m:
        snippet = text[max(0, m.start()-50): m.end()+50].strip()
        return 1, f"[STRONG] ...{snippet}..."

    # Weak match: AI mentioned anywhere in article
    m = AI_WEAK.search(text)
    if m:
        snippet = text[max(0, m.start()-80): m.end()+80].strip()
        return 1, f"[WEAK] ...{snippet}..."

    return 0, ''


# ── Fix 1: Resolve LOW_CONFLICT tickers ──────────────────────────────────────
def fix_low_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """
    For LOW_CONFLICT rows: prefer EDGAR ticker for US-listed companies
    (Yahoo Finance was returning European cross-listings like BI1.MU for BIGC).
    For non-US companies, EDGAR won't have them, so yfinance is correct.
    """
    mask = df['confidence'] == 'LOW_CONFLICT'
    for idx in df[mask].index:
        edgar_tk = str(df.at[idx, 'ticker_edgar'])
        yf_tk    = str(df.at[idx, 'ticker_yfinance'])
        country  = str(df.at[idx, 'country'])

        # If EDGAR has a valid US ticker, always prefer it
        if edgar_tk and edgar_tk != 'nan' and re.match(r'^[A-Z]{1,5}Q?$', edgar_tk):
            df.at[idx, 'ticker']     = edgar_tk
            df.at[idx, 'confidence'] = 'HIGH'   # EDGAR = authoritative
        elif yf_tk and yf_tk != 'nan':
            df.at[idx, 'ticker']     = yf_tk
            df.at[idx, 'confidence'] = 'MEDIUM'

    resolved = mask.sum() - (df['confidence'] == 'LOW_CONFLICT').sum()
    print(f"  LOW_CONFLICT resolved: {resolved} / {mask.sum()}")
    return df


# ── Fix 2: ai_mentioned via URL fetch + regex ─────────────────────────────────
def label_ai_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch source URLs and apply AI keyword regex."""
    df['ai_mentioned'] = 0
    df['ai_evidence']  = ''

    # Deduplicate URLs to avoid refetching same article for duplicate events
    url_cache: dict[str, tuple[int, str]] = {}

    total = len(df)
    labeled_pos = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        url = str(row.get('source_url', ''))

        if url in url_cache:
            label, evidence = url_cache[url]
        else:
            text = fetch_article_text(url)
            label, evidence = label_ai_mention(text, row.get('company_fyi', ''))
            url_cache[url] = (label, evidence)
            # Polite delay only on actual fetches
            time.sleep(random.uniform(0.3, 0.7))

        df.at[idx, 'ai_mentioned'] = label
        df.at[idx, 'ai_evidence']  = evidence
        if label:
            labeled_pos += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] AI-positive so far: {labeled_pos}")

    print(f"  ai_mentioned=1: {labeled_pos} / {total} ({100*labeled_pos/total:.1f}%)")
    return df


# ── Fix 3: US vs INTL tagging + benchmark assignment ─────────────────────────
def tag_listing_region(df: pd.DataFrame) -> pd.DataFrame:
    df['listing_region'] = df.apply(
        lambda r: get_listing_region(r['ticker'], r.get('yf_exchange', ''), r.get('country', '')),
        axis=1
    )
    df['benchmark_index'] = df['ticker'].apply(get_benchmark)

    print(f"  US-listed    : {(df['listing_region']=='US').sum()}")
    print(f"  INTL-listed  : {(df['listing_region']=='INTL').sum()}")
    print(f"  Unknown      : {(df['listing_region']=='UNKNOWN').sum()}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} events from {INPUT}")

    print("\n── Fix 1: Resolve LOW_CONFLICT tickers ──")
    df = fix_low_conflict(df)

    print("\n── Fix 2: Label ai_mentioned via URL fetch + regex ──")
    df = label_ai_mentions(df)

    print("\n── Fix 3: Tag listing region + benchmark index ──")
    df = tag_listing_region(df)

    # Save
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved enriched events → {OUTPUT}")

    # Summary
    ready = df[df['status'] == 'READY']
    print(f"\n=== ENRICHMENT SUMMARY ===")
    print(f"Total events             : {len(df)}")
    print(f"READY events             : {len(ready)}")
    print(f"  US-listed (READY)      : {(ready['listing_region']=='US').sum()}")
    print(f"  INTL-listed (READY)    : {(ready['listing_region']=='INTL').sum()}")
    print(f"  ai_mentioned=1 (READY) : {ready['ai_mentioned'].eq(1).sum()} ({100*ready['ai_mentioned'].eq(1).sum()/len(ready):.1f}%)")
    print(f"  Post-GenAI             : {(ready['period']=='post_genai').sum()}")
    print(f"  Pre-GenAI              : {(ready['period']=='pre_genai').sum()}")


if __name__ == "__main__":
    run()
