"""
Scraper for TechCrunch Tech Layoffs tracker articles.
TechCrunch maintains running lists of layoffs by year, e.g.:
  https://techcrunch.com/2024/01/05/tech-layoffs-2024/
  https://techcrunch.com/2023/01/11/tech-layoffs-tracker-2023/
  https://techcrunch.com/2022/11/21/tech-layoffs-2022/

These are long-form articles with structured bullet points / tables.
We use Playwright + LLM (Claude) to parse the unstructured text into records.
"""

import os
import time
import json
import random
import re
import anthropic
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "techcrunch_raw.csv"

# TechCrunch layoff tracker URLs by year
TC_URLS = {
    2025: "https://techcrunch.com/2025/01/06/tech-layoffs-2025-tracker/",
    2024: "https://techcrunch.com/2024/01/05/tech-layoffs-2024/",
    2023: "https://techcrunch.com/2023/01/11/tech-layoffs-tracker-2023/",
    2022: "https://techcrunch.com/2022/11/21/tech-layoffs-2022/",
    2021: "https://techcrunch.com/2021/04/21/tech-layoffs/",
    2020: "https://techcrunch.com/2020/11/18/here-are-all-the-companies-that-have-done-pandemic-layoffs-2/",
}


def scrape_techcrunch(years: list = None) -> pd.DataFrame:
    """
    Scrape TechCrunch layoff tracker articles and parse with Claude.

    Args:
        years: list of years to scrape. Defaults to [2020..2025].

    Returns:
        DataFrame with layoff records.
    """
    if years is None:
        years = list(TC_URLS.keys())

    all_records = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        for year in years:
            url = TC_URLS.get(year)
            if not url:
                continue

            print(f"Scraping TechCrunch {year}: {url}")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                time.sleep(random.uniform(2, 4))

                # Extract article body text
                article_text = _extract_article_text(page)
                if not article_text:
                    print(f"  No text extracted for {year}")
                    continue

                print(f"  Extracted {len(article_text)} chars. Parsing with Claude...")

                # Parse with Claude API
                records = _parse_with_claude(article_text, year, url)
                all_records.extend(records)
                print(f"  Parsed {len(records)} records for {year}")

                time.sleep(random.uniform(3, 6))  # Be polite

            except Exception as e:
                print(f"  Error scraping {year}: {e}")

        browser.close()

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["source"] = "techcrunch"
    df["source_url"] = df.get("source_url", "")

    print(f"Total TechCrunch records: {len(df)}")
    return df


def _extract_article_text(page) -> str:
    """Extract clean article body text from TechCrunch article."""
    try:
        # TechCrunch article content selectors
        selectors = [
            "article .article-content",
            ".article-content",
            "[class*='article-body']",
            "article",
            "main",
        ]
        for sel in selectors:
            try:
                el = page.query_selector(sel)
                if el:
                    text = el.inner_text()
                    if len(text) > 500:
                        return text
            except Exception:
                continue

        # Fallback: full body text
        return page.inner_text("body")
    except Exception as e:
        print(f"  Text extraction error: {e}")
        return ""


def _parse_with_claude(text: str, year: int, url: str) -> list:
    """
    Use Claude API to extract structured layoff records from unstructured article text.
    Chunks large articles to stay within context limits.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  WARNING: ANTHROPIC_API_KEY not set. Skipping Claude parsing.")
        return _regex_fallback_parse(text, year, url)

    client = anthropic.Anthropic(api_key=api_key)

    # Chunk the text to avoid token limits (max ~6000 chars per chunk)
    chunk_size = 6000
    chunks = [text[i : i + chunk_size] for i in range(0, min(len(text), 60000), chunk_size)]

    all_records = []

    for i, chunk in enumerate(chunks[:10]):  # Max 10 chunks per article
        prompt = f"""The following is part {i+1} of a tech layoffs article from TechCrunch ({year}).
Extract ALL layoff events mentioned. Return a JSON array where each element has:
- company_name (string): company name
- announcement_date (string): date in YYYY-MM-DD format if mentioned, else just the year "{year}"
- layoff_count (number or null): number of employees laid off
- layoff_pct (number or null): percentage of workforce laid off
- affected_teams (string or null): specific teams/divisions mentioned
- ai_mentioned (0 or 1): 1 if AI/automation is mentioned as reason, else 0
- ai_quote (string or null): exact quote mentioning AI if present
- news_source_url: "{url}"

If no layoff events are in this chunk, return an empty array [].
Return ONLY valid JSON, no other text.

Article text:
{chunk}"""

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",  # Use fast/cheap model for extraction
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()

            # Extract JSON from response
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                records = json.loads(json_match.group())
                all_records.extend(records)
        except Exception as e:
            print(f"  Claude parsing error (chunk {i+1}): {e}")

        time.sleep(0.5)  # Rate limit

    return all_records


def _regex_fallback_parse(text: str, year: int, url: str) -> list:
    """
    Regex-based fallback parser when Claude API is not available.
    Less accurate but catches obvious patterns like "Company laid off N employees".
    """
    records = []
    pattern = re.compile(
        r"([A-Z][A-Za-z\s&,\.]+?)\s+(?:laid off|cut|eliminated|reduced)\s+(?:approximately\s+)?(\d[\d,]*)\s+(?:employees|workers|jobs|positions|people|roles)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        company = match.group(1).strip().rstrip(",.")
        count = int(match.group(2).replace(",", ""))
        records.append(
            {
                "company_name": company,
                "announcement_date": str(year),
                "layoff_count": count,
                "layoff_pct": None,
                "affected_teams": None,
                "ai_mentioned": 0,
                "ai_quote": None,
                "news_source_url": url,
            }
        )
    return records


if __name__ == "__main__":
    df = scrape_techcrunch(years=[2023, 2024, 2025])
    if not df.empty:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved {len(df)} records to {OUTPUT_PATH}")
        print(df.head(10).to_string())
    else:
        print("No data collected.")
