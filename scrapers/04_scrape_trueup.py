"""
Scraper for Trueup.io Tech Layoff Tracker using Playwright.
URL: https://www.trueup.io/layoffs
The site renders data dynamically via React/Next.js.
"""

import time
import random
import json
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "trueup_raw.csv"


def scrape_trueup() -> pd.DataFrame:
    """
    Scrape trueup.io/layoffs. Intercepts Next.js API or JSON responses.
    Falls back to DOM table scraping.
    """
    records = []
    api_data = []

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

        def capture_response(response):
            url = response.url
            # Trueup uses Next.js API routes — intercept JSON responses
            if response.status == 200 and (
                "api/layoffs" in url
                or "layoffs" in url
                or "_next/data" in url
            ):
                ct = response.headers.get("content-type", "")
                if "json" in ct:
                    try:
                        data = response.json()
                        api_data.append({"url": url, "data": data})
                    except Exception:
                        pass

        page.on("response", capture_response)

        print("Loading trueup.io/layoffs ...")
        try:
            page.goto(
                "https://www.trueup.io/layoffs",
                wait_until="networkidle",
                timeout=60000,
            )
        except Exception as e:
            print(f"Navigation error: {e}")

        time.sleep(4)

        # Scroll through the page to trigger lazy loading
        for i in range(8):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(random.uniform(1.0, 2.0))

        # Try to parse API responses first
        for resp in api_data:
            extracted = _parse_trueup_json(resp["data"])
            records.extend(extracted)

        # DOM fallback
        if not records:
            print("No API data captured. Trying DOM scrape...")
            records = _dom_scrape_trueup(page)

        browser.close()

    if not records:
        print("WARNING: No records collected from Trueup.io")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["source"] = "trueup.io"
    df["source_url"] = "https://www.trueup.io/layoffs"

    # Normalize column names
    col_map = {
        "company": "company_name",
        "name": "company_name",
        "headcount": "layoff_count",
        "laid_off": "layoff_count",
        "laidOff": "layoff_count",
        "numLaidOff": "layoff_count",
        "date": "announcement_date",
        "layoffDate": "announcement_date",
        "cutPercent": "layoff_pct",
        "percentage": "layoff_pct",
        "industry": "industry",
        "country": "country",
        "city": "location_hq",
        "location": "location_hq",
        "sourceUrl": "news_source_url",
        "url": "news_source_url",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    keep = [
        "company_name", "announcement_date", "layoff_count", "layoff_pct",
        "industry", "country", "location_hq", "news_source_url", "source", "source_url"
    ]
    df = df[[c for c in keep if c in df.columns]]

    print(f"Collected {len(df)} records from Trueup.io")
    return df


def _parse_trueup_json(data) -> list:
    """Recursively search for layoff record arrays in JSON data."""
    records = []

    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict) and any(
            k in first for k in ["company", "name", "laid_off", "laidOff", "headcount"]
        ):
            return data

    if isinstance(data, dict):
        for key in ["data", "layoffs", "records", "items", "results", "companies"]:
            if key in data:
                sub = data[key]
                if isinstance(sub, list):
                    nested = _parse_trueup_json(sub)
                    if nested:
                        return nested
                elif isinstance(sub, dict):
                    nested = _parse_trueup_json(sub)
                    if nested:
                        return nested

    return records


def _dom_scrape_trueup(page) -> list:
    """Fallback DOM scraping for trueup.io."""
    records = []
    try:
        # Look for table or card-based layout
        page.wait_for_selector("table, [class*='table'], [class*='layoff'], [class*='company']", timeout=10000)
        time.sleep(1)

        # Try table structure
        headers = page.eval_on_selector_all(
            "table thead th, table th",
            "els => els.map(el => el.innerText.trim())"
        )
        rows_data = page.eval_on_selector_all(
            "table tbody tr",
            """rows => rows.map(row => {
                const cells = row.querySelectorAll('td');
                return Array.from(cells).map(td => td.innerText.trim());
            })"""
        )

        if headers and rows_data:
            for row in rows_data:
                if len(row) >= 2:  # At least company and count
                    rec = {}
                    for i, h in enumerate(headers):
                        if i < len(row):
                            rec[h] = row[i]
                    records.append(rec)
        else:
            # Try to extract any text that looks like layoff data
            text_content = page.inner_text("body")
            print(f"Page text preview: {text_content[:500]}")

    except Exception as e:
        print(f"Trueup DOM scrape error: {e}")

    return records


if __name__ == "__main__":
    df = scrape_trueup()
    if not df.empty:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to {OUTPUT_PATH}")
        print(df.head())
    else:
        print("No data collected.")
