"""
Scraper for layoffs.fyi using Playwright.
layoffs.fyi is powered by Airtable and renders data client-side.
Strategy: intercept the Airtable API network response directly.
"""

import json
import time
import random
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "layoffs_fyi_raw.csv"


AIRTABLE_EMBED_URL = "https://airtable.com/embed/app1PaujS9zxVGUZ4/shroKsHx3SdYYOzeh?backgroundColor=green&viewControls=on"


def scrape_layoffs_fyi() -> pd.DataFrame:
    """
    Scrape layoffs.fyi by loading the Airtable embed directly and doing
    virtual-scroll DOM extraction (Airtable uses virtualized list rendering).
    """
    records = []

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

        print("Loading Airtable embed (layoffs.fyi) ...")
        try:
            page.goto(AIRTABLE_EMBED_URL, wait_until="domcontentloaded", timeout=45000)
        except Exception as e:
            print(f"  Navigation warning (continuing): {e}")

        # Wait for Airtable JS to fully initialize
        time.sleep(8)

        # Virtual-scroll DOM scrape
        records = _dom_scrape(page)
        browser.close()

    if not records:
        print("WARNING: No records collected from layoffs.fyi")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["source"] = "layoffs.fyi"
    df["source_url"] = "https://layoffs.fyi/"

    # Normalize column names
    col_map = {
        "Company": "company_name",
        "company": "company_name",
        "# Laid Off": "layoff_count",
        "Laid Off": "layoff_count",
        "Date": "announcement_date",
        "date": "announcement_date",
        "% Laid Off": "layoff_pct",
        "Industry": "industry",
        "Stage": "stage",
        "Country": "country",
        "Funds Raised (MM)": "funds_raised_mm",
        "Source": "news_source_url",
        "List of Employees Laid Off": "affected_teams",
        "Location": "location_hq",
        "Headquarters Location": "location_hq",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Keep only relevant columns that exist
    keep = [
        "company_name", "announcement_date", "layoff_count", "layoff_pct",
        "industry", "stage", "country", "location_hq", "funds_raised_mm",
        "affected_teams", "news_source_url", "source", "source_url"
    ]
    df = df[[c for c in keep if c in df.columns]]

    print(f"Collected {len(df)} records from layoffs.fyi")
    return df


def _extract_airtable_records(data: dict) -> list:
    """Parse Airtable JSON response format."""
    records = []

    def flatten(r):
        if isinstance(r, dict) and "fields" in r:
            return r["fields"]
        return r

    if isinstance(data, dict):
        if "records" in data:
            records = [flatten(r) for r in data["records"]]
        elif "data" in data and isinstance(data["data"], dict):
            if "records" in data["data"]:
                records = [flatten(r) for r in data["data"]["records"]]
        elif "rows" in data:
            records = data["rows"]

    return records


def _dom_scrape(page) -> list:
    """
    Virtual scroll scraper for Airtable embed.
    Airtable uses virtualized lists — only ~20 rows visible at a time.
    We scroll through the table systematically to capture all records.
    """
    import re

    EMBED_URL = "https://airtable.com/embed/app1PaujS9zxVGUZ4/shroKsHx3SdYYOzeh?backgroundColor=green&viewControls=on"

    # Get column headers from aria-label attributes
    headers = page.eval_on_selector_all(
        '[aria-label^="Open "][aria-label$=" column menu"]',
        """els => els.map(el => {
            const label = el.getAttribute('aria-label');
            return label.replace('Open ', '').replace(' column menu', '');
        })"""
    )
    if not headers:
        # Fallback: hardcoded headers from inspection
        headers = ["Company", "Location HQ", "# Laid Off", "Date", "%", "Industry", "Source", "Stage", "$ Raised (mm)", "Country", "Date Added"]
    print(f"  Columns: {headers}")

    all_records = {}  # Use dict keyed by row number to avoid duplicates

    # Find the scrollable container
    scroll_container = page.query_selector('[class*="gridView"], [class*="scrollContainer"], .baymax')

    def extract_visible_rows():
        """Extract currently visible rows from the DOM."""
        rows_data = page.eval_on_selector_all(
            '[data-testid="data-row"]',
            """rows => rows.map(row => {
                // Get row number
                const rowNum = row.querySelector('[class*="rowNumberLabel"]');
                const num = rowNum ? rowNum.innerText.trim() : '';

                // Get all cell values
                const cells = row.querySelectorAll('[class*="cellContainer"], [data-columntype]');
                const values = Array.from(cells).map(c => c.innerText.trim());

                return {num: num, values: values, text: row.innerText.trim()};
            })"""
        )
        return rows_data

    # Initial extraction
    visible = extract_visible_rows()
    print(f"  Initial visible rows: {len(visible)}")

    # Scroll through the entire table
    last_max_row = 0
    scroll_attempts = 0
    max_scroll_attempts = 600  # Safety limit (~12,000 rows)
    consecutive_empty = 0
    MAX_CONSECUTIVE_EMPTY = 15  # Stop after 15 scrolls with no new rows

    # Find the scrollable grid element
    grid_js = """
        () => {
            // Find the scrollable element in Airtable
            const candidates = [
                document.querySelector('[class*="GridView"] [class*="scrollContainer"]'),
                document.querySelector('[class*="scrollContainer"]'),
                document.querySelector('.baymax'),
                document.querySelector('[class*="grid"]'),
                document.body
            ];
            for (const el of candidates) {
                if (el && el.scrollHeight > el.clientHeight + 100) {
                    return {scrollHeight: el.scrollHeight, clientHeight: el.clientHeight, scrollTop: el.scrollTop};
                }
            }
            return {scrollHeight: document.body.scrollHeight, clientHeight: window.innerHeight, scrollTop: window.scrollY};
        }
    """

    scroll_js = """
        (amount) => {
            const candidates = [
                document.querySelector('[class*="GridView"] [class*="scrollContainer"]'),
                document.querySelector('[class*="scrollContainer"]'),
            ];
            for (const el of candidates) {
                if (el && el.scrollHeight > el.clientHeight + 100) {
                    el.scrollTop += amount;
                    return el.scrollTop;
                }
            }
            window.scrollBy(0, amount);
            return window.scrollY;
        }
    """

    while scroll_attempts < max_scroll_attempts:
        rows_data = extract_visible_rows()

        new_found = 0
        for row in rows_data:
            row_num = row.get("num", "").replace(",", "")
            if row_num and row_num.isdigit():
                row_int = int(row_num)
                if row_int not in all_records:
                    values = row.get("values", [])
                    if values:
                        rec = {}
                        for i, h in enumerate(headers):
                            rec[h] = values[i] if i < len(values) else ""
                        all_records[row_int] = rec
                        new_found += 1
                        last_max_row = max(last_max_row, row_int)

        if new_found > 0 and scroll_attempts % 20 == 0:
            print(f"  Scroll {scroll_attempts}: {len(all_records)} total records, last row #{last_max_row}")

        # Scroll down by ~300px
        page.evaluate(scroll_js, 300)
        time.sleep(0.3)
        scroll_attempts += 1

        if new_found == 0:
            consecutive_empty += 1
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                print(f"  No new rows after {consecutive_empty} consecutive scrolls. Stopping.")
                break
        else:
            consecutive_empty = 0

    records = [v for _, v in sorted(all_records.items())]
    print(f"  Virtual scroll complete: {len(records)} total rows extracted")
    return records


if __name__ == "__main__":
    df = scrape_layoffs_fyi()
    if not df.empty:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to {OUTPUT_PATH}")
        print(df.head())
    else:
        print("No data collected.")
