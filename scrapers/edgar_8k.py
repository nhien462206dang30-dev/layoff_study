"""
SEC EDGAR Full-Text Search API scraper for 8-K filings mentioning layoffs.
This gives the most authoritative announcement dates for large public companies.

EDGAR EFTS API: https://efts.sec.gov/LATEST/search-index?q=...&dateRange=custom&...
"""

import re
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "edgar_8k_raw.csv"

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
HEADERS = {
    "User-Agent": "Academic Research layoff-study@university.edu",  # EDGAR requires User-Agent
    "Accept-Encoding": "gzip, deflate",
}

LAYOFF_KEYWORDS = [
    "workforce reduction",
    "reduction in force",
    "headcount reduction",
    "layoffs",
    "restructuring plan",
    "job cuts",
    "employee separation",
]


def search_edgar_8k(
    start_date: str = "2019-01-01",
    end_date: str = "2025-12-31",
    max_results: int = 2000,
) -> pd.DataFrame:
    """
    Search EDGAR for 8-K filings mentioning layoff-related keywords.

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        max_results: max filings to collect

    Returns:
        DataFrame with filing metadata
    """
    all_filings = []

    for keyword in LAYOFF_KEYWORDS:
        print(f"Searching EDGAR for: '{keyword}' in 8-K filings ({start_date} to {end_date})")
        filings = _search_keyword(keyword, start_date, end_date, max_per_keyword=500)
        all_filings.extend(filings)
        time.sleep(1)  # Be polite to EDGAR

    if not all_filings:
        return pd.DataFrame()

    df = pd.DataFrame(all_filings)

    # Deduplicate by accession number
    if "accession_no" in df.columns:
        df = df.drop_duplicates(subset=["accession_no"])

    # Sort by filing date
    if "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df = df.sort_values("filing_date", ascending=False)

    df["source"] = "SEC EDGAR 8-K"
    df["source_url"] = df.get("filing_url", "")

    print(f"Total unique 8-K filings found: {len(df)}")
    return df


def _search_keyword(
    keyword: str,
    start_date: str,
    end_date: str,
    max_per_keyword: int = 500,
) -> list:
    """Query EDGAR EFTS search API for a single keyword."""
    filings = []
    page_size = 100
    from_idx = 0

    while from_idx < max_per_keyword:
        params = {
            "q": f'"{keyword}"',
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "forms": "8-K",
            "hits.hits.total.value": 1,
            "hits.hits._source.period_of_report": 1,
            "_source": "file_date,entity_name,file_num,period_of_report,form_type,accession_no",
            "hits.hits.highlight": 1,
            "from": from_idx,
            "size": page_size,
        }

        # Use the EFTS full-text search endpoint
        url = "https://efts.sec.gov/LATEST/search-index"
        # Use the simpler search endpoint instead
        search_url = "https://efts.sec.gov/LATEST/search-index"

        # Actually use the working EDGAR search API
        api_url = "https://efts.sec.gov/LATEST/search-index"

        # The correct EDGAR full text search API
        correct_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{keyword.replace(' ', '+')}%22&dateRange=custom&startdt={start_date}&enddt={end_date}&forms=8-K&from={from_idx}&size={page_size}"

        try:
            resp = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "q": f'"{keyword}"',
                    "dateRange": "custom",
                    "startdt": start_date,
                    "enddt": end_date,
                    "forms": "8-K",
                    "from": from_idx,
                    "size": page_size,
                },
                headers=HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  EDGAR search error: {e}")
            break

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            src = hit.get("_source", {})

            # Parse company name and ticker from display_names: "Company, Inc. (TICK) (CIK ...)"
            display_names = src.get("display_names", [""])
            display = display_names[0] if display_names else ""
            ticker_match = re.search(r"\(([A-Z]{1,5})\)", display)
            ticker = ticker_match.group(1) if ticker_match else ""
            company_name = re.sub(r"\s*\(.*\)", "", display).strip()

            # CIK
            ciks = src.get("ciks", [""])
            cik = ciks[0].lstrip("0") if ciks else ""

            # Accession number (called 'adsh' in EDGAR)
            adsh = src.get("adsh", "")

            filing = {
                "company_name": company_name,
                "ticker_hint": ticker,
                "filing_date": src.get("file_date", ""),
                "period_of_report": src.get("period_ending", ""),
                "form_type": src.get("form", "8-K"),
                "accession_no": adsh,
                "cik": cik,
                "location": src.get("biz_locations", [""])[0] if src.get("biz_locations") else "",
                "sic": src.get("sics", [""])[0] if src.get("sics") else "",
                "search_keyword": keyword,
                "filing_url": _build_filing_url(adsh, cik),
            }
            filings.append(filing)

        total = data.get("hits", {}).get("total", {}).get("value", 0)
        from_idx += page_size

        if from_idx >= min(total, max_per_keyword):
            break

        time.sleep(0.5)

    print(f"  Found {len(filings)} filings for '{keyword}'")
    return filings


def _build_filing_url(accession_no: str, cik: str = "") -> str:
    """Build a direct link to the EDGAR filing viewer."""
    if not accession_no:
        return ""
    clean = accession_no.replace("-", "")
    if not cik:
        cik = clean[:10].lstrip("0")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{clean}/"


def get_company_cik(company_name: str) -> str:
    """Look up a company's CIK number from EDGAR."""
    try:
        resp = requests.get(
            "https://www.sec.gov/cgi-bin/browse-edgar",
            params={
                "company": company_name,
                "CIK": "",
                "type": "8-K",
                "dateb": "",
                "owner": "include",
                "count": "10",
                "search_text": "",
                "action": "getcompany",
                "output": "atom",
            },
            headers=HEADERS,
            timeout=15,
        )
        # Parse atom feed for CIK
        if "<CIK>" in resp.text:
            cik = resp.text.split("<CIK>")[1].split("</CIK>")[0].strip()
            return cik
    except Exception:
        pass
    return ""


if __name__ == "__main__":
    df = search_edgar_8k(start_date="2019-01-01", end_date="2025-03-31")
    if not df.empty:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved {len(df)} records to {OUTPUT_PATH}")
        print(df[["company_name", "filing_date", "accession_no"]].head(10).to_string())
    else:
        print("No filings collected.")
