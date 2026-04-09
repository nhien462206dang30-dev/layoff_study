"""
Combine all raw scraped data sources into a single Master Event List.
Steps:
  1. Load all raw CSVs
  2. Standardize columns and date formats
  3. Deduplicate (same company within 180 days → keep first)
  4. Use Claude API to:
     a. Normalize company names (Meta vs Meta Platforms vs Facebook)
     b. Label ai_mentioned (0/1) from source text if not already labeled
  5. Save master_events.csv
"""

import os
import re
import json
import time
import anthropic
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "master_events.csv"

REQUIRED_COLS = [
    "company_name", "announcement_date", "layoff_count", "layoff_pct",
    "affected_teams", "ai_mentioned", "ai_quote",
    "news_source_url", "source", "source_url",
    "industry", "country", "location_hq",
]


def load_all_raw() -> pd.DataFrame:
    """Load and concatenate all raw CSVs from the raw data directory."""
    dfs = []

    raw_files = {
        "layoffs_fyi_raw.csv": "layoffs.fyi",
        "trueup_raw.csv": "trueup.io",
        "techcrunch_raw.csv": "techcrunch",
        "edgar_8k_raw.csv": "edgar_8k",
    }

    for fname, source_label in raw_files.items():
        fpath = RAW_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath, low_memory=False)
            if "source" not in df.columns:
                df["source"] = source_label
            print(f"Loaded {len(df)} rows from {fname}")
            dfs.append(df)
        else:
            print(f"File not found (skipping): {fpath}")

    # Also check for any manually added CSVs
    for fpath in RAW_DIR.glob("*.csv"):
        if fpath.name not in raw_files:
            df = pd.read_csv(fpath, low_memory=False)
            df["source"] = fpath.stem
            print(f"Loaded {len(df)} rows from {fpath.name} (extra source)")
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No raw data found in {RAW_DIR}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal raw records: {len(combined)}")
    return combined


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns, types, and ensure required fields exist."""

    # Ensure all required columns exist (fill with None if missing)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = None

    # Normalize company names: strip whitespace, title case
    df["company_name"] = df["company_name"].astype(str).str.strip()
    df["company_name"] = df["company_name"].replace("nan", pd.NA)
    df = df.dropna(subset=["company_name"])

    # Parse dates
    df["announcement_date"] = pd.to_datetime(
        df["announcement_date"], errors="coerce", infer_datetime_format=True
    )

    # For EDGAR, use filing_date as announcement_date
    if "filing_date" in df.columns:
        mask = df["announcement_date"].isna() & df["filing_date"].notna()
        df.loc[mask, "announcement_date"] = pd.to_datetime(
            df.loc[mask, "filing_date"], errors="coerce"
        )

    # Clean layoff_count: remove commas, convert to int
    if "layoff_count" in df.columns:
        df["layoff_count"] = (
            df["layoff_count"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(\d+)", expand=False)
        )
        df["layoff_count"] = pd.to_numeric(df["layoff_count"], errors="coerce")

    # Clean layoff_pct: strip %, convert to float
    if "layoff_pct" in df.columns:
        df["layoff_pct"] = (
            df["layoff_pct"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["layoff_pct"] = pd.to_numeric(df["layoff_pct"], errors="coerce")

    # ai_mentioned: ensure numeric
    if "ai_mentioned" in df.columns:
        df["ai_mentioned"] = pd.to_numeric(df["ai_mentioned"], errors="coerce").fillna(pd.NA)

    # Add period label
    df["period"] = df["announcement_date"].apply(
        lambda d: "post_genai" if pd.notna(d) and d.year >= 2023 else "pre_genai"
    )

    # Add data_sources column (track which sources reported each company)
    df["data_sources"] = df["source"]

    return df


def normalize_company_names_with_claude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use Claude to identify and merge duplicate company names.
    E.g. 'Meta', 'Meta Platforms', 'Meta Platforms Inc' → 'Meta Platforms'
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY not set — skipping Claude name normalization.")
        return df

    client = anthropic.Anthropic(api_key=api_key)

    unique_names = df["company_name"].dropna().unique().tolist()
    print(f"\nNormalizing {len(unique_names)} unique company names with Claude...")

    # Process in batches of 100
    batch_size = 100
    name_map = {}

    for i in range(0, len(unique_names), batch_size):
        batch = unique_names[i : i + batch_size]
        prompt = f"""Below is a list of company names from a tech layoffs dataset.
Many may be duplicates with slight variations (e.g., "Meta" vs "Meta Platforms", "Google" vs "Alphabet Inc").

For each name, return the canonical/official company name that should be used.

Input names:
{json.dumps(batch, indent=2)}

Return a JSON object mapping each input name to its canonical name.
Use the most common/official form (typically the legal registered name or most widely recognized name).
If a name is already canonical, map it to itself.
Return ONLY valid JSON, no other text.

Example format:
{{"Meta": "Meta Platforms", "Meta Platforms Inc": "Meta Platforms", "Google": "Alphabet", ...}}"""

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                batch_map = json.loads(json_match.group())
                name_map.update(batch_map)
        except Exception as e:
            print(f"  Name normalization error (batch {i//batch_size}): {e}")

        time.sleep(1)

    # Apply mapping
    df["company_name_original"] = df["company_name"]
    df["company_name"] = df["company_name"].map(lambda x: name_map.get(x, x))

    changed = (df["company_name"] != df["company_name_original"]).sum()
    print(f"  Normalized {changed} company name variants")
    return df


def label_ai_mentions_with_claude(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where ai_mentioned is null and news_source_url is available,
    fetch the article and label ai_mentioned using Claude.
    Only processes rows without existing labels.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY not set — skipping AI mention labeling.")
        return df

    client = anthropic.Anthropic(api_key=api_key)

    # Find rows needing labeling that have company name + date info
    mask = (
        df["ai_mentioned"].isna()
        & df["company_name"].notna()
        & df["announcement_date"].notna()
    )
    to_label = df[mask].copy()
    print(f"\nLabeling ai_mentioned for {len(to_label)} rows...")

    # Batch label using company name + any available text
    batch_size = 50
    results = {}

    for i in range(0, len(to_label), batch_size):
        batch = to_label.iloc[i : i + batch_size]

        companies_info = []
        for _, row in batch.iterrows():
            info = {
                "idx": int(row.name),
                "company": row["company_name"],
                "date": str(row["announcement_date"])[:10] if pd.notna(row["announcement_date"]) else "unknown",
                "affected_teams": str(row.get("affected_teams", "")) if pd.notna(row.get("affected_teams")) else "",
            }
            companies_info.append(info)

        prompt = f"""For each tech company layoff event below, determine if AI/automation/technology efficiency
was mentioned as a reason for the layoffs, based on what you know about these companies' public announcements.

Companies:
{json.dumps(companies_info, indent=2)}

Return a JSON object mapping each idx to:
{{
  "ai_mentioned": 0 or 1,
  "confidence": "high"/"medium"/"low",
  "ai_quote": "relevant quote if known, else null"
}}

Return ONLY valid JSON like: {{"123": {{"ai_mentioned": 1, "confidence": "high", "ai_quote": "..."}}, ...}}"""

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                batch_results = json.loads(json_match.group())
                results.update(batch_results)
        except Exception as e:
            print(f"  AI labeling error (batch {i//batch_size}): {e}")

        time.sleep(1)

    # Apply labels back to DataFrame
    for idx_str, label_data in results.items():
        try:
            idx = int(idx_str)
            if idx in df.index:
                df.at[idx, "ai_mentioned"] = label_data.get("ai_mentioned", pd.NA)
                if label_data.get("ai_quote") and pd.isna(df.at[idx, "ai_quote"]):
                    df.at[idx, "ai_quote"] = label_data.get("ai_quote")
        except (ValueError, KeyError):
            continue

    labeled = (df["ai_mentioned"].notna()).sum()
    print(f"  ai_mentioned labeled for {labeled} total rows")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge records for the same company from different sources.
    Within 180-day windows, keep only the first (earliest) event per company.
    Track which sources reported each event.
    """
    df = df.sort_values(["company_name", "announcement_date"]).reset_index(drop=True)

    # Group by company, then collapse events within 180-day windows
    keep_mask = pd.Series(True, index=df.index)
    last_date = {}

    for idx, row in df.iterrows():
        company = row["company_name"]
        date = row["announcement_date"]

        if pd.isna(date):
            continue

        if company in last_date:
            days_diff = (date - last_date[company]).days
            if days_diff <= 180:
                # Merge: mark this as duplicate, but capture source info
                # Find the kept row for this company
                prev_idx = df[
                    (df["company_name"] == company)
                    & (df["announcement_date"] == last_date[company])
                ].index
                if len(prev_idx) > 0:
                    prev_i = prev_idx[0]
                    # Accumulate source info
                    existing_sources = str(df.at[prev_i, "data_sources"])
                    new_source = str(row["source"])
                    if new_source not in existing_sources:
                        df.at[prev_i, "data_sources"] = existing_sources + ";" + new_source
                    # Prefer non-null values from duplicate
                    for col in ["layoff_count", "layoff_pct", "affected_teams"]:
                        if pd.isna(df.at[prev_i, col]) and pd.notna(row.get(col)):
                            df.at[prev_i, col] = row[col]
                keep_mask[idx] = False
                continue

        last_date[company] = date

    df_deduped = df[keep_mask].reset_index(drop=True)
    print(f"\nAfter deduplication: {len(df_deduped)} events (from {len(df)} raw records)")
    return df_deduped


def run():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading raw data")
    print("=" * 60)
    df = load_all_raw()

    print("\n" + "=" * 60)
    print("STEP 2: Standardizing fields")
    print("=" * 60)
    df = standardize(df)

    print("\n" + "=" * 60)
    print("STEP 3: Normalizing company names (Claude API)")
    print("=" * 60)
    df = normalize_company_names_with_claude(df)

    print("\n" + "=" * 60)
    print("STEP 4: Deduplicating events")
    print("=" * 60)
    df = deduplicate(df)

    print("\n" + "=" * 60)
    print("STEP 5: Labeling AI mentions (Claude API)")
    print("=" * 60)
    df = label_ai_mentions_with_claude(df)

    # Final cleanup
    df = df.sort_values("announcement_date", ascending=False).reset_index(drop=True)

    # Summary stats
    print("\n" + "=" * 60)
    print("MASTER EVENT LIST SUMMARY")
    print("=" * 60)
    print(f"Total events: {len(df)}")
    print(f"Date range: {df['announcement_date'].min()} to {df['announcement_date'].max()}")
    print(f"Pre-GenAI events: {(df['period'] == 'pre_genai').sum()}")
    print(f"Post-GenAI events: {(df['period'] == 'post_genai').sum()}")
    print(f"AI mentioned: {df['ai_mentioned'].eq(1).sum()}")
    print(f"Companies with layoff_count: {df['layoff_count'].notna().sum()}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved master events to: {OUTPUT_PATH}")
    print(df[["company_name", "announcement_date", "layoff_count", "ai_mentioned", "source"]].head(20).to_string())


if __name__ == "__main__":
    run()
