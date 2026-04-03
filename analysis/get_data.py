"""
Phase 3: Download stock returns and Fama-French 4-factor data.

For each event in master_events_enriched.csv:
  - Pull daily adjusted returns via yfinance
  - Window: event_date - 300 days to event_date + 100 days
    (300 days covers the [-260, -11] estimation window + buffer)

Factor data:
  - F-F 3 factors (daily) + Momentum factor → FF4
  - Source: Ken French Data Library via pandas_datareader
"""

import time
import warnings
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from pathlib import Path

warnings.filterwarnings("ignore")

EVENTS_PATH  = Path("data/processed/master_events_enriched.csv")
RETURNS_DIR  = Path("data/processed/stock_returns")
FACTORS_PATH = Path("data/processed/ff_factors.csv")
RETURNS_DIR.mkdir(parents=True, exist_ok=True)

# Days before/after event to download
PRE_DAYS  = 310   # covers estimation window [-260,-11] + gap
POST_DAYS = 100   # covers event windows up to [0,+60] + buffer


# ── 1. Download Fama-French 4-Factor data ────────────────────────────────────
def download_ff4() -> pd.DataFrame:
    """Download FF3 + Momentum from Ken French Data Library."""
    if FACTORS_PATH.exists():
        print(f"FF4 factors already exist at {FACTORS_PATH}, loading...")
        return pd.read_csv(FACTORS_PATH, index_col=0, parse_dates=True)

    print("Downloading Fama-French 3-factor (daily)...")
    ff3 = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                          start="2018-01-01")[0] / 100
    ff3.columns = ["MKT_RF", "SMB", "HML", "RF"]

    print("Downloading Momentum factor (daily)...")
    mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench",
                          start="2018-01-01")[0] / 100
    mom.columns = ["MOM"]

    ff4 = ff3.join(mom, how="inner")
    ff4.index = pd.to_datetime(ff4.index, format="%Y%m%d")
    ff4.to_csv(FACTORS_PATH)
    print(f"FF4 saved → {FACTORS_PATH}  ({len(ff4)} daily obs)")
    return ff4


# ── 2. Download stock returns ────────────────────────────────────────────────
def download_stock_returns(events: pd.DataFrame, overwrite: bool = False) -> dict:
    """
    Download adjusted close prices for each unique ticker.
    Returns dict: {ticker: pd.Series of daily returns}
    """
    # Get unique tickers with at least one READY event
    ready = events[events["status"] == "READY"].copy()
    tickers = ready["ticker"].dropna().unique().tolist()
    tickers = [t for t in tickers if str(t) != "nan" and str(t) != ""]

    print(f"Downloading returns for {len(tickers)} unique tickers...")

    results = {}
    failed = []

    # Batch download — yfinance handles multiple tickers efficiently
    # Process in batches of 50 to avoid timeouts
    batch_size = 50
    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start: batch_start + batch_size]

        # Date range covering all events for this batch
        # Use a wide window covering all events (2018 → 2026)
        try:
            raw = yf.download(
                batch,
                start="2018-01-01",
                end="2026-12-31",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            print(f"  Batch {batch_start//batch_size + 1} download error: {e}")
            failed.extend(batch)
            continue

        # Extract Adj Close (or Close if auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["Adj Close"]
        else:
            close = raw[["Close"]] if "Close" in raw.columns else raw[["Adj Close"]]
            close.columns = batch[:1]

        # Compute daily returns per ticker
        for tk in batch:
            if tk not in close.columns:
                failed.append(tk)
                continue
            prices = close[tk].dropna()
            if len(prices) < 100:
                failed.append(tk)
                continue
            returns = prices.pct_change().dropna()
            returns.name = tk

            # Save individual file
            out_path = RETURNS_DIR / f"{tk.replace('/', '_')}.csv"
            returns.to_csv(out_path, header=True)
            results[tk] = returns

        print(f"  Batch {batch_start//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: "
              f"{len([t for t in batch if t in results])}/{len(batch)} succeeded")
        time.sleep(0.5)

    print(f"\nDownload complete: {len(results)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed tickers: {failed[:20]}{'...' if len(failed)>20 else ''}")

    # Save failed list
    pd.DataFrame({"ticker": failed}).to_csv("data/processed/failed_tickers.csv", index=False)
    return results


# ── 3. Validate coverage ─────────────────────────────────────────────────────
def validate_coverage(events: pd.DataFrame, returns: dict, ff4: pd.DataFrame) -> pd.DataFrame:
    """
    For each READY event, check whether:
      - Stock return data covers the estimation window [-260, -11]
      - Stock return data covers the event window [0, +60]
      - FF4 factor data is available for those dates
    Tag each event as 'usable' or flag the reason for exclusion.
    """
    ready = events[events["status"] == "READY"].copy()
    ready["event_study_usable"] = False
    ready["exclusion_reason"]   = ""

    for idx, row in ready.iterrows():
        tk   = str(row["ticker"])
        date = pd.to_datetime(row["announcement_date"])

        if pd.isna(date):
            ready.at[idx, "exclusion_reason"] = "missing_date"
            continue
        if tk not in returns:
            ready.at[idx, "exclusion_reason"] = "no_return_data"
            continue

        ret = returns[tk]
        trading_days = ret.index

        # Find position of event date (nearest trading day)
        if date not in trading_days:
            nearest = trading_days[trading_days.get_indexer([date], method="nearest")[0]]
            if abs((nearest - date).days) > 5:
                ready.at[idx, "exclusion_reason"] = "event_date_too_far_from_trading"
                continue
            t0 = nearest
        else:
            t0 = date

        t0_pos = trading_days.get_loc(t0)

        # Check estimation window: need at least 100 obs in [-260, -11]
        est_start_pos = t0_pos - 260
        est_end_pos   = t0_pos - 11
        if est_start_pos < 0:
            ready.at[idx, "exclusion_reason"] = "insufficient_pre_event_history"
            continue
        est_obs = len(trading_days[est_start_pos: est_end_pos])
        if est_obs < 100:
            ready.at[idx, "exclusion_reason"] = f"too_few_estimation_obs_{est_obs}"
            continue

        # Check event window: need data through +60
        evt_end_pos = t0_pos + 60
        if evt_end_pos >= len(trading_days):
            ready.at[idx, "exclusion_reason"] = "insufficient_post_event_data"
            continue

        # Check FF4 coverage
        est_dates = trading_days[est_start_pos: evt_end_pos]
        ff4_coverage = est_dates.isin(ff4.index).mean()
        if ff4_coverage < 0.8:
            ready.at[idx, "exclusion_reason"] = f"low_ff4_coverage_{ff4_coverage:.2f}"
            continue

        ready.at[idx, "event_study_usable"] = True

    usable = ready["event_study_usable"].sum()
    print(f"\nEvent study coverage:")
    print(f"  Usable events : {usable} / {len(ready)}")
    print(f"  Exclusion reasons:")
    excl = ready[~ready["event_study_usable"]]["exclusion_reason"].value_counts()
    for reason, count in excl.items():
        print(f"    {reason}: {count}")

    return ready


# ── Main ─────────────────────────────────────────────────────────────────────
def run():
    events = pd.read_csv(EVENTS_PATH)
    events["announcement_date"] = pd.to_datetime(events["announcement_date"], errors="coerce")
    print(f"Loaded {len(events)} events ({events['status'].eq('READY').sum()} READY)")

    # Step 1: FF4 factors
    print("\n── Step 1: Fama-French 4 factors ──")
    ff4 = download_ff4()
    print(f"FF4 shape: {ff4.shape}, date range: {ff4.index.min().date()} → {ff4.index.max().date()}")

    # Step 2: Stock returns
    print("\n── Step 2: Stock returns ──")
    returns = download_stock_returns(events)

    # Step 3: Validate coverage
    print("\n── Step 3: Validate event study coverage ──")
    ready_validated = validate_coverage(events, returns, ff4)

    # Merge usability flags back into main events
    events = events.merge(
        ready_validated[["company_fyi", "ticker", "announcement_date",
                          "event_study_usable", "exclusion_reason"]],
        on=["company_fyi", "ticker", "announcement_date"],
        how="left"
    )
    events["event_study_usable"] = events["event_study_usable"].fillna(False)

    events.to_csv("data/processed/master_events_enriched.csv", index=False)
    print(f"\nUpdated master_events_enriched.csv with usability flags")
    print(f"Final usable sample: {events['event_study_usable'].sum()} events")

    # Period breakdown of usable events
    usable = events[events["event_study_usable"]]
    print(f"  Post-GenAI: {(usable['period']=='post_genai').sum()}")
    print(f"  Pre-GenAI : {(usable['period']=='pre_genai').sum()}")
    print(f"  AI=1      : {usable['ai_mentioned'].eq(1).sum()}")


if __name__ == "__main__":
    run()
