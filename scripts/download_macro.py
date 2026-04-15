"""
Download macro/market-wide data using yfinance.

Downloads India VIX, USD/INR, Brent Crude, and sector indices
for use as macro context in trading observations.

Usage:
    pip install yfinance
    python scripts/download_macro.py
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MACRO_DIR = Path(__file__).parent.parent / "data" / "macro"

# Macro instruments: name -> yfinance ticker
MACRO_TICKERS = {
    "INDIA_VIX": "^INDIAVIX",
    "USDINR": "USDINR=X",
    "BRENT_CRUDE": "BZ=F",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_PHARMA": "^CNXPHARMA",
}

START_DATE = "2020-01-01"


def download_macro_data() -> None:
    """Download all macro instruments and save as CSVs."""
    MACRO_DIR.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().strftime("%Y-%m-%d")
    total = len(MACRO_TICKERS)
    downloaded = 0
    failed = 0

    logger.info("Downloading macro data for %d instruments", total)
    logger.info("Date range: %s to %s", START_DATE, end_date)
    logger.info("Output: %s", MACRO_DIR)
    logger.info("=" * 60)

    for name, ticker in MACRO_TICKERS.items():
        try:
            logger.info("  [%d/%d] %s (%s)...", downloaded + failed + 1, total, name, ticker)

            df = yf.download(
                ticker,
                start=START_DATE,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                logger.warning("  %s: no data returned", name)
                failed += 1
                continue

            # Flatten multi-level columns if present (yfinance sometimes returns these)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standardize to match stock CSV format: timestamp,open,high,low,close,volume
            df = df.reset_index()
            df = df.rename(columns={"Date": "timestamp"})

            # Ensure we have the expected columns
            expected_cols = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
            available = [c for c in expected_cols if c in df.columns]
            df = df[available]
            df.columns = [c.lower() for c in df.columns]

            # Sort by date
            df = df.sort_values("timestamp").reset_index(drop=True)

            filepath = MACRO_DIR / f"{name}_daily.csv"
            df.to_csv(filepath, index=False)
            downloaded += 1
            logger.info("  %s: %d rows saved", name, len(df))

            time.sleep(1)  # Rate limit courtesy

        except Exception as e:
            logger.error("  %s: error - %s", name, e)
            failed += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("Downloaded: %d", downloaded)
    logger.info("Failed: %d", failed)
    logger.info("Total CSVs: %d", len(list(MACRO_DIR.glob("*_daily.csv"))))


if __name__ == "__main__":
    download_macro_data()
