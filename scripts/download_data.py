"""
Download historical daily OHLCV data for NIFTY stocks using Zerodha Kite Connect.

Usage:
    python scripts/download_data.py
"""

import time
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "ohlcv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load Kite session from trader project
TRADER_SESSION = Path(__file__).parent.parent.parent / "trader" / ".kite_session"
KITE_API_KEY = "es57jly5mimjlreh"

# NIFTY 50 stocks (core universe)
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "TITAN", "SUNPHARMA", "ULTRACEMCO", "NESTLEIND", "WIPRO",
    "HCLTECH", "TATAMOTORS", "POWERGRID", "NTPC", "TECHM",
    "M&M", "BAJAJFINSV", "ONGC", "ADANIENT", "ADANIPORTS",
    "COALINDIA", "JSWSTEEL", "TATASTEEL", "GRASIM", "INDUSINDBK",
    "BRITANNIA", "CIPLA", "DRREDDY", "DIVISLAB", "EICHERMOT",
    "HEROMOTOCO", "BPCL", "APOLLOHOSP", "SBILIFE", "TATACONSUM",
    "HINDALCO", "BAJAJ-AUTO", "SHRIRAMFIN", "TRENT",
]

# NIFTY 100 stocks beyond NIFTY 50 (full NIFTY 100 coverage)
NIFTY_100_EXTRA = [
    # Already downloaded (20)
    "HDFCLIFE", "DABUR", "PIDILITIND", "HAVELLS", "DLF",
    "GODREJCP", "COLPAL", "LUPIN", "AUROPHARMA", "BANKBARODA",
    "PNB", "IOC", "GAIL", "BEL", "IRCTC",
    "POLYCAB", "SRF", "PERSISTENT", "TATAPOWER", "NHPC",
    # New additions to reach NIFTY 100
    "ADANIGREEN", "AMBUJACEM", "ATGL", "BOSCHLTD", "CANBK",
    "CGPOWER", "CHOLAFIN", "ICICIGI", "ICICIPRULI", "INDHOTEL",
    "IRFC", "JINDALSTEL", "JSWENERGY", "LICI", "LTIM",
    "MANKIND", "MAXHEALTH", "MOTHERSON", "NMDC", "OBEROIRLTY",
    "PFC", "RECLTD", "SBICARD", "SHREECEM", "SIEMENS",
    "TATAELXSI", "TORNTPHARM", "VBL", "VEDL", "ZOMATO",
    "ZYDUSLIFE",
]

ALL_SYMBOLS = NIFTY_50 + NIFTY_100_EXTRA
RATE_LIMIT_DELAY = 0.35  # 3 req/sec limit


def main():
    from kiteconnect import KiteConnect

    # Load session
    with open(TRADER_SESSION) as f:
        session = json.load(f)

    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(session["access_token"])

    # Verify connection
    try:
        profile = kite.profile()
        print(f"Connected as: {profile['user_name']}")
    except Exception as e:
        print(f"Session expired. Re-authenticate in trader project first.")
        print(f"Error: {e}")
        return

    # Load instruments cache
    print("Loading instruments...")
    time.sleep(RATE_LIMIT_DELAY)
    instruments = kite.instruments(exchange="NSE")
    token_map = {inst["tradingsymbol"]: inst["instrument_token"] for inst in instruments}

    # Check which symbols we already have with enough data
    existing = {}
    for f in DATA_DIR.glob("*_daily.csv"):
        sym = f.stem.replace("_daily", "")
        rows = sum(1 for _ in open(f)) - 1  # minus header
        existing[sym] = rows

    # Download settings
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)  # 5 years

    total = len(ALL_SYMBOLS)
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"\nDownloading daily data for {total} stocks")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Existing with 1000+ rows: {sum(1 for r in existing.values() if r >= 1000)}")
    print("=" * 60)

    for i, symbol in enumerate(ALL_SYMBOLS, 1):
        # Skip if we already have enough data
        if symbol in existing and existing[symbol] >= 1000:
            print(f"  [{i}/{total}] {symbol}: already have {existing[symbol]} rows, skipping")
            skipped += 1
            continue

        token = token_map.get(symbol)
        if not token:
            print(f"  [{i}/{total}] {symbol}: instrument not found, skipping")
            failed += 1
            continue

        try:
            time.sleep(RATE_LIMIT_DELAY)
            data = kite.historical_data(
                instrument_token=token,
                from_date=start_date,
                to_date=end_date,
                interval="day",
            )

            if not data:
                print(f"  [{i}/{total}] {symbol}: no data returned")
                failed += 1
                continue

            df = pd.DataFrame(data)
            df = df.rename(columns={"date": "timestamp"})
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            filepath = DATA_DIR / f"{symbol}_daily.csv"
            df.to_csv(filepath, index=False)
            downloaded += 1
            print(f"  [{i}/{total}] {symbol}: {len(df)} rows saved")

        except Exception as e:
            print(f"  [{i}/{total}] {symbol}: error - {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (already had data): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total CSVs: {len(list(DATA_DIR.glob('*_daily.csv')))}")


if __name__ == "__main__":
    main()
