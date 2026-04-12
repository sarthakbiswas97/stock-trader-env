"""
Market simulator — replays historical OHLCV data for the trading environment.
Loads real price data from Indian equity markets (NIFTY stocks).
"""

import random
from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data" / "ohlcv"

# Stock universe for each task difficulty
TASK_STOCKS = {
    "single_stock": ["RELIANCE"],
    "portfolio": [
        "RELIANCE", "INFY", "TCS", "HDFCBANK", "SBIN",
        "ICICIBANK", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
    ],
    "full_autonomous": [
        "RELIANCE", "INFY", "TCS", "HDFCBANK", "SBIN",
        "ICICIBANK", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
        "AXISBANK", "BAJFINANCE", "SUNPHARMA", "TITAN", "HINDUNILVR",
        "HCLTECH", "WIPRO", "NTPC", "POWERGRID", "ADANIENT",
        "TATASTEEL", "JSWSTEEL", "COALINDIA", "ONGC", "MARUTI",
    ],
}

TASK_EPISODE_DAYS = {
    "single_stock": 20,
    "portfolio": 30,
    "full_autonomous": 40,
}

TASK_INITIAL_CAPITAL = {
    "single_stock": 100_000,
    "portfolio": 200_000,
    "full_autonomous": 500_000,
}


def _load_stock_data(symbol: str) -> pd.DataFrame:
    """Load daily OHLCV data for a symbol."""
    path = DATA_DIR / f"{symbol}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


class MarketSimulator:
    """Replays historical market data for a set of stocks."""

    def __init__(self, task_id: str, seed: int | None = None):
        self.task_id = task_id
        self.symbols = TASK_STOCKS[task_id]
        self.episode_days = TASK_EPISODE_DAYS[task_id]
        self.initial_capital = TASK_INITIAL_CAPITAL[task_id]
        self.rng = random.Random(seed)

        # Load all stock data
        self._all_data: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            self._all_data[sym] = _load_stock_data(sym)

        # Episode state
        self._start_idx: dict[str, int] = {}
        self._current_day: int = 0

    def reset(self) -> None:
        """Pick a random start date window for the episode."""
        self._current_day = 0

        # Find a date range where all stocks have data
        # Use the stock with least data to constrain
        min_rows = min(len(df) for df in self._all_data.values())
        # Need enough lookback for indicators (50 days) + episode length
        lookback = 50
        max_start = min_rows - self.episode_days - lookback
        if max_start < 1:
            max_start = 1

        start = self.rng.randint(0, max_start)

        for sym in self.symbols:
            self._start_idx[sym] = start

    @property
    def current_day(self) -> int:
        return self._current_day

    @property
    def is_done(self) -> bool:
        return self._current_day >= self.episode_days

    def advance_day(self) -> None:
        self._current_day += 1

    def get_price(self, symbol: str) -> float:
        """Get current close price for a symbol."""
        idx = self._start_idx[symbol] + 50 + self._current_day
        df = self._all_data[symbol]
        if idx >= len(df):
            idx = len(df) - 1
        return float(df.iloc[idx]["close"])

    def get_daily_change(self, symbol: str) -> float:
        """Get today's return in percent."""
        idx = self._start_idx[symbol] + 50 + self._current_day
        df = self._all_data[symbol]
        if idx < 1 or idx >= len(df):
            return 0.0
        today = df.iloc[idx]["close"]
        yesterday = df.iloc[idx - 1]["close"]
        return (today - yesterday) / yesterday * 100

    def get_lookback_data(self, symbol: str, lookback: int = 50) -> pd.DataFrame:
        """Get lookback window of OHLCV data ending at current day (for computing indicators)."""
        end_idx = self._start_idx[symbol] + 50 + self._current_day
        start_idx = max(0, end_idx - lookback)
        df = self._all_data[symbol]
        if end_idx >= len(df):
            end_idx = len(df) - 1
        return df.iloc[start_idx:end_idx + 1].copy()

    def get_market_breadth(self) -> dict:
        """Compute market-wide metrics (for regime gate)."""
        up = 0
        down = 0
        total_change = 0.0
        for sym in self.symbols:
            change = self.get_daily_change(sym)
            total_change += change
            if change > 0:
                up += 1
            elif change < 0:
                down += 1

        n = len(self.symbols)
        avg_change = total_change / n if n > 0 else 0.0
        breadth_pct = down / n * 100 if n > 0 else 0.0

        return {
            "avg_change": round(avg_change, 2),
            "advancing": up,
            "declining": down,
            "breadth_weak": breadth_pct > 70,
            "market_down": avg_change < -0.5,
        }

    def get_5day_trend(self, symbol: str) -> float:
        """Get 5-day cumulative return for a symbol."""
        idx = self._start_idx[symbol] + 50 + self._current_day
        df = self._all_data[symbol]
        if idx < 5 or idx >= len(df):
            return 0.0
        today = df.iloc[idx]["close"]
        five_ago = df.iloc[idx - 5]["close"]
        return (today - five_ago) / five_ago * 100
