"""Rule-based trading agent — parses market text and applies simple RSI/trend rules.

Strategy:
    - BUY when RSI < 35 and trend is not bearish (mean-reversion entry)
    - SELL when RSI > 65 or position P&L > +3% (take profit)
    - SELL when position P&L < -3% (stop loss)
    - HOLD otherwise

This is the "human expert" baseline. It reads the same text an LLM would
and applies straightforward technical analysis rules.
"""

from __future__ import annotations

import re


def rule_based_agent(observation: str) -> str:
    """Parse observation text and return a trading action."""
    stocks = _parse_stocks(observation)
    positions = _parse_positions(observation)
    is_single_stock = len(stocks) <= 1

    # Check exits first (sell signals)
    for symbol, pnl in positions.items():
        if pnl > 3.0 or pnl < -3.0:
            return f"SELL {symbol}" if not is_single_stock else "SELL"
        if symbol in stocks and stocks[symbol]["rsi"] > 65:
            return f"SELL {symbol}" if not is_single_stock else "SELL"

    # Check entries (buy signals) — only if not already holding
    for symbol, data in stocks.items():
        if symbol in positions:
            continue
        if data["rsi"] < 35 and data["trend"] != "bearish":
            return f"BUY {symbol}" if not is_single_stock else "BUY"

    return "HOLD"


def _parse_stocks(observation: str) -> dict[str, dict]:
    """Extract per-stock RSI and trend from observation text.

    Expected format per stock:
        RELIANCE: Rs1,179 (+0.4% today)
          RSI: 34 (neutral) | MACD: bearish
          Trend: bearish | Bollinger: lower_band (oversold)
    """
    stocks = {}
    current_symbol = None

    for line in observation.split("\n"):
        line = line.strip()

        # Match stock header: "SYMBOL: Rs1,234 (+1.2% today)"
        header = re.match(r"^([A-Z]{2,20}):\s+Rs", line)
        if header:
            current_symbol = header.group(1)
            stocks[current_symbol] = {"rsi": 50, "trend": "sideways"}
            continue

        if current_symbol is None:
            continue

        # Parse RSI: "RSI: 34 (neutral)"
        rsi_match = re.search(r"RSI:\s*(\d+)", line)
        if rsi_match:
            stocks[current_symbol]["rsi"] = int(rsi_match.group(1))

        # Parse Trend: "Trend: bearish"
        trend_match = re.search(r"Trend:\s*(\w+)", line)
        if trend_match:
            stocks[current_symbol]["trend"] = trend_match.group(1)

    return stocks


def _parse_positions(observation: str) -> dict[str, float]:
    """Extract held positions and their P&L from observation text.

    Expected format:
        Your Positions:
          RELIANCE: 84 shares @ Rs1,179 | Current: Rs1,200 | P&L: +1.8%
    """
    positions = {}
    in_positions = False

    for line in observation.split("\n"):
        line = line.strip()

        if "Your Positions:" in line:
            in_positions = True
            continue

        if in_positions and line.startswith("Constraints:"):
            break

        if in_positions:
            # Match: "SYMBOL: N shares ... P&L: +1.8%"
            match = re.match(r"([A-Z]{2,20}):\s+\d+\s+shares.*P&L:\s*([+-]?\d+\.?\d*)%", line)
            if match:
                positions[match.group(1)] = float(match.group(2))

    return positions
