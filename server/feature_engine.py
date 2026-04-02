"""
Technical indicator engine — computes features from OHLCV data.
Ported from the production trading system's indicator library.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    """RSI (0-100). >70 overbought, <30 oversold."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 1) if not np.isnan(val) else 50.0


def compute_macd(close: pd.Series) -> dict:
    """MACD components: line, signal, histogram."""
    ema_fast = ema(close, 12)
    ema_slow = ema(close, 26)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, 9)
    histogram = macd_line - signal_line

    h = histogram.iloc[-1]
    h_prev = histogram.iloc[-2] if len(histogram) > 1 else 0

    if np.isnan(h):
        return {"signal": "neutral", "histogram": 0.0, "crossover": False}

    crossover = (h > 0 and h_prev <= 0) or (h < 0 and h_prev >= 0)
    signal = "bullish" if h > 0 else "bearish"
    return {"signal": signal, "histogram": round(float(h), 2), "crossover": crossover}


def compute_volume_spike(volume: pd.Series, period: int = 20) -> float:
    """Volume relative to 20-day average. >1.5 = high, <0.5 = low."""
    avg = volume.rolling(window=period).mean().iloc[-1]
    if np.isnan(avg) or avg == 0:
        return 1.0
    return round(float(volume.iloc[-1] / avg), 2)


def compute_trend(close: pd.Series) -> str:
    """Trend direction from EMA crossover."""
    if len(close) < 30:
        return "unknown"
    ema_fast = ema(close, 10)
    ema_slow = ema(close, 30)
    diff = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1]
    if diff > 0.001:
        return "bullish"
    elif diff < -0.001:
        return "bearish"
    return "sideways"


def compute_bollinger_position(close: pd.Series, period: int = 20) -> str:
    """Position within Bollinger Bands."""
    if len(close) < period:
        return "middle"
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + 2 * std
    lower = middle - 2 * std

    price = close.iloc[-1]
    u, l = upper.iloc[-1], lower.iloc[-1]
    if np.isnan(u) or np.isnan(l) or u == l:
        return "middle"

    pos = (price - l) / (u - l)
    if pos > 0.8:
        return "upper_band (overbought)"
    elif pos < 0.2:
        return "lower_band (oversold)"
    elif pos > 0.6:
        return "above_middle"
    elif pos < 0.4:
        return "below_middle"
    return "middle"


def compute_volatility(close: pd.Series, period: int = 20) -> str:
    """Volatility level from return std dev."""
    if len(close) < period:
        return "unknown"
    returns = close.pct_change()
    vol = returns.rolling(window=period).std().iloc[-1]
    if np.isnan(vol):
        return "unknown"
    annualized = vol * np.sqrt(252) * 100
    if annualized > 40:
        return "very_high"
    elif annualized > 25:
        return "high"
    elif annualized > 15:
        return "moderate"
    return "low"


def compute_momentum(close: pd.Series, period: int = 10) -> str:
    """Momentum over period."""
    if len(close) < period + 1:
        return "flat"
    change = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period] * 100
    if change > 3:
        return f"strong_up (+{change:.1f}%)"
    elif change > 1:
        return f"up (+{change:.1f}%)"
    elif change < -3:
        return f"strong_down ({change:.1f}%)"
    elif change < -1:
        return f"down ({change:.1f}%)"
    return f"flat ({change:+.1f}%)"


def compute_all_features(df: pd.DataFrame) -> dict:
    """Compute all features for a stock. Input: DataFrame with open,high,low,close,volume columns."""
    close = df["close"]
    volume = df["volume"]

    return {
        "rsi": compute_rsi(close),
        "macd": compute_macd(close),
        "volume_spike": compute_volume_spike(volume),
        "trend": compute_trend(close),
        "bollinger": compute_bollinger_position(close),
        "volatility": compute_volatility(close),
        "momentum": compute_momentum(close),
    }


def features_to_text(symbol: str, price: float, daily_change_pct: float, features: dict) -> str:
    """Convert features to human-readable text for LLM consumption."""
    rsi = features["rsi"]
    rsi_label = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"

    macd = features["macd"]
    macd_str = f"MACD: {macd['signal']}"
    if macd["crossover"]:
        macd_str += " (CROSSOVER)"

    vol_spike = features["volume_spike"]
    vol_label = "very high" if vol_spike > 2.0 else "high" if vol_spike > 1.5 else "normal" if vol_spike > 0.7 else "low"

    sign = "+" if daily_change_pct >= 0 else ""
    lines = [
        f"{symbol}: Rs{price:,.0f} ({sign}{daily_change_pct:.1f}% today)",
        f"  RSI: {rsi:.0f} ({rsi_label}) | {macd_str}",
        f"  Trend: {features['trend']} | Bollinger: {features['bollinger']}",
        f"  Volume: {vol_spike:.1f}x avg ({vol_label}) | Volatility: {features['volatility']}",
        f"  Momentum: {features['momentum']}",
    ]
    return "\n".join(lines)
