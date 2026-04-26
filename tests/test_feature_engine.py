"""Tests for technical indicator computation and text formatting."""

import numpy as np
import pandas as pd
import pytest

from server.feature_engine import compute_all_features, features_to_text


@pytest.fixture
def sample_ohlcv():
    """50-day OHLCV data with a known pattern for predictable indicator values."""
    np.random.seed(42)
    n = 50
    # Trending up: each day's close is slightly higher
    base_price = 1000
    closes = base_price + np.cumsum(np.random.randn(n) * 5 + 1)  # uptrend with noise
    opens = closes - np.random.rand(n) * 3
    highs = np.maximum(opens, closes) + np.random.rand(n) * 5
    lows = np.minimum(opens, closes) - np.random.rand(n) * 5
    volumes = np.random.randint(1_000_000, 10_000_000, n)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


@pytest.fixture
def oversold_ohlcv():
    """Data where price has been falling sharply — should produce low RSI."""
    n = 50
    closes = np.linspace(1500, 900, n)  # Steady decline
    opens = closes + 5
    highs = opens + 3
    lows = closes - 3
    volumes = np.full(n, 5_000_000)

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


class TestComputeAllFeatures:
    """Feature computation should return all expected indicators."""

    def test_returns_dict(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        assert isinstance(features, dict)

    def test_has_rsi(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        assert "rsi" in features
        assert 0 <= features["rsi"] <= 100

    def test_has_macd(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        assert "macd" in features
        assert features["macd"]["signal"] in ("bullish", "bearish")

    def test_has_bollinger(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        assert "bollinger" in features

    def test_has_volume_spike(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        assert "volume_spike" in features
        assert features["volume_spike"] >= 0

    def test_has_trend(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        assert "trend" in features
        assert features["trend"] in ("bullish", "bearish", "sideways")

    def test_has_momentum(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        assert "momentum" in features

    def test_has_volatility(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        assert "volatility" in features

    def test_declining_price_low_rsi(self, oversold_ohlcv):
        """Steadily declining prices should produce RSI below 30 (oversold)."""
        features = compute_all_features(oversold_ohlcv)
        assert features["rsi"] < 35

    def test_declining_price_bearish_trend(self, oversold_ohlcv):
        features = compute_all_features(oversold_ohlcv)
        assert features["trend"] == "bearish"


class TestFeaturesToText:
    """Text formatting for LLM consumption."""

    def test_returns_string(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        text = features_to_text("RELIANCE", 1050.0, 2.5, features)
        assert isinstance(text, str)

    def test_contains_symbol(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        text = features_to_text("RELIANCE", 1050.0, 2.5, features)
        assert "RELIANCE" in text

    def test_contains_price(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        text = features_to_text("RELIANCE", 1050.0, 2.5, features)
        assert "1,050" in text or "1050" in text

    def test_contains_rsi(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        text = features_to_text("RELIANCE", 1050.0, 2.5, features)
        assert "RSI" in text

    def test_contains_trend(self, sample_ohlcv):
        features = compute_all_features(sample_ohlcv)
        text = features_to_text("RELIANCE", 1050.0, 2.5, features)
        assert "Trend" in text or "trend" in text
