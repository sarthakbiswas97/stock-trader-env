"""Tests for action parsing logic."""

import pytest

from server.environment import StockTradingEnvironment
from server.tasks import TASK_CONFIGS


@pytest.fixture
def env_single():
    """Environment configured for single_stock task."""
    env = StockTradingEnvironment()
    env._task_config = TASK_CONFIGS["single_stock"]
    return env


@pytest.fixture
def env_portfolio():
    """Environment configured for portfolio task."""
    env = StockTradingEnvironment()
    env._task_config = TASK_CONFIGS["portfolio"]
    return env


class TestSingleStockParsing:
    """Single-stock task: bare BUY/SELL should resolve to the default symbol."""

    def test_hold(self, env_single):
        result = env_single._parse_action("HOLD")
        assert result == {"type": "HOLD"}

    def test_buy_bare(self, env_single):
        """Bare 'BUY' should default to the only symbol (RELIANCE)."""
        result = env_single._parse_action("BUY")
        assert result["type"] == "BUY"
        assert result["symbol"] == "RELIANCE"
        assert result["fraction"] == 1.0

    def test_sell_bare(self, env_single):
        """Bare 'SELL' should default to the only symbol."""
        result = env_single._parse_action("SELL")
        assert result["type"] == "SELL"
        assert result["symbol"] == "RELIANCE"

    def test_buy_with_symbol(self, env_single):
        """Explicit symbol should work even in single-stock mode."""
        result = env_single._parse_action("BUY RELIANCE")
        assert result["type"] == "BUY"
        assert result["symbol"] == "RELIANCE"

    def test_buy_with_fraction(self, env_single):
        result = env_single._parse_action("BUY RELIANCE 0.5")
        assert result["fraction"] == 0.5

    def test_case_insensitive(self, env_single):
        """Actions should be case-insensitive."""
        result = env_single._parse_action("buy reliance 0.3")
        assert result["type"] == "BUY"
        assert result["symbol"] == "RELIANCE"
        assert result["fraction"] == 0.3


class TestPortfolioParsing:
    """Multi-stock task: symbol is required for BUY/SELL."""

    def test_buy_requires_symbol(self, env_portfolio):
        """Bare 'BUY' without symbol should default to HOLD in multi-stock."""
        result = env_portfolio._parse_action("BUY")
        assert result["type"] == "HOLD"

    def test_buy_with_valid_symbol(self, env_portfolio):
        result = env_portfolio._parse_action("BUY INFY")
        assert result["type"] == "BUY"
        assert result["symbol"] == "INFY"

    def test_buy_invalid_symbol(self, env_portfolio):
        """Symbol not in task config should default to HOLD."""
        result = env_portfolio._parse_action("BUY GOOGL")
        assert result["type"] == "HOLD"

    def test_sell_with_fraction(self, env_portfolio):
        result = env_portfolio._parse_action("SELL TCS 0.5")
        assert result["type"] == "SELL"
        assert result["symbol"] == "TCS"
        assert result["fraction"] == 0.5


class TestEdgeCases:
    """Malformed inputs, boundary conditions."""

    def test_empty_string(self, env_single):
        result = env_single._parse_action("")
        assert result["type"] == "HOLD"

    def test_whitespace_only(self, env_single):
        result = env_single._parse_action("   ")
        assert result["type"] == "HOLD"

    def test_invalid_action_word(self, env_single):
        result = env_single._parse_action("YOLO RELIANCE")
        assert result["type"] == "HOLD"

    def test_fraction_clamped_above_one(self, env_single):
        """Fraction > 1.0 should be clamped to 1.0."""
        result = env_single._parse_action("BUY RELIANCE 5.0")
        assert result["fraction"] == 1.0

    def test_fraction_clamped_below_zero(self, env_single):
        """Negative fraction should be clamped to 0.0."""
        result = env_single._parse_action("BUY RELIANCE -0.5")
        assert result["fraction"] == 0.0

    def test_invalid_fraction_defaults_to_one(self, env_single):
        """Non-numeric fraction should default to 1.0."""
        result = env_single._parse_action("BUY RELIANCE abc")
        assert result["fraction"] == 1.0

    def test_extra_whitespace(self, env_single):
        """Extra whitespace should be handled."""
        result = env_single._parse_action("  BUY   RELIANCE   0.3  ")
        assert result["type"] == "BUY"
        assert result["symbol"] == "RELIANCE"
